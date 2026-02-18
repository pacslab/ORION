# Table 3 Results Evaluation

import os, re, sys
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---- RMSE without deprecation warnings ----
try:
    from sklearn.metrics import root_mean_squared_error as _sk_rmse

    def rmse(y_true, y_pred):
        return float(_sk_rmse(y_true, y_pred))
except Exception:
    def rmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


try:
    import xgboost as xgb
except Exception:
    print("Please install: pip install xgboost pandas numpy scikit-learn")
    sys.exit(1)

# ====================== CONFIG ======================
CSV_NAME = "Transformer.csv"

# ORION uses avg_batch_time_ms as target
TARGET_ORION = "avg_batch_time_ms"

# PreNeT baseline uses T_gpu_ms as training target,
# but will be evaluated vs avg_batch_time_ms.
TARGET_PRENET = "T_gpu_ms"

INCLUDE_MODEL_OH = False
INCLUDE_OPT_OH   = False

# Columns whose presence would leak "true time" into features
LEAKY_COLS = {
    "avg_batch_time_ms",
    "p90_batch_time_ms",
    "T_gpu_ms",
    "T_cpu_io_ms",
    "T_step_ms",
    "loader_ratio",
    # Throughput-like quantities are also derived from time
    "examples_per_sec",
    "tokens_per_sec",
}

# ID / textual columns that should never be used as raw numeric features
ID_TXT_COLS_DEFAULT = {
    "gpu_name",
    "dataset",
    "optimizer",
    "model",
    "architecture",
    "arch_family",
    "activation",
    "norm_style",
    "norm_impl",
    "positional_encoding",
    "attention_pattern",
    "bottleneck",
    "Memory Type",
    "Bus",
}

# Categorical base columns that will be one-hot encoded separately
# (and thus excluded as raw features)
CATEGORICAL_GPU_COLS = ["Memory Type", "Bus", "storage"]

# ====================== Utilities ======================
def clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))

def safe_add_ratio(df: pd.DataFrame, num: str, den: str, out: str):
    if num in df.columns and den in df.columns:
        df[out] = df[num].astype(float) / np.maximum(1e-9, df[den].astype(float))

def _contains_any(s: str, keys) -> bool:
    try:
        s_low = str(s).lower()
    except Exception:
        return False
    return any(k in s_low for k in keys)

# ====================== Feature Engineering ======================
def engineer_gpu_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FLOPs / bandwidth / peak FLOPs ratios + one-hot GPU/storage.
    """
    df = df.copy()

    # FLOPs / bandwidth / peak FLOPs ratios
    if "flops_train_per_batch" in df.columns and "GPU Memory Bandwidth (GB/s)" in df.columns:
        safe_add_ratio(df, "flops_train_per_batch", "GPU Memory Bandwidth (GB/s)", "flops_to_bw_ratio")
    for peak_col in ["FP32", "FP16", "FP64"]:
        if peak_col in df.columns and "flops_train_per_batch" in df.columns:
            safe_add_ratio(df, "flops_train_per_batch", peak_col, f"flops_to_{peak_col}_ratio")

    if "GPU Memory Bandwidth (GB/s)" in df.columns and "Memory (GB)" in df.columns:
        safe_add_ratio(df, "GPU Memory Bandwidth (GB/s)", "Memory (GB)", "bw_per_gb")

    if "arithmetic_intensity_train" in df.columns and "FP32" in df.columns:
        safe_add_ratio(df, "arithmetic_intensity_train", "FP32", "ai_over_fp32")

    # One-hot encode base categorical GPU/host fields, including storage
    for col in CATEGORICAL_GPU_COLS:
        if col in df.columns and df[col].notna().any():
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)

    return df

def add_hardware_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive coarse-grained GPU-architecture flags from gpu_name.
    """
    df = df.copy()

    def infer_arch_family(name: str) -> str:
        n = str(name).upper()
        # Hopper
        if "H100" in n or "H200" in n:
            return "Hopper"
        # Ada: L4, L40, L40S, RTX 5090, RTX 4000 Ada Generation, etc.
        if (
            "L40S" in n
            or "L40" in n
            or "L4" in n
            or re.search(r"RTX\s*5\d{3}", n)  # e.g., RTX 5090
            or "RTX 4000" in n                # RTX 4000 Ada Generation
            or " ADA" in n                    # generic Ada indicator
        ):
            return "Ada"
        # Ampere
        if "A100" in n or "A40" in n or "A10" in n:
            return "Ampere"
        # Turing
        if "V100" in n or "T4" in n:
            return "Turing"
        # Pascal
        if "P100" in n or "P4" in n:
            return "Pascal"
        return "Unknown"

    if "gpu_name" in df.columns:
        df["arch_family_inferred"] = df["gpu_name"].apply(infer_arch_family)
        df = pd.concat(
            [df, pd.get_dummies(df["arch_family_inferred"], prefix="arch", drop_first=True)],
            axis=1,
        )
        df["has_tensor_cores"] = df["gpu_name"].apply(
            lambda s: int(
                _contains_any(
                    s,
                    ["a100", "a40", "a10", "l40", "l4", "h100", "rtx 50", "rtx 5090", "v100", "t4"],
                )
            )
        ).astype(int)
        df["is_pcie"] = df["gpu_name"].astype(str).str.contains(
            "PCIE|PCIe|PCI-E", case=False, na=False
        ).astype(int)
        df["is_sxm"] = df["gpu_name"].astype(str).str.contains(
            "SXM", case=False, na=False
        ).astype(int)
        df["is_datacenter"] = df["gpu_name"].apply(
            lambda s: int(
                _contains_any(
                    s,
                    ["a100", "h100", "v100", "a40", "l40", "l4", "a10", "t4", "p100", "p4"],
                )
            )
        ).astype(int)
    else:
        df["arch_family_inferred"] = "Unknown"
        df["has_tensor_cores"] = 0
        df["is_pcie"] = 0
        df["is_sxm"] = 0
        df["is_datacenter"] = 0

    if "Memory Type" in df.columns:
        df["has_HBM"] = df["Memory Type"].astype(str).str.contains(
            "HBM", case=False, na=False
        ).astype(int)
    else:
        df["has_HBM"] = 0

    # Extra peak FLOP ratios to training FLOPs (if peak columns exist)
    for peak_col in ["FP16", "FP32"]:
        if peak_col in df.columns and "flops_train_per_batch" in df.columns:
            safe_add_ratio(df, peak_col, "flops_train_per_batch", f"{peak_col}_per_trainflop")

    return df

def add_transformer_computational_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformer-specific FLOPs features (per sequence and per batch).
    """
    df = df.copy()

    # Per-sequence FLOPs
    if "flops_train_per_seq" in df.columns:
        df["F_TR_seq"] = df["flops_train_per_seq"].astype(float)
    else:
        df["F_TR_seq"] = np.nan

    # Per-batch FLOPs
    if "flops_train_per_batch" in df.columns:
        df["F_TR_batch"] = df["flops_train_per_batch"].astype(float)
    else:
        df["F_TR_batch"] = df["F_TR_seq"] * df.get("batch_size", 1).astype(float)

    df["log_F_TR_seq"] = np.log1p(df["F_TR_seq"].astype(float))
    df["log_F_TR_batch"] = np.log1p(df["F_TR_batch"].astype(float))

    # Per-token / per-sample proxies from seq_len / tgt_len
    seq_len = df.get("seq_len", pd.Series(np.nan, index=df.index)).astype(float)
    src_len = df.get("src_len", pd.Series(np.nan, index=df.index)).astype(float)
    tgt_len = df.get("tgt_len", pd.Series(np.nan, index=df.index)).astype(float)

    eff_len = seq_len.copy()
    eff_len = eff_len.fillna(tgt_len).fillna(src_len)
    df["eff_seq_len"] = eff_len
    df["log_eff_seq_len"] = np.log1p(
        eff_len.replace({np.inf: np.nan}).fillna(
            eff_len.median() if eff_len.notna().any() else 1.0
        )
    )

    if "flops_train_per_batch" in df.columns and "batch_size" in df.columns:
        df["F_TR_sample"] = df["flops_train_per_batch"].astype(float) / np.maximum(
            1.0, df["batch_size"].astype(float)
        )
        df["log_F_TR_sample"] = np.log1p(df["F_TR_sample"].astype(float))
    else:
        df["F_TR_sample"] = np.nan
        df["log_F_TR_sample"] = np.nan

    return df

def add_batch_aware_hardware_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch-size and hardware interaction features.
    """
    df = df.copy()
    B = df["batch_size"].astype(float)

    # Batch features
    df["log2_batch_size"] = np.log2(np.maximum(1.0, B))
    df["batch_bin_small"] = (B <= 32).astype(int)
    df["batch_bin_medium"] = ((B > 32) & (B <= 128)).astype(int)
    df["batch_bin_large"] = (B > 128).astype(int)

    # Hardware fields
    sm = df.get("SM", pd.Series(np.nan, index=df.index)).astype(float)
    mem_gb = df.get("Memory (GB)", pd.Series(np.nan, index=df.index)).astype(float)
    bw_gbs = df.get("GPU Memory Bandwidth (GB/s)", pd.Series(np.nan, index=df.index)).astype(float)

    # Interactions between batch and hardware
    df["B_per_SM"] = B / np.maximum(1.0, sm.replace(0, np.nan)).fillna(1.0)
    df["B_over_GB"] = B / np.maximum(1.0, mem_gb.replace(0, np.nan)).fillna(1.0)
    df["B_times_BW"] = B * bw_gbs.fillna(bw_gbs.median() if bw_gbs.notna().any() else 0.0)
    df["B_times_SM"] = B * sm.fillna(sm.median() if sm.notna().any() else 0.0)

    # Memory pressure proxies
    param_bytes = df.get("param_bytes", pd.Series(0, index=df.index)).astype(float)
    total_state_bytes = df.get("total_state_bytes", pd.Series(param_bytes, index=df.index)).astype(float)
    act_bytes_per_seq = df.get("act_bytes_per_seq_proxy", pd.Series(0, index=df.index)).astype(float)
    act_bytes_per_batch = act_bytes_per_seq * B
    total_required_bytes = total_state_bytes + act_bytes_per_batch

    vram_bytes = mem_gb.fillna(0.0) * (1024 ** 3)
    df["mem_pressure"] = (total_required_bytes / np.maximum(1.0, vram_bytes)).astype(float)
    df["mem_headroom"] = clip01(1.0 - df["mem_pressure"])
    df["near_vram_limit"] = (df["mem_pressure"] > 0.90).astype(int)

    # AMP and tensor core utilization proxy
    if "amp_enabled" in df.columns:
        df["amp"] = (
            df["amp_enabled"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .astype(int)
        )
    else:
        df["amp"] = 0
    amp = df["amp"].astype(int)
    has_tc = df.get("has_tensor_cores", pd.Series(0, index=df.index)).astype(int)
    df["tc_util_proxy"] = (amp * has_tc).astype(int)

    # Roofline-style compute vs memory time proxy (if peak hw cols exist)
    fp16 = df.get("FP16", pd.Series(np.nan, index=df.index)).astype(float)
    fp32 = df.get("FP32", pd.Series(np.nan, index=df.index)).astype(float)
    peak_flops = np.where(amp.values.astype(bool) & fp16.notna(), fp16.fillna(0.0), fp32.fillna(0.0))
    flops_btch = df.get("flops_train_per_batch", pd.Series(np.nan, index=df.index)).astype(float)
    t_compute_ms = (flops_btch / np.maximum(1e-9, peak_flops)) * 1000.0

    bw_bytes = (bw_gbs * (1024 ** 3)).fillna(0.0)
    t_memory_ms = (act_bytes_per_batch / np.maximum(1.0, bw_bytes)) * 1000.0
    roof_ms = np.maximum(t_compute_ms, t_memory_ms)
    if np.isfinite(roof_ms).any():
        df["roof_time_ms"] = np.nan_to_num(roof_ms, nan=np.nanmedian(roof_ms))
    else:
        df["roof_time_ms"] = 0.0

    # More interactions
    df["B_times_tc"] = B * df["tc_util_proxy"]
    df["B_times_headroom"] = B * df["mem_headroom"]
    df["B_times_arch_Ampere"] = B * df.get("arch_Ampere", pd.Series(0, index=df.index))
    df["B_times_arch_Ada"] = B * df.get("arch_Ada", pd.Series(0, index=df.index))
    df["B_times_arch_Hopper"] = B * df.get("arch_Hopper", pd.Series(0, index=df.index))

    return df

def auto_log1p_heavy_tails(df: pd.DataFrame, target: str, extra_drop: set) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add log1p-transformed versions of heavily skewed non-negative numeric features.
    """
    df = df.copy()
    add_cols = []
    candidates = [
        c
        for c in df.columns
        if c not in extra_drop and c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in candidates:
        if c in LEAKY_COLS:
            continue
        s = df[c].astype(float)
        if (s < 0).any():
            continue
        # Skip pure binary 0/1 features
        if s.nunique() <= 10 and set(np.unique(s)) <= {0.0, 1.0}:
            continue
        skew_val = s.skew()
        if np.isfinite(skew_val) and skew_val > 1.0:
            newc = f"log1p__{c}"
            df[newc] = np.log1p(s)
            add_cols.append(newc)
    return df, add_cols

# ====================== Matrix build / modeling ======================
def build_feature_matrix(
    df: pd.DataFrame,
    target: str,
    include_model_oh: bool,
    include_opt_oh: bool,
    extra_exclude_cols: List[str] = None,
):
    df = df.copy()
    # Exclude target, explicit leaky time columns, and base categorical columns
    exclude = set([target]) | LEAKY_COLS | set(CATEGORICAL_GPU_COLS)
    id_txt_cols = set(ID_TXT_COLS_DEFAULT)

    # Optional one-hot encoding for model and optimizer
    if include_model_oh and "model" in df.columns:
        d = pd.get_dummies(df["model"].astype(str), prefix="model", drop_first=True)
        df = pd.concat([df, d], axis=1)
        id_txt_cols.discard("model")
    if include_opt_oh and "optimizer" in df.columns:
        d = pd.get_dummies(df["optimizer"].astype(str), prefix="opt", drop_first=True)
        df = pd.concat([df, d], axis=1)
        id_txt_cols.discard("optimizer")

    exclude |= id_txt_cols
    if extra_exclude_cols:
        exclude |= set(extra_exclude_cols)

    feat_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feat_cols:
        raise ValueError("No numeric feature columns after filtering.")
    X = df[feat_cols].copy()
    y = df[target].astype(float).values
    return X, y, feat_cols

def fit_predict_with_scaling(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame):
    """
    StandardScaler + XGBoost, with NaN/inf cleanup and dropping all-NaN cols.
    """
    Xtr = Xtr.copy()
    Xte = Xte.copy()

    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.replace([np.inf, -np.inf], np.nan)

    drop_cols = []
    for col in Xtr.columns:
        col_train = Xtr[col]
        non_na = col_train.notna().sum()
        if non_na == 0:
            drop_cols.append(col)
        else:
            med = col_train.median()
            if not np.isfinite(med):
                med = 0.0
            Xtr[col] = col_train.fillna(med)
            if col in Xte.columns:
                Xte[col] = Xte[col].fillna(med)

    if drop_cols:
        Xtr = Xtr.drop(columns=drop_cols)
        Xte = Xte.drop(columns=[c for c in drop_cols if c in Xte.columns])

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr.values)
    Xte_s = scaler.transform(Xte.values)

    reg = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.045,
        max_depth=9,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    reg.fit(Xtr_s, ytr)
    yhat = reg.predict(Xte_s)
    return yhat

# ====================== K-fold evaluation ======================
def kfold_ORION_vs_prenet(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    Returns a DataFrame with RMSE for ORION and PreNeT
    per model (aggregated across vcpu and storage), plus
    % improvement of ORION.

    ORION: full feature set (including vcpu/storage-related features).
    PreNeT: excludes any feature whose name contains 'vcpu' or 'storage'.
    """

    # Feature engineering
    df = engineer_gpu_model_features(df)
    df = add_hardware_flags(df)
    df = add_transformer_computational_feature(df)
    df = add_batch_aware_hardware_interactions(df)

    # Heavy-tail logs with ORION target as "main" target for transformations
    df, _ = auto_log1p_heavy_tails(df, target=TARGET_ORION, extra_drop=set())

    # ---------- Build feature matrices ----------
    # ORION: full feature set
    X_ORION, y_ORION, feats_ORION = build_feature_matrix(
        df,
        TARGET_ORION,
        include_model_oh=INCLUDE_MODEL_OH,
        include_opt_oh=INCLUDE_OPT_OH,
    )

    # PreNeT: drop any vcpu/storage-related features (host-agnostic)
    host_exclude = [
        c
        for c in df.columns
        if ("vcpu" in c.lower()) or ("storage" in c.lower())
    ]
    X_prenet, y_prenet, feats_prenet = build_feature_matrix(
        df,
        TARGET_PRENET,
        include_model_oh=INCLUDE_MODEL_OH,
        include_opt_oh=INCLUDE_OPT_OH,
        extra_exclude_cols=host_exclude,
    )

    N = len(df)
    preds_ORION = np.zeros(N, dtype=float)
    preds_prenet = np.zeros(N, dtype=float)
    y_true_avg = y_ORION.copy()  # ground-truth "avg_batch_time_ms"

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X_ORION):
        # ORION: train & eval on avg_batch_time_ms
        Xtr_ORION = X_ORION.iloc[train_idx]
        Xte_ORION = X_ORION.iloc[test_idx]
        ytr_ORION = y_ORION[train_idx]
        yhat_ORION = fit_predict_with_scaling(Xtr_ORION, ytr_ORION, Xte_ORION)
        preds_ORION[test_idx] = yhat_ORION

        # PreNeT: train on T_gpu_ms, eval vs avg_batch_time_ms (no host features)
        Xtr_pre = X_prenet.iloc[train_idx]
        Xte_pre = X_prenet.iloc[test_idx]
        ytr_pre = y_prenet[train_idx]
        yhat_pre_gpu_ms = fit_predict_with_scaling(Xtr_pre, ytr_pre, Xte_pre)
        preds_prenet[test_idx] = yhat_pre_gpu_ms

    # ---- RMSE per model across all vcpus and storage tiers ----
    models = df["model"].astype(str)
    rows = []
    for m in sorted(models.unique()):
        mask = (models == m)
        if mask.sum() < 2:
            continue

        y = y_true_avg[mask]
        yhat_n = preds_ORION[mask]
        yhat_p = preds_prenet[mask]

        rmse_n = rmse(y, yhat_n)
        rmse_p = rmse(y, yhat_p)

        if rmse_p > 0:
            improvement_pct = 100.0 * (rmse_p - rmse_n) / rmse_p
        else:
            improvement_pct = np.nan

        rows.append(
            {
                "model": m,
                "RMSE_ORION_ms": rmse_n,
                "RMSE_PreNeT_ms": rmse_p,
                "Improvement_%": improvement_pct,
            }
        )

    result = pd.DataFrame(rows).sort_values(["model"]).reset_index(drop=True)
    return result

# ====================== Main ======================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, CSV_NAME)
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {CSV_NAME} not found in {here}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {CSV_NAME}")

    required_cols = {
        "gpu_name",
        "batch_size",
        "model",
        TARGET_ORION,
        TARGET_PRENET,
        "vcpu",
        "storage",
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in CSV: {missing}")

    df = df.dropna(subset=[TARGET_ORION, TARGET_PRENET]).copy()

    summary = kfold_ORION_vs_prenet(df, n_splits=5)

    if summary.empty:
        print("No models with sufficient samples for RMSE computation.")
    else:
        summary_fmt = summary.copy()
        for col in ["RMSE_ORION_ms", "RMSE_PreNeT_ms", "Improvement_%"]:
            summary_fmt[col] = summary_fmt[col].map(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "nan"
            )

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("\n=== Transformer RMSE per model (aggregated across vCPU & storage) ===")
            print(summary_fmt.to_string(index=False))

if __name__ == "__main__":
    main()
