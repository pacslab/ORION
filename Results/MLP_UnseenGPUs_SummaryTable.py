import os, re, sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

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
    print("Please install: pip install xgboost pandas numpy scikit-learn xgboost")
    sys.exit(1)

# ====================== CONFIG ======================
CSV_NAME = "MLP.csv"

TARGET_ORION  = "avg_batch_time_ms"
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
    "loader_wait_ms",
    "compute_ms",
    "p50_batch_ms",
    "batch_time_p50_ms",
    "std_batch_ms",
    "avg_loader_ms",
    "avg_compute_ms",
    "iter_time_ms",
    "iteration_time_ms",
    "batch_time_ms",
    "time_ms",
}

# ID / textual columns that should never be used as raw numeric features
ID_TXT_COLS_DEFAULT = {
    "gpu_name",
    "dataset",
    "input_shape",
    "optimizer",
    "model",
    "Bus",
    "Memory Type",
}

# Categorical base columns that will be one-hot encoded separately
# (and thus excluded as raw features)
CATEGORICAL_GPU_COLS = ["Memory Type", "Bus", "gpu_arch", "storage"]

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

    # One-hot encode categorical GPU/host fields, including storage
    for col in CATEGORICAL_GPU_COLS:
        if col in df.columns and df[col].notna().any():
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)

    return df

def add_hardware_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def infer_arch_family(name: str) -> str:
        # Normalize to upper for easier matching
        n = str(name).upper()

        # Hopper (e.g., "NVIDIA H100 PCIe")
        if "H100" in n or "H200" in n:
            return "Hopper"

        # Ada (L4, L40, L40S, RTX 5090, RTX 4000 Ada Generation, etc.)
        if (
            "L40S" in n
            or "L40" in n
            or "L4" in n
            or re.search(r"RTX\s*5\d{3}", n)  # e.g., RTX 5090
            or "RTX 4000" in n                # RTX 4000 Ada Generation
            or " ADA" in n                    # generic Ada hint in name
        ):
            return "Ada"

        # Ampere (A100 80GB PCIe, A40, A10, etc.)
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
        df["arch_family"] = df["gpu_name"].apply(infer_arch_family)
        df = pd.concat(
            [df, pd.get_dummies(df["arch_family"], prefix="arch", drop_first=True)],
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
        df["arch_family"] = "Unknown"
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

    # Extra peak FLOP ratios to training FLOPs
    for peak_col in ["FP16", "FP32"]:
        if peak_col in df.columns and "flops_train_per_batch" in df.columns:
            safe_add_ratio(df, peak_col, "flops_train_per_batch", f"{peak_col}_per_trainflop")

    return df

def add_mlp_computational_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    MLP-specific FLOPs features (mirrors add_cnn_computational_feature from K-fold script).
    """
    df = df.copy()
    if "flops_train_per_sample" in df.columns:
        df["F_MLP_sample"] = df["flops_train_per_sample"].astype(float)
    elif "flops_fwd_per_sample" in df.columns:
        df["F_MLP_sample"] = df["flops_fwd_per_sample"].astype(float) * 3.0
    elif "flops_train_per_batch" in df.columns and "batch_size" in df.columns:
        df["F_MLP_sample"] = df["flops_train_per_batch"].astype(float) / np.maximum(
            1, df["batch_size"].astype(float)
        )
    else:
        df["F_MLP_sample"] = np.nan

    if "flops_train_per_batch" in df.columns:
        df["F_MLP_batch"] = df["flops_train_per_batch"].astype(float)
    else:
        df["F_MLP_batch"] = df["F_MLP_sample"] * df.get("batch_size", 1).astype(float)

    df["log_F_MLP_sample"] = np.log1p(df["F_MLP_sample"].astype(float))
    df["log_F_MLP_batch"] = np.log1p(df["F_MLP_batch"].astype(float))
    return df

def add_batch_aware_hardware_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    B = df["batch_size"].astype(float)

    # Batch features
    df["log2_batch_size"] = np.log2(np.maximum(1.0, B))
    df["batch_bin_small"] = (B <= 32).astype(int)
    df["batch_bin_medium"] = ((B > 32) & (B <= 128)).astype(int)
    df["batch_bin_large"] = (B > 128).astype(int)

    # Hardware fields
    sm = df.get("SM", df.get("sm_count", pd.Series(np.nan, index=df.index))).astype(float)
    mem_gb = df.get("Memory (GB)", pd.Series(np.nan, index=df.index)).astype(float)
    bw_gbs = df.get("GPU Memory Bandwidth (GB/s)", pd.Series(np.nan, index=df.index)).astype(float)
    vram_bytes = df.get("total_vram_bytes", pd.Series(np.nan, index=df.index)).astype(float)

    # Interactions between batch and hardware
    df["B_per_SM"] = B / np.maximum(1.0, sm.replace(0, np.nan)).fillna(1.0)
    df["B_over_GB"] = B / np.maximum(1.0, mem_gb.replace(0, np.nan)).fillna(1.0)
    df["B_times_BW"] = B * bw_gbs.fillna(bw_gbs.median() if bw_gbs.notna().any() else 0.0)
    df["B_times_SM"] = B * sm.fillna(sm.median() if sm.notna().any() else 0.0)

    # Memory pressure proxies
    param_bytes = df.get("param_bytes", pd.Series(0, index=df.index)).astype(float)
    total_state_bytes = df.get("total_state_bytes", pd.Series(param_bytes, index=df.index)).astype(float)
    act_bytes_per_sample = df.get("act_bytes_per_sample_proxy", pd.Series(0, index=df.index)).astype(float)
    act_bytes_per_batch = act_bytes_per_sample * B
    total_required_bytes = total_state_bytes + act_bytes_per_batch
    if vram_bytes.isna().all():
        vram_bytes = (mem_gb.fillna(0.0) * (1024 ** 3))
    df["mem_pressure"] = (total_required_bytes / np.maximum(1.0, vram_bytes)).astype(float)
    df["mem_headroom"] = clip01(1.0 - df["mem_pressure"])
    df["near_vram_limit"] = (df["mem_pressure"] > 0.90).astype(int)

    # AMP and tensor core utilization proxy
    if "amp" in df.columns:
        df["amp"] = (
            df["amp"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .astype(int)
        )
    amp = df.get("amp", pd.Series(0, index=df.index)).astype(int)
    has_tc = df.get("has_tensor_cores", pd.Series(0, index=df.index)).astype(int)
    df["tc_util_proxy"] = (amp * has_tc).astype(int)

    # Roofline-style compute vs memory time proxy
    fp16 = df.get("FP16", pd.Series(np.nan, index=df.index)).astype(float)
    fp32 = df.get("FP32", pd.Series(np.nan, index=df.index)).astype(float)
    peak_flops = np.where(amp.values.astype(bool) & fp16.notna(), fp16.fillna(0.0), fp32.fillna(0.0))
    flops_btch = df.get("flops_train_per_batch", pd.Series(np.nan, index=df.index)).astype(float)
    t_compute_ms = (flops_btch / np.maximum(1e-9, peak_flops)) * 1000.0

    bw_bytes = (bw_gbs * (1024 ** 3)).fillna(0.0)
    t_memory_ms = (act_bytes_per_batch / np.maximum(1.0, bw_bytes)) * 1000.0
    roof_ms = np.maximum(t_compute_ms, t_memory_ms)
    df["roof_time_ms"] = np.nan_to_num(roof_ms, nan=np.nanmedian(roof_ms))

    # More interactions
    df["B_times_tc"] = B * df["tc_util_proxy"]
    df["B_times_headroom"] = B * df["mem_headroom"]
    df["B_times_arch_Ampere"] = B * df.get("arch_Ampere", pd.Series(0, index=df.index))
    df["B_times_arch_Ada"] = B * df.get("arch_Ada", pd.Series(0, index=df.index))
    df["B_times_arch_Hopper"] = B * df.get("arch_Hopper", pd.Series(0, index=df.index))

    return df

def auto_log1p_heavy_tails(df: pd.DataFrame, target: str, extra_drop: set) -> Tuple[pd.DataFrame, List[str]]:
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
        if s.nunique() <= 10 and set(np.unique(s)).issubset({0.0, 1.0}):
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
    StandardScaler + XGBoost, with basic cleanup to avoid numerical issues.
    """
    Xtr = Xtr.copy()
    Xte = Xte.copy()

    # Replace inf with NaN first
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.replace([np.inf, -np.inf], np.nan)

    # For each column, drop if all NaN in train; otherwise fill with train median
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

# ---------- LOGO: unseen-GPU evaluation with model/vcpu/storage/batch breakdown ----------
def logo_unseen_gpu_vcpu_storage_ORION_prenet(
    df: pd.DataFrame,
    X_ORION: pd.DataFrame,
    y_ORION: np.ndarray,
    X_prenet: pd.DataFrame,
    y_prenet: np.ndarray,
    feat_cols_ORION: List[str],
    feat_cols_prenet: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Leave-One-GPU-Out:
      - For each GPU, train:
          * ORION model on (X_ORION, y_ORION = avg_batch_time_ms)
          * PreNeT model on (X_prenet, y_prenet = T_gpu_ms) with NO vcpu/storage features
      - Evaluate both vs actual avg_batch_time_ms on held-out GPU.
      - Return:
          * summary_df: one row per GPU with RMSE for ORION and PreNeT
          * tables: gpu_name -> {
                "vcpu":    per-(model, vcpu)       table,
                "storage": per-(model, storage)    table,
                "batch":   per-(model, batch_size) table
            }
    """
    if "gpu_name" not in df.columns:
        raise ValueError("Expected 'gpu_name' column for LOGO.")
    if "vcpu" not in df.columns or "storage" not in df.columns:
        raise ValueError("Expected 'vcpu' and 'storage' columns for LOGO breakdown.")
    if "batch_size" not in df.columns:
        raise ValueError("Expected 'batch_size' column for batch-size breakdown.")
    if "model" not in df.columns:
        raise ValueError("Expected 'model' column for per-model breakdown.")

    gpus = sorted(df["gpu_name"].dropna().unique().tolist())
    summary_rows = []
    tables: Dict[str, Dict[str, pd.DataFrame]] = {}

    y_true_all = df[TARGET_ORION].astype(float).values  # ground-truth avg_batch_time_ms

    for gpu in gpus:
        mask = (df["gpu_name"] == gpu)
        if mask.sum() < 3:
            # too few samples to be meaningful
            continue

        # ORION: train on avg_batch_time_ms, eval vs avg_batch_time_ms
        Xtr_n, Xte_n = X_ORION.loc[~mask, feat_cols_ORION], X_ORION.loc[mask, feat_cols_ORION]
        ytr_n, yte_true = y_ORION[~mask], y_true_all[mask]
        yhat_n = fit_predict_with_scaling(Xtr_n, ytr_n, Xte_n)

        # PreNeT: train on T_gpu_ms, eval vs avg_batch_time_ms (no host features)
        Xtr_p, Xte_p = X_prenet.loc[~mask, feat_cols_prenet], X_prenet.loc[mask, feat_cols_prenet]
        ytr_p = y_prenet[~mask]
        yhat_p = fit_predict_with_scaling(Xtr_p, ytr_p, Xte_p)

        # Base sub-DataFrame for this GPU with predictions
        sub = df.loc[mask, ["model", "vcpu", "storage", "batch_size", TARGET_ORION]].copy()
        sub = sub.assign(
            Pred_ORION_ms=yhat_n.astype(float),
            Pred_PreNeT_ms=yhat_p.astype(float),
            Actual_ms=sub[TARGET_ORION].astype(float),
        )

        def build_group_table(group_cols: List[str]) -> pd.DataFrame:
            rows = []
            for keys, g in sub.groupby(group_cols):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                key_dict = {col: val for col, val in zip(group_cols, keys)}

                y_true_g = g["Actual_ms"].values
                y_ORION_g = g["Pred_ORION_ms"].values
                y_prenet_g = g["Pred_PreNeT_ms"].values

                act_mean = float(np.mean(y_true_g))
                pred_ORION_mean = float(np.mean(y_ORION_g))
                pred_prenet_mean = float(np.mean(y_prenet_g))

                # Signed percentage error based on mean values
                if act_mean != 0.0:
                    pct_err_ORION = 100.0 * (pred_ORION_mean - act_mean) / act_mean
                    pct_err_prenet = 100.0 * (pred_prenet_mean - act_mean) / act_mean
                else:
                    pct_err_ORION = np.nan
                    pct_err_prenet = np.nan

                rmse_ORION = rmse(y_true_g, y_ORION_g)
                rmse_prenet = rmse(y_true_g, y_prenet_g)

                row = {
                    **key_dict,
                    "n_samples": int(len(g)),
                    "Actual_ms_mean": act_mean,
                    "Pred_ORION_ms_mean": pred_ORION_mean,
                    "ORION_pct_error": pct_err_ORION,
                    "RMSE_ORION_ms": rmse_ORION,
                    "Pred_PreNeT_ms_mean": pred_prenet_mean,
                    "PreNeT_pct_error": pct_err_prenet,
                    "RMSE_PreNeT_ms": rmse_prenet,
                }
                rows.append(row)

            tab = pd.DataFrame(rows)
            # Sort columns by group keys for readability
            sort_cols = [c for c in group_cols if c in tab.columns]
            if sort_cols:
                tab = tab.sort_values(sort_cols).reset_index(drop=True)
            return tab

        # Three tables: varying vcpu, storage, batch size for each model
        tab_vcpu = build_group_table(["model", "vcpu"])
        tab_storage = build_group_table(["model", "storage"])
        tab_batch = build_group_table(["model", "batch_size"])

        # GPU-level RMSE (across all rows for this GPU)
        rmse_gpu_ORION = rmse(yte_true, yhat_n)
        rmse_gpu_prenet = rmse(yte_true, yhat_p)
        improvement_pct = (
            100.0 * (rmse_gpu_prenet - rmse_gpu_ORION) / rmse_gpu_prenet
            if rmse_gpu_prenet > 0
            else np.nan
        )

        summary_rows.append(
            {
                "GPU": gpu,
                "n_samples": int(mask.sum()),
                "RMSE_ORION_ms": rmse_gpu_ORION,
                "RMSE_PreNeT_ms": rmse_gpu_prenet,
                "Improvement_ORION_vs_PreNeT_%": improvement_pct,
            }
        )
        tables[gpu] = {
            "vcpu": tab_vcpu,
            "storage": tab_storage,
            "batch": tab_batch,
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE_ORION_ms").reset_index(drop=True)
    return summary_df, tables

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
        "vcpu",
        "storage",
        TARGET_ORION,
        TARGET_PRENET,
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in CSV: {missing}")

    # Drop rows without both targets
    df = df.dropna(subset=[TARGET_ORION, TARGET_PRENET]).copy()

    # ---------- Feature engineering ----------
    df = engineer_gpu_model_features(df)
    df = add_hardware_flags(df)
    df = add_mlp_computational_feature(df)
    df = add_batch_aware_hardware_interactions(df)

    # Heavy-tail logs (using ORION target as main target)
    df, _ = auto_log1p_heavy_tails(df, target=TARGET_ORION, extra_drop=set())

    # ---------- Build feature matrices ----------
    # ORION: full feature set (including vcpu/storage-related features)
    X_ORION, y_ORION, feats_ORION = build_feature_matrix(
        df,
        target=TARGET_ORION,
        include_model_oh=INCLUDE_MODEL_OH,
        include_opt_oh=INCLUDE_OPT_OH,
    )

    # PreNeT: drop any vcpu/storage-related features
    host_exclude = [
        c
        for c in df.columns
        if ("vcpu" in c.lower()) or ("storage" in c.lower())
    ]
    X_prenet, y_prenet, feats_prenet = build_feature_matrix(
        df,
        target=TARGET_PRENET,
        include_model_oh=INCLUDE_MODEL_OH,
        include_opt_oh=INCLUDE_OPT_OH,
        extra_exclude_cols=host_exclude,
    )

    # ---------- LOGO unseen-GPU evaluation ----------
    summary_df, tables = logo_unseen_gpu_vcpu_storage_ORION_prenet(
        df=df,
        X_ORION=X_ORION,
        y_ORION=y_ORION,
        X_prenet=X_prenet,
        y_prenet=y_prenet,
        feat_cols_ORION=feats_ORION,
        feat_cols_prenet=feats_prenet,
    )

    # Pretty print per-GPU tables
    for gpu, tab_dict in tables.items():
        print(f"\n================ GPU: {gpu} ================")

        # 1) Varying vCPUs per model
        print("\n--- Per-(model, vcpu) breakdown ---")
        tab_vcpu = tab_dict["vcpu"].copy()
        for c in [
            "Actual_ms_mean",
            "Pred_ORION_ms_mean",
            "ORION_pct_error",
            "RMSE_ORION_ms",
            "Pred_PreNeT_ms_mean",
            "PreNeT_pct_error",
            "RMSE_PreNeT_ms",
        ]:
            if c in tab_vcpu.columns:
                tab_vcpu[c] = tab_vcpu[c].map(lambda v: f"{v:.3f}" if pd.notnull(v) else "nan")
        print(tab_vcpu.to_string(index=False))

        # 2) Varying storage per model
        print("\n--- Per-(model, storage) breakdown ---")
        tab_storage = tab_dict["storage"].copy()
        for c in [
            "Actual_ms_mean",
            "Pred_ORION_ms_mean",
            "ORION_pct_error",
            "RMSE_ORION_ms",
            "Pred_PreNeT_ms_mean",
            "PreNeT_pct_error",
            "RMSE_PreNeT_ms",
        ]:
            if c in tab_storage.columns:
                tab_storage[c] = tab_storage[c].map(lambda v: f"{v:.3f}" if pd.notnull(v) else "nan")
        print(tab_storage.to_string(index=False))

        # 3) Varying batch size per model
        print("\n--- Per-(model, batch_size) breakdown ---")
        tab_batch = tab_dict["batch"].copy()
        for c in [
            "Actual_ms_mean",
            "Pred_ORION_ms_mean",
            "ORION_pct_error",
            "RMSE_ORION_ms",
            "Pred_PreNeT_ms_mean",
            "PreNeT_pct_error",
            "RMSE_PreNeT_ms",
        ]:
            if c in tab_batch.columns:
                tab_batch[c] = tab_batch[c].map(lambda v: f"{v:.3f}" if pd.notnull(v) else "nan")
        print(tab_batch.to_string(index=False))

        # GPU-level RMSE line
        row = summary_df.loc[summary_df["GPU"] == gpu].iloc[0]
        print(
            "\nGPU-level RMSE — "
            f"ORION: {row['RMSE_ORION_ms']:.3f} ms, "
            f"PreNeT: {row['RMSE_PreNeT_ms']:.3f} ms, "
            f"Improvement (ORION vs PreNeT): "
            f"{row['Improvement_ORION_vs_PreNeT_%']:.2f}% "
            f"(n={row['n_samples']})"
        )

    # Summary table
    if not summary_df.empty:
        print("\n=== LOGO Summary per GPU (ORION vs PreNeT) ===")
        fmt_sum = summary_df.copy()
        for c in ["RMSE_ORION_ms", "RMSE_PreNeT_ms", "Improvement_ORION_vs_PreNeT_%"]:
            fmt_sum[c] = fmt_sum[c].map(lambda v: f"{v:.3f}" if pd.notnull(v) else "nan")
        print(fmt_sum.to_string(index=False))

if __name__ == "__main__":
    main()
