# Interactive_Figure_Adjustable.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
# ---- USER SETTINGS ----
# =========================

# This is the code to generate figures 5-7
# Figure 5 = MLP_ModelbyModel.csv
# Figure 6 = CNN_ModelbyModel.csv
# Figure 7 = Transformer_ModelbyModel.csv
# The data in the csv files are gathered from the Unseen evaluation codes in the Results folder
CSV_PATH = "CNN_ModelbyModel.csv"

# Fonts (global + specific)
FONT_BASE = 40
AXIS_LABEL_FT = 40
TICK_LABEL_FT = 40
LEGEND_FT = 37
PCT_FT = 29          # percentage labels
BATCH_FT = 40        # batch sizes
MODEL_FT = 40        # model names

# Colors
COLOR_MEASURED = "#4C72B0"
COLOR_PRED     = "#DD8452"

# Bars & grouping
BAR_WIDTH = 0.43
GROUP_GAP = 0.9

# Offsets (all easy to tweak)
PCT_DX = -0.15            # shift % labels left (<0) / right (>0)
PCT_DY = 0.03             # fraction of ymax added above the bar tops
BATCH_Y = -0.05           # fraction of ymax (negative = below axis) for batch text
MODEL_Y = -0.17           # fraction of ymax for model text
TICK_BELOW = 0.02         # length of the small downward divider tick below x-axis (fraction of ymax)
Y_BOTTOM_PAD = 0.05       # how much space you want below 0 (fraction of ymax)

# Figure size (change width if you have many models)
FIGSIZE = (12, 5)

# =========================
# ---- GLOBAL STYLE ----
# =========================
rcParams["font.family"]   = "Times New Roman"
rcParams["font.size"]     = FONT_BASE
plt.rcParams.update({
    "axes.labelsize": AXIS_LABEL_FT,
    "axes.titlesize": AXIS_LABEL_FT,
    "xtick.labelsize": TICK_LABEL_FT,
    "ytick.labelsize": TICK_LABEL_FT,
    "legend.fontsize": LEGEND_FT,
})

# =========================
# ---- LOAD DATA ----
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

col_model = next(c for c in df.columns if c.lower() == "model")
col_batch = next(c for c in df.columns if "batch" in c.lower())
col_actual = next(c for c in df.columns if "actual" in c.lower())
col_pred   = next(c for c in df.columns if "pred"   in c.lower())
col_err    = next((c for c in df.columns if "error" in c.lower()), None)

def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True), errors="coerce")

df[col_batch] = to_num(df[col_batch])
df[col_actual] = to_num(df[col_actual])
df[col_pred]   = to_num(df[col_pred])
if col_err is None:
    df["Error (%)"] = 100 * (df[col_pred] - df[col_actual]).abs() / df[col_actual].replace(0, np.nan)
    col_err = "Error (%)"
else:
    df[col_err] = to_num(df[col_err])

df = df.dropna(subset=[col_model, col_batch, col_actual, col_pred])
df = df.sort_values([col_model, col_batch])

# =========================
# ---- PLOT (INTERACTIVE) ----
# =========================
plt.ion()
fig, ax = plt.subplots(figsize=FIGSIZE)

models = df[col_model].unique()
bar_w = BAR_WIDTH
group_gap = GROUP_GAP
cursor = 0.0

ymax_data = df[[col_actual, col_pred]].max().max()
pct_y_pad = PCT_DY * ymax_data

for idx, model in enumerate(models):
    sub = df[df[col_model] == model]
    if sub.empty:
        continue

    n = len(sub)
    pos = np.arange(n) + cursor

    # bars
    ax.bar(
        pos - bar_w / 2,
        sub[col_actual].values,
        width=bar_w,
        color=COLOR_MEASURED,
        edgecolor="black",
        linewidth=1.2,
        label="Measured" if idx == 0 else ""
    )

    ax.bar(
        pos + bar_w / 2,
        sub[col_pred].values,
        width=bar_w,
        color=COLOR_PRED,
        edgecolor="black",
        linewidth=1.2,
        label="Predicted" if idx == 0 else ""
    )

    # small downward divider tick between measured/predicted for each batch
    for px in pos:
        ax.plot([px, px], [-TICK_BELOW * ymax_data, 0],
                color="black", linewidth=1.0, clip_on=False)

    # percentage labels over predicted bars
    for (px, a, p, e) in zip(pos + bar_w/2, sub[col_actual].values, sub[col_pred].values, sub[col_err].astype(float).values):
        if pd.notna(e):
            ax.text(px + PCT_DX, max(a, p) + pct_y_pad,
                    f"{float(e):.1f}%", ha="center", va="bottom", fontsize=PCT_FT)

    # batch sizes above model names (two-line annotation)
    for px, bs in zip(pos, sub[col_batch].astype(int).astype(str)):
        ax.text(px, BATCH_Y * ymax_data, bs, ha="center", va="top", fontsize=BATCH_FT)
    ax.text(np.mean(pos), MODEL_Y * ymax_data, str(model), ha="center", va="top", fontsize=MODEL_FT)

    cursor += n + group_gap

# axes & style
ax.set_xticks([])  # we draw custom labels
ax.set_ylabel("Iteration Execution Time (ms)")
leg = ax.legend(frameon=True)   # you can drag this in interactive mode
try:
    leg.set_draggable(True)
except Exception:
    pass

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# allow room below x-axis for the downward ticks and labels
ax.set_ylim(-Y_BOTTOM_PAD * ymax_data, ymax_data * 1.35)

# pad bottom for the two-line x annotations
plt.subplots_adjust(bottom=0.30, top=0.93, left=0.09, right=0.98)

# =========================
# ---- INTERACTIVE ----
# =========================
print("\n[Interactive mode ON]")
print("Adjust anything (drag legend, resize window, zoom).")
print('Save when ready, e.g.: plt.savefig("my_final_figure.png", dpi=300)\n')

plt.show(block=True)
