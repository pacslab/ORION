# Interactive_Figure_Unseen.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
# ---- USER SETTINGS ----
# =========================

# This is the code to generate figures 2-4
# Figure 2 = MLP_Unseen.csv
# Figure 3 = CNN_Unseen.csv
# Figure 4 = Transformer_Unseen.csv
# The data in the csv files are gathered from the Unseen evaluation codes in the Results folder

CSV_PATH = "Transformer_Unseen.csv"

# Fonts (global + specific)
FONT_BASE = 40
AXIS_LABEL_FT = 40
TICK_LABEL_FT = 40
LEGEND_FT = 35

# Colors (professional palette)
COLOR_PRENET = "#4C72B0"   # blue
COLOR_ORION  = "#DD8452"   # orange

# Bars
BAR_WIDTH = 0.20

# Offsets
Y_BOTTOM_PAD = 0.02   # room below zero

# Figure size
FIGSIZE = (10, 6)

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

def to_num(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
        errors="coerce"
    )

# ---- Infer columns by name ----
# GPU column
col_gpu = next(c for c in df.columns if "gpu" in c.lower())

# ORION column
col_orion = next(c for c in df.columns if "orion" in c.lower())

# PreNeT column
col_prenet = next(c for c in df.columns if "prenet" in c.lower())

# Clean & convert
df[col_orion]  = to_num(df[col_orion])
df[col_prenet] = to_num(df[col_prenet])

# Drop incomplete rows
df = df.dropna(subset=[col_gpu, col_orion, col_prenet])

# Sort by GPU name for consistent ordering
df = df.sort_values(col_gpu).reset_index(drop=True)

# =========================
# ---- PLOT ----
# =========================
plt.ion()
fig, ax = plt.subplots(figsize=FIGSIZE)

gpus = df[col_gpu].astype(str).values
n = len(df)
x = np.arange(n)

# Bar positions: PreNeT (left), ORION (right)
pre_positions   = x - BAR_WIDTH / 2
orion_positions = x + BAR_WIDTH / 2

# Plot bars
pren_bars = ax.bar(
    pre_positions, df[col_prenet].values, width=BAR_WIDTH,
    color=COLOR_PRENET, edgecolor="black", label="PreNeT"
)
orion_bars = ax.bar(
    orion_positions, df[col_orion].values, width=BAR_WIDTH,
    color=COLOR_ORION, edgecolor="black", label="ORION"
)

# Axes, ticks, legend
ymax_data = float(
    np.nanmax([df[col_prenet].values, df[col_orion].values])
)
ax.set_xticks(x)
ax.set_xticklabels(gpus, rotation=0, ha="center")  # centered, no angle
ax.set_ylabel("RMSE (ms)")
ax.set_xlabel("Unseen GPU")
ax.set_ylim(-Y_BOTTOM_PAD * ymax_data, ymax_data * 1.10)

leg = ax.legend(frameon=True)
try:
    leg.set_draggable(True)
except Exception:
    pass

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# =========================
# ---- INTERACTIVE ----
# =========================
print("\n[Interactive mode ON]")
print("You can drag the legend, resize, and zoom.")
print('Save when ready, e.g.: plt.savefig("Transformer_Unseen_GPU_RMSE.png", dpi=300)\n')

plt.show(block=True)
