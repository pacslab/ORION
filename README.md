# ORION: Integrated Runtime Modeling for Predicting Deep Learning Training Time

<p align="center">
  <img src="ORION_LOGO.jpg" alt="ORION Logo" width="500"/>
</p>

ORION is an integrated runtime prediction framework that jointly models **GPU compute**, **CPU data preparation**, and **storage throughput** to accurately estimate the per-iteration training time of modern deep neural networks across diverse hardware configurations.

This repository includes:
- Scripts for **data generation** across CNN, MLP, and Transformer architectures
- Scripts for **RMSE evaluation**, including baseline comparisons
- Scripts for **unseen-GPU (LOGO) evaluation**
- Scripts for **reproducing all figures** included in the paper (ICPE 2026 submission)

---

## Installation

```bash
git clone https://github.com/genericgitrepos/ORION.git
cd ORION
pip install -r requirements.txt

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```
---
## Generating Benchmark Data (Data Generation)
```bash
Data Generation
cd "Data Generation"
python "CNN_Training.py"
# For MLPs -> python MLP_Training.py
# For Transformers -> python Transformer_Training.py
```

---
## Results Reproducing 
All evaluation scripts reside in the Results/ directory.
### Per-Model Baseline RMSE
```bash
cd Results
python CNN_RMSE_BaselineEval.py
# For MLPs -> python MLP_RMSE_BaselineEval.py
# For Transformers -> python Transformer_RMSE_BaselineEval.py
```
### Overall RMSE (Aggregated Across Configurations)
```bash
cd Results
python CNN_RMSE_BaselineEval_Overall.py
# For MLPs -> python MLP_RMSE_BaselineEval_Overall.py
# For Transformers -> python Transformer_RMSE_BaselineEval_Overall.py
```
### Unseen-GPU (LOGO) Evaluation
```bash
cd Results
python CNN_UnseenGPUs_SummaryTable.py
# For MLPs -> python MLP_UnseenGPUs_SummaryTable.py
# For Transformers -> python Transformer_UnseenGPUs_SummaryTable.py
```
---
## Generating Figures
All plotting scripts and required CSV files are inside:
```bash
Figures/
```
### RMSE on each unseen GPU Figures (across all models)
```bash
cd Figures
python RMSE_Figures.py
```
### Predicted vs actual unseen GPU figures 
```bash
python GPU_Figures.py
```
---
ORION: Integrated Runtime Modeling for Predicting Deep Learning Training Time, submitted to ICPE 2026.






