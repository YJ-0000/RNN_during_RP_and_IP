# 🧠 RNN_during_RP_and_IP

> **Analysis code for the paper:**
> Song, Y., Kim, H., & Kim, T. A minimal recurrent neural network models the robustness of interleave practice on motor sequence learning.

## 🚀 Main Scripts

* **`Code01_Random_vs_Blocked.py`**

  * Trains two identical minimal RNNs on synthetic motor-sequence data under **Random Practice (RP)** and **Blocked/Interleaved Practice (IP)**, then evaluates **pre-test, retention, and transfer** performance.
  * Repeats the full pipeline in parallel (multiple runs), performs vulnerability tests (**noise injection, weight pruning, interference retraining**), and saves all metrics as `.npy` files for downstream analysis.

* **`Code02_Plot.py`**

  * Loads the saved `.npy` results and runs statistical analyses (e.g., **correlations** and **paired t-tests**) to quantify differences between RP and IP across evaluation phases.
  * Generates publication-style figures (learning curves, bar plots with error bars, CI plots with per-point significance markers, and violin plots) and exports all arrays to `.csv`.

## 📂 Directories
* **`utils/`**

  * Contains core utility functions required to run the main experiments, including data generation, network model definitions, and training/evaluation helpers.

* **`Results_lr_0_02_save/`**

  * Contains the fixed result datasets used in the original paper experiments (learning rate = 0.02), preserved for reproducibility and reference.


