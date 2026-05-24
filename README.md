# Reassessing CIFAR-10 Robust Models Across Different ε Values Using AutoAttack

## Assignment
Select five CIFAR-10 (ℓ∞) models from RobustBench and re-evaluate them using AutoAttack under different values of the radius epsilon (e.g., from 1/255 to 16/255, regularly spaced interval including the baseline 8/255), using a subset of 100-200 samples. Compare the resulting robust accuracies and model rankings across these settings. Evaluate the stability of model rankings across different epsilon values. Identify cases where these changes lead to significant rank shifts and discuss what this reveals about the reliability of RobustBench leaderboards. 


## Repository Overview
This repository contains the source code and material for the final project of the **Machine Learning Security** course.  
The study evaluates the impact of varying the perturbation radius ε on the stability of adversarial robustness leaderboards on CIFAR-10. By analyzing these shifts, we highlight the structural limitations of static benchmarks (such as RobustBench) in light of recent findings in the literature.



## Project Description

The project assignment requires selecting five certified $\ell_\infty$ threat model breakthroughs on CIFAR-10 from **RobustBench**. These models are re-evaluated using **AutoAttack** across a regularly spaced interval of $\varepsilon$ values (ranging from $2/255$ to $16/255$, explicitly including the standard baseline of $8/255$) using a sample subset of 100 images at first and then of 150 images. The core objective is to critically assess the stability of model rankings and investigate significant rank shifts.

### Selected Models:
1. `Carmon2019Unlabeled` (Carmon)
2. `Rice2020Overfitting` (Rice)
3. `Engstrom2019Robustness` (Engstrom)
4. `Rebuffi2021Fixing_28_10_cutmix_ddpm` (Rebuffi)
5. `Gowal2021Improving_28_10_ddpm_100m` (Gowal)



## Repository Structure

The repository is organized as follows:

* **`code.py`**: the main evaluation script that runs the local pipeline on a subset of 150 CIFAR-10 test samples. It downloads the target models via RobustBench, deploys AutoAttack across the selected $\varepsilon$ steps, tracks point-wise rankings, computes Spearman rank correlation relative to the baseline and prints a final rank shift analysis.
* **`config.py`**: a clean configuration helper module designed to systematically manage output file paths for results and plots.
* **`plot_results.py`**: a utility script to recreate overall comparative plots (Robust Accuracy vs. $\varepsilon$ and Ranking vs. $\varepsilon$) from the raw execution log data.
* **`plot_worst_case_acc_and_ranking.py`**: an aggregation script focused on **Worst-Case Robust Accuracy** (extracting the absolute minimum resilience achieved by a model under the strongest sub-attack within the AutoAttack ensemble).
* **`requirements.txt`**: list of Python dependencies required to replicate the computational environment (including proper PyTorch wheel indices and RobustBench wrappers).
* **`csv_files/`**: contains execution logs and outputs stored in tabular data format (`results.csv`, `worst_case_accuracy.csv`, `worst_case_ranking.csv`).
* **`plots/`**: storage for exported high-resolution visualization graphs used in the final report document.



## Setup and Installation

To install the necessary dependencies and prepare the execution environment, make sure you have Python 3.10+ installed along with up-to-date GPU drivers (highly recommended to accelerate AutoAttack evaluation).

Run the following installation command:

```bash
pip install -r requirements.txt
```

## How to Run the Code

1. **Execute the Evaluation Pipeline:**
   ```bash
   python code.py

2. **Generate Worst-Case Graphs (Highly Recommended for the Report):**
   ```bash
   python plot_worst_case_acc_and_ranking.py

3. **Generate Global Sub-Attack Plots:**
   ```bash
   python plot_results.py
