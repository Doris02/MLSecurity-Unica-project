# INSTALL

!pip install -q git+https://github.com/fra31/auto-attack
!pip install -q git+https://github.com/RobustBench/robustbench.git
!pip install -q pandas matplotlib seaborn scipy


# IMPORT

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from robustbench.utils import load_model
from robustbench.data import load_cifar10
from autoattack import AutoAttack
from scipy.stats import spearmanr


# DEVICE (if it's available it will use GPU (CUDA), otherwise CPU)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)


# CONFIG

MODELS = [
    'Carmon2019Unlabeled',
    'Rice2020Overfitting',
    'Engstrom2019Robustness',
    'Rebuffi2021Fixing_28_10_cutmix_ddpm',
    'Gowal2021Improving_28_10_ddpm_100m'
]

EPSILONS_255 = [4, 8, 12]
EPSILONS = [e / 255 for e in EPSILONS_255]

N_EXAMPLES = 20   # test, poi devo aumentare almeno a 100 (consegna 100-200)
BATCH_SIZE = 16


# DATASET

print("Loading CIFAR-10...")

x_test, y_test = load_cifar10(n_examples=N_EXAMPLES)

x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

print("Shape:", x_test.shape)


# EVALUATION

def evaluate(model_name, eps):

    print(f"\nModel: {model_name} | eps: {eps:.4f}")

    model = load_model(
        model_name=model_name,
        dataset='cifar10',
        threat_model='Linf'
    ).to(DEVICE)

    model.eval()

    attack = AutoAttack(
        model,
        norm='Linf',
        eps=eps,
        version='standard',
        device=DEVICE
    )

    x_adv = attack.run_standard_evaluation(
        x_test,
        y_test,
        bs=BATCH_SIZE
    )

    with torch.no_grad():
        preds = model(x_adv).argmax(1)
        acc = (preds == y_test).float().mean().item()

    print("Robust acc:", acc)

    return acc


# RUN 

results = []

for m in MODELS:
    for eps255, eps in zip(EPSILONS_255, EPSILONS):

        acc = evaluate(m, eps)

        results.append({
            "model": m,
            "epsilon": eps,
            "eps_255": eps255,
            "robust_acc": acc
        })


df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\nSaved results.csv")


# RANKING

rank_df = []

for eps in df["epsilon"].unique():

    sub = df[df["epsilon"] == eps].copy()
    sub = sub.sort_values("robust_acc", ascending=False)
    sub["rank"] = range(1, len(sub) + 1)

    rank_df.append(sub)

rank_df = pd.concat(rank_df)


print("\nRANKING:")
print(rank_df[["eps_255","model","robust_acc","rank"]])


# SPEARMAN CORRELATION WITH RESPECT TO THE BASELINE RANKING AT ε = 8/255
# (spearman rank correlation misura la somiglianza tra due classifiche/ranking: 1=ranking identici//0=nessuna relazione// 
# //-1=ranking completamente invertiti) 

baseline = rank_df[rank_df["eps_255"] == 8].sort_values("model")
base_ranks = baseline["rank"].values

corrs = []

for e in sorted(rank_df["eps_255"].unique()):

    cur = rank_df[rank_df["eps_255"] == e].sort_values("model")
    r, _ = spearmanr(base_ranks, cur["rank"].values)

    corrs.append((e, r))

corr_df = pd.DataFrame(corrs, columns=["eps_255", "spearman"])
print("\nSpearman correlation:")
print(corr_df)


# PLOT ACCURACY

plt.figure(figsize=(10,6))

sns.lineplot(
    data=df,
    x="eps_255",
    y="robust_acc",
    hue="model",
    marker="o"
)

plt.title("Robust Accuracy vs Epsilon")
plt.grid()
plt.show()


# PLOT RANK

plt.figure(figsize=(10,6))

sns.lineplot(
    data=rank_df,
    x="eps_255",
    y="rank",
    hue="model",
    marker="o"
)

plt.gca().invert_yaxis()
plt.title("Ranking vs Epsilon")
plt.grid()
plt.show()


# SHIFT ANALYSIS (study of how rankings change when ε varies)

print("\nRANK SHIFTS:\n")

for m in MODELS:

    sub = rank_df[rank_df["model"] == m]

    print(
        m,
        "best:", sub["rank"].min(),
        "worst:", sub["rank"].max(),
        "shift:", sub["rank"].max() - sub["rank"].min()
    )
