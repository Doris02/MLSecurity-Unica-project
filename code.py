"""Local runner for RobustBench evaluation on CIFAR-10 using AutoAttack."""

# Requirements (run once):
# python -m pip install torch torchvision
# python -m pip install git+https://github.com/RobustBench/robustbench.git
# python -m pip install pandas matplotlib seaborn scipy

import time
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from robustbench.utils import load_model
from robustbench.data import load_cifar10
from autoattack import AutoAttack
from scipy.stats import spearmanr


def evaluate(model_name, eps, x_test, y_test, device, batch_size):
    print(f"\nModel: {model_name} | eps: {eps:.4f}")

    model = load_model(
        model_name=model_name,
        dataset='cifar10',
        threat_model='Linf'
    ).to(device)

    model.eval()

    base_attack = AutoAttack(
        model,
        norm='Linf',
        eps=eps,
        version='standard',
        device=device
    )
    attack_names = list(base_attack.attacks_to_run)

    results = []

    for attack_name in attack_names:
        print(f"Attack: {attack_name}")

        attack = AutoAttack(
            model,
            norm='Linf',
            eps=eps,
            version='standard',
            device=device
        )
        attack.attacks_to_run = [attack_name]

        x_adv = attack.run_standard_evaluation(
            x_test,
            y_test,
            bs=batch_size
        )

        with torch.no_grad():
            preds = model(x_adv).argmax(1)
            acc = (preds == y_test).float().mean().item()

        print(f"Robust acc ({attack_name}):", acc)

        results.append({
            "attack": attack_name,
            "robust_acc": acc
        })

    return results


def save_progress(results, path="results.csv"):
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"Saved {path}")


def main():
    # DEVICE (if it's available it will use GPU (CUDA), otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    
    # CONFIG
    models = [
        'Carmon2019Unlabeled',
        'Rice2020Overfitting',
        'Engstrom2019Robustness',
        'Rebuffi2021Fixing_28_10_cutmix_ddpm',
        'Gowal2021Improving_28_10_ddpm_100m'
    ]

    epsilons_255 = [4, 8, 12]
    epsilons = [e / 255 for e in epsilons_255]

    n_examples = 100   # test, poi devo aumentare almeno a 100 (consegna 100-200)
    batch_size = 100
    cooldown_seconds = 30  # cooldown between runs to avoid overheating the GPU

    # DATASET
    print("Loading CIFAR-10...")

    x_test, y_test = load_cifar10(n_examples=n_examples)

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    print("Shape:", x_test.shape)

    # RUN
    results = []

    for mi, m in enumerate(models):
        for i, (eps255, eps) in enumerate(zip(epsilons_255, epsilons)):
            attack_results = evaluate(m, eps, x_test, y_test, device, batch_size)

            for attack_result in attack_results:
                results.append({
                    "model": m,
                    "epsilon": eps,
                    "eps_255": eps255,
                    "attack": attack_result["attack"],
                    "robust_acc": attack_result["robust_acc"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                save_progress(results)

            if i < len(epsilons) - 1:
                print(f"Cooling down for {cooldown_seconds}s...")
                time.sleep(cooldown_seconds)

        if mi < len(models) - 1:
            print(f"Cooling down between models for {cooldown_seconds}s...")
            time.sleep(cooldown_seconds)

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

    print("\nSaved results.csv")

    # RANKING
    rank_df = []

    for (eps, attack_name), sub in df.groupby(["epsilon", "attack"]):
        sub = sub.copy()
        sub = sub.sort_values("robust_acc", ascending=False)
        sub["rank"] = range(1, len(sub) + 1)

        rank_df.append(sub)

    rank_df = pd.concat(rank_df, ignore_index=True)

    print("\nRANKING (per attack):")
    print(rank_df[["eps_255", "attack", "model", "robust_acc", "rank"]])

    # SPEARMAN CORRELATION WITH RESPECT TO THE BASELINE RANKING AT ε = 8/255
    # (spearman rank correlation misura la somiglianza tra due classifiche/ranking: 1=ranking identici//0=nessuna relazione//
    # //-1=ranking completamente invertiti)
    corrs = []

    for attack_name in sorted(rank_df["attack"].unique()):
        baseline = rank_df[
            (rank_df["attack"] == attack_name) & (rank_df["eps_255"] == 8)
        ].sort_values("model")
        if baseline.empty:
            continue
        base_ranks = baseline["rank"].values

        for e in sorted(rank_df["eps_255"].unique()):
            cur = rank_df[
                (rank_df["attack"] == attack_name) & (rank_df["eps_255"] == e)
            ].sort_values("model")
            if len(cur) != len(base_ranks):
                continue
            r, _ = spearmanr(base_ranks, cur["rank"].values)

            corrs.append((attack_name, e, r))

    corr_df = pd.DataFrame(corrs, columns=["attack", "eps_255", "spearman"])
    print("\nSpearman correlation:")
    print(corr_df)

    # PLOT ACCURACY
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="eps_255",
        y="robust_acc",
        hue="model",
        style="attack",
        markers=True,
        dashes=False,
        estimator=None
    )

    plt.title("Robust Accuracy vs Epsilon")
    plt.grid()
    plt.show()

    # PLOT RANK
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=rank_df,
        x="eps_255",
        y="rank",
        hue="model",
        style="attack",
        markers=True,
        dashes=False,
        estimator=None
    )

    plt.gca().invert_yaxis()
    plt.title("Ranking vs Epsilon")
    plt.grid()
    plt.show()

    # SHIFT ANALYSIS (study of how rankings change when ε varies)
    print("\nRANK SHIFTS (per attack):\n")

    for m in models:
        for attack_name in sorted(rank_df["attack"].unique()):
            sub = rank_df[
                (rank_df["model"] == m) & (rank_df["attack"] == attack_name)
            ]
            if sub.empty:
                continue

            print(
                m,
                "|",
                attack_name,
                "best:", sub["rank"].min(),
                "worst:", sub["rank"].max(),
                "shift:", sub["rank"].max() - sub["rank"].min()
            )


if __name__ == "__main__":
    main()
