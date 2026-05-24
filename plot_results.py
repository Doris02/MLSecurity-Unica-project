"""Rebuild plots from results.csv."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import Config


def load_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    if "attack" not in df.columns:
        df["attack"] = "standard"

    if "eps_255" not in df.columns and "epsilon" in df.columns:
        df["eps_255"] = (df["epsilon"] * 255).round().astype(int)

    return df


def build_rank_df(df):
    rank_rows = []

    for (eps, attack_name), sub in df.groupby(["epsilon", "attack"]):
        sub = sub.copy()
        sub = sub.sort_values("robust_acc", ascending=False)
        sub["rank"] = range(1, len(sub) + 1)
        rank_rows.append(sub)

    if not rank_rows:
        return pd.DataFrame()

    return pd.concat(rank_rows, ignore_index=True)


def plot_accuracy(df, out_path):
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="eps_255",
        y="robust_acc",
        hue="model",
        style="attack",
        markers=True,
        dashes=False,
        estimator=None,
    )

    plt.title("Robust Accuracy vs Epsilon")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rank(rank_df, out_path):
    if rank_df.empty:
        return

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=rank_df,
        x="eps_255",
        y="rank",
        hue="model",
        style="attack",
        markers=True,
        dashes=False,
        estimator=None,
    )

    plt.gca().invert_yaxis()
    plt.title("Ranking vs Epsilon")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    cfg = Config()
    file = cfg.get_results_path("results_8_epsilons_150_samples.csv")
    parser = argparse.ArgumentParser(description="Rebuild plots from results.csv")
    parser.add_argument("--results", default=file, help="Path to results.csv")
    parser.add_argument("--outdir", default=cfg.get_plots_dir(), help="Directory to save plots")
    args = parser.parse_args()

    df = load_results(args.results)
    rank_df = build_rank_df(df)

    acc_path = cfg.get_plots_path("robust_accuracy_vs_8_epsilon.png")
    rank_path = cfg.get_plots_path("ranking_vs_8_epsilon.png")

    plot_accuracy(df, acc_path)
    plot_rank(rank_df, rank_path)

    print(f"Saved: {acc_path}")
    if not rank_df.empty:
        print(f"Saved: {rank_path}")


if __name__ == "__main__":
    main()
