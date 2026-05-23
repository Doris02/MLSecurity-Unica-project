import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD RESULTS
df = pd.read_csv("results.csv")

# columns check
required_cols = {"model", "eps_255", "attack", "robust_acc"}
assert required_cols.issubset(df.columns), "Missing columns in CSV"

# SHORT MODEL NAMES (better plot readability)
model_names = {
    "Carmon2019Unlabeled": "Carmon",
    "Rice2020Overfitting": "Rice",
    "Engstrom2019Robustness": "Engstrom",
    "Rebuffi2021Fixing_28_10_cutmix_ddpm": "Rebuffi",
    "Gowal2021Improving_28_10_ddpm_100m": "Gowal"
}

# WORST-CASE ROBUSTNESS (minimum across attacks)
worst_df = (
    df.groupby(["model", "eps_255"])["robust_acc"]
    .min()
    .reset_index()
    .rename(columns={"robust_acc": "robust_acc_worst"})
)

# add short names
worst_df["short_model"] = worst_df["model"].map(model_names)

# sort epsilon values
worst_df = worst_df.sort_values("eps_255")

print("\nWORST-CASE ROBUST ACCURACY:")

# Reset the dataframe index to a clean, consecutive range removing the old indexes from original operations
worst_df = worst_df.reset_index(drop=True)
print(worst_df)


# GLOBAL RANKING PER EPSILON
rank_rows = []

for eps, sub in worst_df.groupby("eps_255"):
    sub = sub.sort_values("robust_acc_worst", ascending=False).copy()

    # assign ranks
    sub["rank"] = range(1, len(sub) + 1)

    rank_rows.append(sub)

rank_df = pd.concat(rank_rows, ignore_index=True)

# sort epsilon values
rank_df = rank_df.sort_values("eps_255")

print("\nGLOBAL RANKING (worst-case):")
print(rank_df[["eps_255", "short_model", "robust_acc_worst", "rank"]].to_string(index=False))

# SAVE CSV FILES
worst_df.to_csv("worst_case_accuracy.csv", index=False)
rank_df.to_csv("worst_case_ranking.csv", index=False)

print("\nSaved:")
print("- worst_case_accuracy.csv")
print("- worst_case_ranking.csv")


# PLOT 1 — WORST-CASE GLOBAL RANKING

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=rank_df,
    x="eps_255",
    y="rank",
    hue="short_model",
    marker="o"
)

# rank 1 at the top
plt.gca().invert_yaxis()

# show only integer ranks
plt.yticks(range(1, len(rank_df["model"].unique()) + 1))

plt.title("Worst-Case Ranking vs Epsilon")
plt.xlabel("Epsilon (eps / 255)")
plt.ylabel("Rank")

plt.grid(True)

plt.tight_layout()

# save image
plt.savefig("worst_case_ranking.png", dpi=300)

plt.show()

print("Saved: worst_case_ranking.png")


# PLOT 2 — WORST-CASE ROBUST ACCURACY

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=worst_df,
    x="eps_255",
    y="robust_acc_worst",
    hue="short_model",
    marker="o"
)

plt.title("Worst-Case Robust Accuracy vs Epsilon")
plt.xlabel("Epsilon (eps / 255)")
plt.ylabel("Worst-Case Robust Accuracy")

plt.grid(True)

plt.tight_layout()

# save image
plt.savefig("worst_case_accuracy.png", dpi=300)

plt.show()

print("Saved: worst_case_accuracy.png")
