import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load comparison data
df = pd.read_csv("minhash_comparison_output.csv")

# Compute similarity shift
df["Shift"] = df["Weighted Similarity"] - df["Base Similarity"]

# Scatter plot: Base vs. Weighted
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Base Similarity", y="Weighted Similarity", data=df, alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Weighted vs. Base MinHash Similarity")
plt.xlabel("Base MinHash Similarity")
plt.ylabel("Weighted MinHash Similarity")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/scatter_similarity_shift.png")
plt.show()

# Histogram of similarity shifts
plt.figure(figsize=(8, 5))
sns.histplot(df["Shift"], bins=30, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribution of Similarity Shifts (Weighted - Base)")
plt.xlabel("Similarity Shift")
plt.ylabel("Pair Count")
plt.tight_layout()
plt.savefig("results/hist_similarity_shift.png")
plt.show()

# Top-N Analysis
top_n = 10

top_base = df.sort_values(by="Base Similarity", ascending=False).head(top_n)
top_weighted = df.sort_values(by="Weighted Similarity", ascending=False).head(top_n)
top_shift = df.sort_values(by="Shift", ascending=False).head(top_n)

# Save to CSVs
top_base.to_csv("top_base_similarities.csv", index=False)
top_weighted.to_csv("top_weighted_similarities.csv", index=False)
top_shift.to_csv("top_similarity_shifts.csv", index=False)

print("Visualizations Rendered. Top comparisons exported to:")
print("top_base_similarities.csv")
print("top_weighted_similarities.csv")
print("top_similarity_shifts.csv")
