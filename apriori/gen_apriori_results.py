import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("benchmark_matrix.csv")

# Set Seaborn theme
sns.set(style="whitegrid")

# Plot 1: Memory usage by min_support
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="MinSupport", y="MemoryMB", hue="Algorithm", style="Size", markers=True)
plt.title("Memory Usage vs Min Support")
plt.ylabel("Memory (MB)")
plt.xlabel("Minimum Support")
plt.legend(title="Algorithm / Dataset Size")
plt.tight_layout()
plt.savefig("results/memory_vs_support.png")
plt.show()

# Plot 2: Execution time by min_support
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="MinSupport", y="TimeSec", hue="Algorithm", style="Size", markers=True)
plt.title("Execution Time vs Min Support")
plt.ylabel("Time (sec)")
plt.xlabel("Minimum Support")
plt.legend(title="Algorithm / Dataset Size")
plt.tight_layout()
plt.savefig("results/time_vs_support.png")
plt.show()

# Heatmap for quick comparison
pivot_mem = df.pivot_table(index="MinSupport", columns="Algorithm", values="MemoryMB", aggfunc='mean')
pivot_time = df.pivot_table(index="MinSupport", columns="Algorithm", values="TimeSec", aggfunc='mean')

plt.figure(figsize=(8, 5))
sns.heatmap(pivot_mem, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Avg. Memory Usage (MB)")
plt.ylabel("Min Support")
plt.tight_layout()
plt.savefig("results/heatmap_memory.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(pivot_time, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Avg. Execution Time (s)")
plt.ylabel("Min Support")
plt.tight_layout()
plt.savefig("results/heatmap_time.png")
plt.show()
