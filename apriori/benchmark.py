import time
from memory_profiler import memory_usage
from base_apriori import run_apriori_from_csv
from trie_apriori import load_transactions, get_frequent_itemsets
import pandas as pd

CSV_PATH = "processed_grocery_transactions.csv"
SUPPORTS = [0.01, 0.015, 0.02, 0.025, 0.03]
SIZES = [0.25, 0.5, 0.75, 1.0]

results = []

def run_base(df, min_support):
    def wrapper():
        global base_frequent_itemsets, base_rules
        df.to_csv("temp_subset.csv", index=False)
        base_frequent_itemsets, base_rules = run_apriori_from_csv("temp_subset.csv", min_support, min_support)
        return base_frequent_itemsets
    mem = memory_usage(wrapper)
    return max(mem)

def run_trie(df, min_support):
    def wrapper():
        global trie_frequent_itemsets
        df.to_csv("temp_subset.csv", index=False)
        transactions = load_transactions("temp_subset.csv")
        support_count = int(len(transactions) * min_support)
        trie_frequent_itemsets = get_frequent_itemsets(transactions, support_count)
        return trie_frequent_itemsets
    mem = memory_usage(wrapper)
    return max(mem)

def main():
    full_df = pd.read_csv(CSV_PATH)
    for size in SIZES:
        df_sampled = full_df.sample(frac=size, random_state=42).reset_index(drop=True)
        for support in SUPPORTS:
            # Benchmark base
            start = time.time()
            mem = run_base(df_sampled, support)
            duration = time.time() - start
            results.append(["Base", size, support, mem, duration])

            # Benchmark trie
            start = time.time()
            mem = run_trie(df_sampled, support)
            duration = time.time() - start
            results.append(["Trie", size, support, mem, duration])

    # Output results
    print("Benchmark Matrix (Algorithm, Size, Min Support, Memory MB, Time sec):")
    for row in results:
        print(row)

    # Save results to CSV
    pd.DataFrame(results, columns=["Algorithm", "Size", "MinSupport", "MemoryMB", "TimeSec"]).to_csv("benchmark_matrix.csv", index=False)

if __name__ == "__main__":
    main()
