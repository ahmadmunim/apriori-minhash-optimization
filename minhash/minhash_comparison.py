import pandas as pd
import ast
import random
from base_minhash import compute_base_minhash_signatures
from weighted_minhash import compute_weighted_minhash_signatures

# Load and sample datasets
base_df = pd.read_csv("preprocessed_reviews.csv")
base_df['Tokens'] = base_df['Tokens'].apply(ast.literal_eval)

weighted_df = pd.read_csv("preprocessed_reviews_weighted.csv")
weighted_df['TokenWeights'] = weighted_df['TokenWeights'].apply(ast.literal_eval)

# Sample the same 1,000 reviews by index
sample_indices = base_df.sample(n=1000, random_state=42).index
base_sample = base_df.loc[sample_indices].reset_index(drop=True)
weighted_sample = weighted_df.loc[sample_indices].reset_index(drop=True)

# Compute signatures
base_sample['MinHash'] = compute_base_minhash_signatures(base_sample['Tokens'])
weighted_sample['WeightedMinHash'] = compute_weighted_minhash_signatures(weighted_sample['TokenWeights'])

# Randomly select 250 review pairs to compare
pairs = random.sample([(i, j) for i in range(1000) for j in range(i+1, 1000)], 250)

results = []
for i, j in pairs:
    base_sim = base_sample['MinHash'][i].jaccard(base_sample['MinHash'][j])
    weighted_sim = weighted_sample['WeightedMinHash'][i].jaccard(weighted_sample['WeightedMinHash'][j])
    results.append((i, j, round(base_sim, 3), round(weighted_sim, 3)))

# Output to CSV
comparison_df = pd.DataFrame(results, columns=["Review A", "Review B", "Base Similarity", "Weighted Similarity"])
comparison_df.to_csv("minhash_comparison_output.csv", index=False)
print("Output saved to minhash_comparison_output.csv")
