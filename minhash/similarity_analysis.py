import pandas as pd
import ast

# Load source data
top_base = pd.read_csv("top_base_similarities.csv")
top_weighted = pd.read_csv("top_weighted_similarities.csv")
top_shift = pd.read_csv("top_similarity_shifts.csv")

preprocessed = pd.read_csv("preprocessed_reviews.csv")
preprocessed['Tokens'] = preprocessed['Tokens'].apply(ast.literal_eval)

preprocessed_weighted = pd.read_csv("preprocessed_reviews_weighted.csv")
preprocessed_weighted['TokenWeights'] = preprocessed_weighted['TokenWeights'].apply(ast.literal_eval)

def extract_analysis(df_top, pre_df, weighted_df):
    records = []
    for _, row in df_top.iterrows():
        i, j = int(row['Review A']), int(row['Review B'])
        tokens_i = set(pre_df.loc[i, 'Tokens'])
        tokens_j = set(pre_df.loc[j, 'Tokens'])
        shared = tokens_i & tokens_j

        weights_i = weighted_df.loc[i, 'TokenWeights']
        weights_j = weighted_df.loc[j, 'TokenWeights']

        shared_weights = {
            t: {
                "weight_i": weights_i.get(t, 0),
                "weight_j": weights_j.get(t, 0),
                "avg": round((weights_i.get(t, 0) + weights_j.get(t, 0)) / 2, 3)
            }
            for t in shared
        }

        avg_weight = round(sum(w['avg'] for w in shared_weights.values()) / len(shared_weights), 3) if shared_weights else 0
        top_tokens = sorted(shared_weights.items(), key=lambda x: -x[1]['avg'])[:5]
        top_tokens_str = "; ".join([f"{t[0]} ({t[1]['avg']})" for t in top_tokens])

        records.append({
            "Review A": i,
            "Review B": j,
            "Shared Tokens": list(shared),
            "Shared Token Count": len(shared),
            "Avg Shared Token Weight": avg_weight,
            "Top Weighted Shared Tokens": top_tokens_str
        })
    return pd.DataFrame(records)

# Generate all three and save
extract_analysis(top_base, preprocessed, preprocessed_weighted).to_csv("results/similarity-analysis/top_base_similarity_analysis.csv", index=False)
extract_analysis(top_weighted, preprocessed, preprocessed_weighted).to_csv("results/similarity-analysis/top_weighted_similarity_analysis.csv", index=False)
extract_analysis(top_shift, preprocessed, preprocessed_weighted).to_csv("results/similarity-analysis/top_similarity_shift_analysis.csv", index=False)

print("Exported all three similarity analysis CSVs.")
