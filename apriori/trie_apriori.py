from trienode import Trie
import pandas as pd
import ast
from collections import defaultdict
from itertools import combinations

def load_transactions(csv_path):
    df = pd.read_csv(csv_path)
    df['Items'] = df['Items'].apply(ast.literal_eval)
    return df['Items'].tolist()

def get_frequent_itemsets(transactions, min_support_count):
    itemsets = []
    item_counts = defaultdict(int)

    # Get frequent 1-itemsets
    for txn in transactions:
        for item in txn:
            item_counts[frozenset([item])] += 1

    L1 = {item for item, count in item_counts.items() if count >= min_support_count}
    frequent = {frozenset([item]): count for item, count in item_counts.items() if count >= min_support_count}
    itemsets.append(frequent)

    k = 2
    current_L = L1
    while current_L:
        # Generate candidates from last level
        candidates = set()
        current_L_list = list(current_L)
        for i in range(len(current_L_list)):
            for j in range(i + 1, len(current_L_list)):
                union = current_L_list[i].union(current_L_list[j])
                if len(union) == k:
                    candidates.add(frozenset(union))

        # Insert into Trie
        trie = Trie()
        for candidate in candidates:
            trie.insert(candidate)

        # Count support for candidates
        candidate_counts = defaultdict(int)
        for txn in transactions:
            txn_set = set(txn)
            for candidate in candidates:
                if candidate.issubset(txn_set):
                    candidate_counts[candidate] += 1

        # Keep only those above min_support
        current_L = set()
        frequent_k = {}
        for candidate, count in candidate_counts.items():
            if count >= min_support_count:
                frequent_k[candidate] = count
                current_L.add(candidate)

        if not frequent_k:
            break

        itemsets.append(frequent_k)
        k += 1

    return itemsets