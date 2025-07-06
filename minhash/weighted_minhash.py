import pandas as pd
import ast
import numpy as np
from datasketch import WeightedMinHashGenerator

def compute_weighted_minhash_signatures(token_weight_dicts, num_perm=128):
    # Build vocab
    vocab = sorted({token for d in token_weight_dicts for token in d})
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    generator = WeightedMinHashGenerator(len(vocab), sample_size=num_perm, seed=42)

    def to_vector(weight_dict):
        vec = np.zeros(len(vocab))
        for token, weight in weight_dict.items():
            if token in token_to_index:
                vec[token_to_index[token]] = weight
        return vec

    return token_weight_dicts.apply(lambda w: generator.minhash(to_vector(w)))
