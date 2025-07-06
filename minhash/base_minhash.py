import pandas as pd
from datasketch import MinHash
import ast

def compute_base_minhash_signatures(token_lists, num_perm=128):
    def compute(tokens):
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(token.encode('utf8'))
        return m
    return token_lists.apply(compute)
