import numpy as np

def pad_sequence(seq, max_len, feature_dim):
    if len(seq) == 0:
        return np.zeros((max_len, feature_dim))
    padded = np.zeros((max_len, feature_dim))
    length = min(len(seq), max_len)
    padded[:length] = seq[:length]
    return padded
