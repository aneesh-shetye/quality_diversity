import numpy as np 

def kl_divergence(p, q, epsilon=1e-10):
    """
    Computes the KL divergence D_KL(p || q) for two probability distributions.
    p and q are assumed to be dictionaries mapping actions to probabilities.
    A small epsilon is added to avoid log(0) issues.
    """
    all_keys = set(p.keys()).union(q.keys())
    kl = 0.0
    for key in all_keys:
        p_val = p.get(key, epsilon)
        q_val = q.get(key, epsilon)
        kl += p_val * np.log(p_val / q_val)
    return kl
