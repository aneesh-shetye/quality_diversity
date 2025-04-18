import numpy as np 
from scipy.stats import gaussian_kde
import ot 
from sklearn.decomposition import PCA 

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

def wasserstein(p, q): 
    """
    Computes Wasserstein distance. 
    Inputs are two lists of actions taken. 
    """
    print("####################################")
    print("p and q shapes in scoring") 
    print(p.shape) 
    print(q.shape) 
    print("####################################")


    pca = PCA()
    p_pca = pca.fit_transform(p.T).T
    explained = pca.explained_variance_
    nonzero_dims = np.sum(explained > 1e-10)
    p_reduced = p_pca[:nonzero_dims]
    
    pca = PCA()
    q_pca = pca.fit_transform(q.T).T
    explained = pca.explained_variance_
    nonzero_dims = np.sum(explained > 1e-10)
    q_reduced = q_pca[:nonzero_dims]


    print("####################################")
    print("p_reduced and q_reduced shapes in scoring") 
    print(p_reduced.shape) 
    print(q_reduced.shape) 
    print("####################################")

    kde1 = gaussian_kde(p_reduced)
    kde2 = gaussian_kde(q_reduced)

    n_samples = 1000 
    samples1 = kde1.resample(n_samples).T  # Shape: (N, 3)
    samples2 = kde2.resample(n_samples).T

# Compute pairwise cost matrix (Euclidean distance)
    M = ot.dist(samples1, samples2, metric='euclidean')
    M /= M.max()  # Normalize for numerical stability

    a = np.ones((n_samples,))/n_samples
    b = np.ones((n_samples,))/n_samples

    wasserstein_distance = ot.emd2(a,b, M) 
    return wasserstein_distance 

def wasserstein_with_weights(p, q, w1, w2): 

    '''
    print("####################################")
    print("p, q and w1, w2 shapes in wasserstein_with_weights") 
    print(p.shape) 
    print(q.shape) 
    print(w1.shape) 
    print(w2.shape) 
    print("####################################")
    '''

    cost_matrix = ot.dist(p, q)#, metric='euclidean') 
    wasserstein_distance_squared = ot.emd2(w1, w2, cost_matrix) 
    wasserstein_distance = np.sqrt(wasserstein_distance_squared)
    
    return wasserstein_distance 

