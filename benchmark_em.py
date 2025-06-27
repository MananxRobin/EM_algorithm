# bechmark_em.py - Manan Ambaliya(121118776)

import time
import tracemalloc
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import pandas as pd

# --- Import your EM implementation ---
from em_algorithm import em_gmm_with_realtime

def run_benchmark_my_em(X, n_clusters=5, max_iter=10):
    """
        Benchmarks the custom EM implementation (`em_gmm_with_realtime`) in terms of time, memory usage, and log-likelihood.

        Args:
            X (numpy.ndarray): Input feature matrix (TF-IDF normalized vectors).
            n_clusters (int, optional): Number of clusters for the GMM, defaults to 5.
            max_iter (int, optional): Maximum number of EM algorithm iterations, defaults to 10.

        Returns:
            tuple: Tuple containing:
                - Time taken (in seconds) to execute the algorithm.
                - Peak memory usage (in MB) during execution.
                - Log-likelihood progression over iterations.
    """
    tracemalloc.start()
    t0 = time.time()
    mu, sigma2, weights, cluster_assignments, log_likelihoods = em_gmm_with_realtime(
        X, n_clusters=n_clusters, max_iter=max_iter, verbose=False
    )
    t1 = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return t1-t0, peak_mem / 1e6, log_likelihoods

def run_benchmark_sklearn(X, n_clusters=5, max_iter=10):
    """
        Benchmarks Scikit-learn's GaussianMixture implementation in terms of time, memory usage, and log-likelihood.

        Args:
            X (numpy.ndarray): Input feature matrix (TF-IDF normalized vectors).
            n_clusters (int, optional): Number of clusters for the GMM, defaults to 5.
            max_iter (int, optional): Maximum number of iterations of EM, defaults to 10.

        Returns:
            tuple: Tuple containing:
                - Time taken (in seconds) to execute the algorithm.
                - Peak memory usage (in MB) during execution.
                - [Log-likelihood]: A single-element list containing the final log-likelihood value.
    """

    tracemalloc.start()
    t0 = time.time()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', max_iter=max_iter, verbose=1)
    gmm.fit(X)
    t1 = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # sklearn only returns final lower_bound_ for log-likelihood, but not per-iteration unless verbose
    return t1-t0, peak_mem / 1e6, [gmm.lower_bound_ * X.shape[0]]

if __name__ == "__main__":
    """
        Main benchmarking script that loads text data, extracts features, and benchmarks both the custom EM
        implementation and Scikit-learn's GaussianMixture implementation.
    """

    n_clusters = 5
    max_features = 5000
    max_iter = 50

    print("Loading data and extracting TF-IDF features...")
    df = pd.read_csv("people_wiki.csv")
    docs = df['text'].tolist() if 'text' in df.columns else df.iloc[:,1].tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(docs)
    X = normalize(X, norm='l2', axis=1).toarray()
    print("Shape of TF-IDF matrix:", X.shape)

    print("\nBenchmarking your custom EM implementation...")
    time_my, mem_my, ll_my = run_benchmark_my_em(X, n_clusters=n_clusters, max_iter=max_iter)
    print(f"Custom EM: {time_my:.2f}s, {mem_my:.1f} MB, Final LL: {ll_my[-1]:.2f}")

    print("\nBenchmarking sklearn GaussianMixture...")
    time_sk, mem_sk, ll_sk = run_benchmark_sklearn(X, n_clusters=n_clusters, max_iter=max_iter)
    print(f"sklearn EM: {time_sk:.2f}s, {mem_sk:.1f} MB, Final LL: {ll_sk[-1]:.2f}")

    # Output table
    print("\n| Implementation | Time (s) | Peak Mem (MB) | Final Log-Likelihood |")
    print(f"| Custom EM      | {time_my:.2f}   | {mem_my:.1f}        | {ll_my[-1]:.2f} |")
    print(f"| Sklearn EM     | {time_sk:.2f}   | {mem_sk:.1f}        | {ll_sk[-1]:.2f} |")