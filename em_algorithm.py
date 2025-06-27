# em_algorithm.py - Manan Ambaliya(121118776)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import curses
import time

# Real-time visualization
def show_stats_realtime(stdscr, it, total_iters, log_likelihoods, cluster_sizes, ma_window=5):
    """
        Displays real-time EM progress including log likelihood and cluster sizes during iterations.

        Args:
            stdscr (curses screen): Terminal screen used for real-time visualization.
            it (int): Current iteration number.
            total_iters (int): Total number of iterations for EM.
            log_likelihoods (list): Log-likelihood values at each iteration.
            cluster_sizes (list): A list of cluster sizes recorded at each iteration.
            ma_window (int, optional): Size of the moving average window for visualization. Defaults to 5.

        Functionality:
            This function dynamically updates the terminal screen with iteration details,
            moving average log-likelihood, and cluster size statistics.
    """
    curses.curs_set(0)
    stdscr.clear()
    ll_ma = np.mean(log_likelihoods[max(0, it-ma_window+1):it+1])
    stdscr.addstr(0, 0, f"EM Iteration: {it+1}/{total_iters}")
    stdscr.addstr(2, 0, f"Log-Likelihood: {log_likelihoods[it]:.2f}")
    stdscr.addstr(3, 0, f"Moving Avg ({ma_window}): {ll_ma:.2f}")
    cs = cluster_sizes[it]
    cs_ma = np.mean(cluster_sizes[max(0, it-ma_window+1):it+1], axis=0)
    stdscr.addstr(5, 0, "Cluster Sizes:     " + ", ".join([f"{int(x)}" for x in cs]))
    stdscr.addstr(6, 0, "Moving Avg Sizes:  " + ", ".join([f"{x:.1f}" for x in cs_ma]))
    stdscr.refresh()
    time.sleep(0.1)

# EM with real-time curses
def em_gmm_with_realtime(X, n_clusters=5, max_iter=50, tol=1e-4, ma_window=5, verbose=True):

    """
        Implements the Expectation-Maximization (EM) algorithm for GMM with real-time terminal updates.

        Args:
            X (numpy.ndarray): Feature matrix representing the dataset.
            n_clusters (int, optional): Number of clusters for GMM. Defaults to 5.
            max_iter (int, optional): Maximum number of iterations. Defaults to 50.
            tol (float, optional): Convergence threshold for log-likelihood changes. Defaults to 1e-4.
            ma_window (int, optional): Size of the moving average window for the log-likelihood. Defaults to 5.
            verbose (bool, optional): Display log-likelihood updates during the process. Defaults to True.

        Highlights:
            - Utilizes k-means clustering for initialization of cluster means.
            - Alternates between E-step (responsibility calculation) and M-step (parameter updates).
            - Evaluates convergence based on the change in log-likelihood.
            - Displays progress in real-time using the `curses` module.
            - Returns the final parameters of the GMM and the convergence log.

        Returns:
            tuple: Contains the following elements:
                - mu (numpy.ndarray): Means of the GMM clusters.
                - sigma2 (numpy.ndarray): Variances of the GMM clusters.
                - weights (numpy.ndarray): Weights of the GMM clusters.
                - cluster_assignments (numpy.ndarray): Cluster assignments for each data point.
                - log_likelihoods (list): Log-likelihood history of the algorithm.
    """

    N, D = X.shape
    # ---- Initialization ----
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    mu = kmeans.cluster_centers_
    assignments = kmeans.labels_

    weights = np.array([(assignments == k).sum() / N for k in range(n_clusters)])
    sigma2 = np.zeros((n_clusters, D))
    for k in range(n_clusters):
        members = X[assignments == k]
        if len(members) == 0:
            sigma2[k] = 1e-8
        else:
            var = np.var(members, axis=0)
            var[var < 1e-8] = 1e-8
            sigma2[k] = var

    log_likelihoods = []
    cluster_sizes = []

    def log_gaussian(X, mu, sigma2):
        N, D = X.shape
        K = mu.shape[0]
        log_prob = np.zeros((N, K))
        for kk in range(K):
            term1 = -0.5 * np.sum(np.log(2 * np.pi * sigma2[kk]))
            term2 = -0.5 * np.sum(((X - mu[kk]) ** 2) / sigma2[kk], axis=1)
            log_prob[:, kk] = term1 + term2
        return log_prob

    def curses_main(stdscr):
        prev_ll = -np.inf
        for it in range(max_iter):
            # E-step
            log_resp = log_gaussian(X, mu, sigma2) + np.log(weights + 1e-16)
            log_sum = np.logaddexp.reduce(log_resp, axis=1)
            resp = np.exp(log_resp - log_sum[:, None])

            # Log-likelihood
            ll = np.sum(log_sum)
            log_likelihoods.append(ll)
            Nk = resp.sum(axis=0)
            cluster_sizes.append(Nk.copy())

            # Real-time terminal update
            show_stats_realtime(stdscr, it, max_iter, log_likelihoods, cluster_sizes, ma_window=ma_window)

            if verbose:
                stdscr.addstr(8, 0, f"Iteration {it+1}: log-likelihood = {ll:.4f}")

            # Check for convergence
            if np.abs(ll - prev_ll) < tol:
                stdscr.addstr(10, 0, "Converged. Press any key to exit.")
                stdscr.refresh()
                stdscr.getch()
                break
            prev_ll = ll

            # M-step
            weights[:] = Nk / N
            mu[:] = (resp.T @ X) / Nk[:, None]
            for k in range(n_clusters):
                diff = X - mu[k]
                sigma2_k = (resp[:, k][:, None] * (diff ** 2)).sum(axis=0) / Nk[k]
                sigma2_k[sigma2_k < 1e-10] = 1e-10
                sigma2[k] = sigma2_k

        else:
            stdscr.addstr(10, 0, "Reached max_iter. Press any key to exit.")
            stdscr.refresh()
            stdscr.getch()

    curses.wrapper(curses_main)

    # Final assignments
    final_log_resp = log_gaussian(X, mu, sigma2) + np.log(weights + 1e-16)
    final_resp = np.exp(final_log_resp - np.logaddexp.reduce(final_log_resp, axis=1)[:, None])
    cluster_assignments = np.argmax(final_resp, axis=1)

    return mu, sigma2, weights, cluster_assignments, log_likelihoods

# Main Script
if __name__ == "__main__":

    """
        Executes the EM-GMM clustering workflow step-by-step, including data loading, preprocessing,
        and saving of results after clustering.

        Steps:
        1. Load textual data from a CSV file.
        2. Preprocess text using TF-IDF vectorization with dimensionality reduction.
        3. Cluster the features using the EM-GMM algorithm with real-time visualization.
        4. Save the clustering results, parameters, and convergence logs for later analysis.
    """
    # --------- Settings ---------
    csv_file = "people_wiki.csv"
    n_clusters = 5
    max_features = 5000
    max_iter = 50

    # 1. Load Data
    df = pd.read_csv(csv_file)
    docs = df['text'].tolist() if 'text' in df.columns else df.iloc[:,1].tolist()
    doc_ids = df['id'] if 'id' in df.columns else df.index

    # 2. TF-IDF
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(docs)
    X = normalize(X, norm='l2', axis=1).toarray()
    print("Shape of TF-IDF matrix:", X.shape)

    # 3. EM with real-time visualization
    print("Starting EM algorithm with real-time terminal visualization...")
    mu, sigma2, weights, cluster_assignments, log_likelihoods = em_gmm_with_realtime(
        X, n_clusters=n_clusters, max_iter=max_iter
    )

    # 4. Save binary outputs for future analysis
    print("Saving results...")
    np.savez("em_parameters.npz", mu=mu, sigma2=sigma2, weights=weights)
    np.save("cluster_assignments.npy", cluster_assignments)
    np.save("convergence_log.npy", np.array(log_likelihoods))

    # Also as text files
    with open("cluster_assignments.txt", "w") as f:
        for idx, clust in zip(doc_ids, cluster_assignments):
            f.write(f"{idx}\t{clust}\n")
    with open("convergence_log.txt", "w") as f:
        for val in log_likelihoods:
            f.write(f"{val}\n")
    with open("em_parameters.txt", "w") as f:
        f.write(f"Weights: {weights.tolist()}\n")
        f.write(f"Means shape: {mu.shape}\n")
        f.write(f"Variances shape: {sigma2.shape}\n")

    print("EM and visualization complete. All outputs saved.")

