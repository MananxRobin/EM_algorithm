# output_formatter.py - Manan Ambaliya(121118776)

import numpy as np
import json

def load_idx2word(json_file):
    """
        Loads and processes the JSON file containing an index-to-word mapping.

        Args:
            json_file (str): Path to the JSON file mapping indices to words.

        Returns:
            dict: A dictionary where the key is the index (integer) and the value is the word (string).
    """

    with open(json_file, 'r') as f:
        idx2word = json.load(f)
    idx2word = {int(v): k for k, v in idx2word.items()}
    return idx2word

def top_words_per_cluster(mu, sigma2, idx2word, topn=5):
    """
        Extracts the top words for each cluster based on the mean (mu) values and includes variance (sigma2) stats.

        Args:
            mu (numpy.ndarray): The mean matrix (K x D) for the clusters, where K is the number of clusters
                                and D is the number of dimensions (features/words).
            sigma2 (numpy.ndarray): The variance matrix (K x D) for the clusters.
            idx2word (dict): A dictionary mapping feature indices to human-readable words.
            topn (int, optional): Number of top words to extract per cluster. Defaults to 5.

        Returns:
            tuple:
                - topwords (list): A list of lists where each inner list contains the top `topn` words for a cluster.
                - stats_lines (list): A list of formatted strings summarizing the top words and their variances for each cluster.
    """

    topwords = []
    stats_lines = []
    K, D = mu.shape
    for k in range(K):
        # Top indices by mean
        topidx = np.argsort(-mu[k])[:topn]
        words = [idx2word.get(i, f"IDX_{i}") for i in topidx]
        vars_ = [sigma2[k][i] for i in topidx]
        topwords.append(words)
        stats = [f"{w} (var={v:.3e})" for w, v in zip(words, vars_)]
        stats_lines.append(f"Cluster {k} top words: " + ", ".join(stats))
    return topwords, stats_lines

if __name__ == "__main__":
    """
        Main script to process the EM algorithm outputs and generate human-readable summaries.
    """


    # Load EM parameters and word mapping
    data = np.load("em_parameters.npz")
    mu, sigma2, weights = data['mu'], data['sigma2'], data['weights']
    idx2word = load_idx2word("4_map_index_to_word.json")

    topwords, stats_lines = top_words_per_cluster(mu, sigma2, idx2word, topn=5)
    with open("cluster_stats.txt", "w") as f:
        for line in stats_lines:
            f.write(line + "\n")
    with open("em_parameters.txt", "w") as f:
        f.write(f"Weights: {weights.tolist()}\n")
        f.write(f"Means shape: {mu.shape}\n")
        f.write(f"Variances shape: {sigma2.shape}\n")