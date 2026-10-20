"""
clustering.py
-------------
Unsupervised Research Area Clustering and Topic Discovery.
Evaluated with Silhouette Score, Calinski-Harabasz Score, and Centroid Term Inspection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class ResearchClusterAnalyzer:
    """
    Performs K-Means clustering on faculty profile embeddings,
    evaluates cluster separation via Silhouette analysis, and extracts top cluster keywords.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_k: int = 5
        self.best_model: Any = None
        self.best_silhouette: float = -1.0

    def find_optimal_k(self, 
                       X: np.ndarray, 
                       k_range: range = range(3, 8)) -> pd.DataFrame:
        """
        Evaluate clustering across multiple values of k using Silhouette & Calinski-Harabasz metrics.
        """
        metrics = []

        for k in k_range:
            if k >= X.shape[0]:
                break
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(X)

            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)

            metrics.append({
                "k": k,
                "Silhouette Score": round(float(sil), 4),
                "Calinski-Harabasz Score": round(float(ch), 2),
                "Davies-Bouldin Score": round(float(db), 4)
            })

            if sil > self.best_silhouette:
                self.best_silhouette = sil
                self.best_k = k
                self.best_model = kmeans

        return pd.DataFrame(metrics)

    def extract_cluster_topics(self, 
                               feature_names: List[str], 
                               top_n: int = 6) -> Dict[int, Dict[str, Any]]:
        """
        Extract top terms for each cluster centroid for human interpretation.
        """
        if self.best_model is None:
            raise ValueError("Must run find_optimal_k or fit a clustering model first.")

        centroids = self.best_model.cluster_centers_
        cluster_info = {}

        for cluster_idx in range(self.best_k):
            centroid = centroids[cluster_idx]
            top_term_indices = centroid.argsort()[::-1][:top_n]
            top_terms = [feature_names[i] for i in top_term_indices if i < len(feature_names)]
            
            cluster_info[cluster_idx] = {
                "top_keywords": top_terms,
                "topic_label": " / ".join(top_terms[:3]).title()
            }

        return cluster_info
