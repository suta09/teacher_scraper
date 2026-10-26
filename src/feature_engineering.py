"""
feature_engineering.py
----------------------
TF-IDF Vectorization, Feature Importance Analysis, and Dimensionality Reduction.
Fulfills IDC409 Requirement 4:
  - "Features to be understood and to be seen if one can reduce."
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_selection import chi2, SelectKBest


class FeaturePipeline:
    """
    Handles feature representation, interpretability (top terms per department),
    and dimensionality reduction via TruncatedSVD / LSA.
    """

    def __init__(self, 
                 max_features: int = 400,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 n_components: int = 25):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.n_components = n_components

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.feature_names: List[str] = []

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer on profile texts and return TF-IDF feature matrix."""
        X_tfidf = self.vectorizer.fit_transform(texts).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        return X_tfidf

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform unseen texts into TF-IDF space."""
        return self.vectorizer.transform(texts).toarray()

    def reduce_dimensions(self, X_tfidf: np.ndarray) -> np.ndarray:
        """
        Reduce high-dimensional TF-IDF representation into dense latent semantic components.
        Fulfills Requirement 4 (dimensionality reduction).
        """
        # Ensure n_components <= number of features and samples
        actual_components = min(self.n_components, X_tfidf.shape[1], X_tfidf.shape[0] - 1)
        if actual_components != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=actual_components, random_state=42)

        X_reduced = self.svd.fit_transform(X_tfidf)
        return X_reduced

    def transform_dimensions(self, X_tfidf: np.ndarray) -> np.ndarray:
        """Project TF-IDF features into already-fitted latent SVD components without refitting."""
        return self.svd.transform(X_tfidf)

    def get_explained_variance(self) -> Dict[str, Any]:
        """Return variance explained by SVD components."""
        if not hasattr(self.svd, "explained_variance_ratio_"):
            return {}
        evr = self.svd.explained_variance_ratio_
        return {
            "n_components": len(evr),
            "individual_variance": evr.tolist(),
            "cumulative_variance": float(np.sum(evr))
        }

    def analyze_top_terms_by_department(self, 
                                        X_tfidf: np.ndarray, 
                                        labels: List[str], 
                                        top_k: int = 8) -> Dict[str, List[Tuple[str, float]]]:
        """
        Understand feature importance per discipline using Chi-Square scoring.
        Fulfills Requirement 4 (understanding features).
        """
        dept_labels = np.array(labels)
        unique_depts = np.unique(dept_labels)
        top_terms_by_dept = {}

        for dept in unique_depts:
            # Binary target: is department vs rest
            y_binary = (dept_labels == dept).astype(int)
            if np.sum(y_binary) < 2:
                continue

            chi2_scores, _ = chi2(X_tfidf, y_binary)
            top_indices = np.argsort(chi2_scores)[::-1][:top_k]
            
            top_terms = [
                (self.feature_names[idx], round(float(chi2_scores[idx]), 3))
                for idx in top_indices if chi2_scores[idx] > 0
            ]
            top_terms_by_dept[dept] = top_terms

        return top_terms_by_dept
