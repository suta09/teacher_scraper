"""
similarity_search.py
--------------------
Cosine Similarity Engine for Academic Faculty Expertise Matching.
Fulfills AI / ML task: "Similarity search: find faculty with similar expertise."
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import clean_text


class FacultySimilarityEngine:
    """
    Search engine calculating cosine similarity between faculty profile embeddings
    and user research queries.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 tfidf_matrix: np.ndarray, 
                 vectorizer: Any):
        self.df = df.reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        
        # Build lookup from faculty_id to row index
        self.id_to_idx = {
            fid: idx for idx, fid in enumerate(self.df['faculty_id'])
        }

    def find_similar_by_faculty_id(self, faculty_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find faculty members with the closest research overlap to a given professor.
        """
        if faculty_id not in self.id_to_idx:
            raise KeyError(f"Faculty ID '{faculty_id}' not found in database.")

        target_idx = self.id_to_idx[faculty_id]
        target_vec = self.tfidf_matrix[target_idx].reshape(1, -1)

        sims = cosine_similarity(target_vec, self.tfidf_matrix)[0]

        # Sort similarity descending, excluding target itself
        ranked_indices = np.argsort(sims)[::-1]
        results = []

        for idx in ranked_indices:
            if idx == target_idx:
                continue
            row = self.df.iloc[idx]
            results.append({
                "faculty_id": row['faculty_id'],
                "name": row['name'],
                "institution": row['institution'],
                "department": row['department'],
                "department_code": row.get('department_code', ''),
                "designation": row['designation'],
                "research_area": row.get('research_area', ''),
                "similarity_score": round(float(sims[idx]), 4)
            })
            if len(results) >= top_n:
                break

        return results

    def search_by_query(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find faculty profiles matching a free-form research query string.
        """
        cleaned = clean_text(query)
        if not cleaned:
            return []

        query_vec = self.vectorizer.transform([cleaned]).toarray()
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        ranked_indices = np.argsort(sims)[::-1]
        results = []

        for idx in ranked_indices:
            score = float(sims[idx])
            if score <= 0.0:
                break
            row = self.df.iloc[idx]
            results.append({
                "faculty_id": row['faculty_id'],
                "name": row['name'],
                "institution": row['institution'],
                "department": row['department'],
                "designation": row['designation'],
                "research_area": row.get('research_area', ''),
                "similarity_score": round(score, 4)
            })
            if len(results) >= top_n:
                break

        return results
