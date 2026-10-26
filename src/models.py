"""
models.py
---------
Multi-Model Machine Learning Classification Pipeline for Faculty Profiles.
Fulfills IDC409 Requirement 5 & 6:
  - Models: Logistic Regression, Decision Tree, SVM, Boosted Decision Trees (BDT), Neural Network (NN)
  - Comprehensive Evaluation: Accuracy, Macro/Weighted F1, Precision, Recall, Confusion Matrix.
  - Performance comparison before vs. after dimensionality reduction.
  - Strict Data Leakage Prevention: Encapsulates TfidfVectorizer and TruncatedSVD inside
    sklearn.pipeline.Pipeline so fitting occurs strictly on training folds/splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class DisciplineClassificationBenchmark:
    """
    Evaluates and benchmarks 5 machine learning architectures
    on faculty discipline/department prediction with zero data leakage.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=1.0, random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=7, min_samples_split=4, random_state=self.random_state
            ),
            "Support Vector Machine (SVM)": SVC(
                kernel='linear', C=1.0, probability=True, random_state=self.random_state
            ),
            "Boosted Decision Tree (BDT)": GradientBoostingClassifier(
                n_estimators=70, learning_rate=0.1, max_depth=3, random_state=self.random_state
            ),
            "Neural Network (MLP)": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=600, random_state=self.random_state
            )
        }
        self.fitted_pipelines: Dict[str, Any] = {}
        self.fitted_models: Dict[str, Any] = {}

    def build_pipeline(self, model: Any, reduce_dim: bool = False, n_components: int = 25) -> Pipeline:
        """
        Build an end-to-end sklearn Pipeline encapsulating feature extraction
        and optional dimensionality reduction with the classifier to eliminate data leakage.
        """
        steps = [
            ('tfidf', TfidfVectorizer(max_features=400, ngram_range=(1, 2), min_df=2, sublinear_tf=True))
        ]
        if reduce_dim:
            steps.append(('svd', TruncatedSVD(n_components=n_components, random_state=self.random_state)))
        steps.append(('clf', clone(model)))
        return Pipeline(steps)

    def train_and_evaluate(self,
                           X: Union[np.ndarray, List[str], pd.Series],
                           y: Union[np.ndarray, List[str], pd.Series],
                           test_size: float = 0.25,
                           reduce_dim: bool = False,
                           n_components: int = 25) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Split dataset into train/test sets, train all 5 models using leakage-free Pipelines,
        and compute evaluation metrics.

        If X contains raw text strings, sklearn.pipeline.Pipeline is used to ensure
        TfidfVectorizer (and TruncatedSVD) fit ONLY on training splits/folds.
        If X is a precomputed numeric feature matrix, models are evaluated directly.
        """
        X_arr = np.array(X)
        y_arr = np.array(y)
        is_text_data = (len(X_arr) > 0 and isinstance(X_arr[0], (str, np.str_)))

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=self.random_state, stratify=y_arr
        )

        results = []
        detailed_results = {
            "test_y_true": y_test.tolist(),
            "models": {}
        }

        unique_classes = np.unique(y_arr)
        min_class_count = int(np.min(np.bincount(pd.Categorical(y_arr).codes)))
        n_splits = max(2, min(5, min_class_count)) if min_class_count >= 2 else 2

        for name, model in self.models.items():
            if is_text_data:
                pipe = self.build_pipeline(model, reduce_dim=reduce_dim, n_components=n_components)
                # Fit strictly on X_train to prevent leakage
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                self.fitted_pipelines[name] = pipe
                self.fitted_models[name] = pipe.named_steps['clf']
                eval_estimator = pipe
                eval_X = X_arr
            else:
                m = clone(model)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                self.fitted_models[name] = m
                eval_estimator = m
                eval_X = X_arr

            # Compute standard metrics
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

            # Leakage-free Cross-Validation
            try:
                cv = StratifiedKFold(n_splits=n_splits)
                cv_scores = cross_val_score(eval_estimator, eval_X, y_arr, cv=cv, scoring='accuracy')
                cv_mean = round(float(np.mean(cv_scores)), 4)
                cv_std = round(float(np.std(cv_scores)), 4)
            except Exception as cv_err:
                # Explicit error reporting: never silently substitute held-out test accuracy
                print(f"[Warning] Cross-validation failed for {name}: {cv_err}")
                cv_mean = float("nan")
                cv_std = float("nan")

            results.append({
                "Model": name,
                "Test Accuracy": round(acc, 4),
                "F1 Score (Macro)": round(f1_macro, 4),
                "F1 Score (Weighted)": round(f1_weighted, 4),
                "Precision (Macro)": round(prec_macro, 4),
                "Recall (Macro)": round(rec_macro, 4),
                "5-Fold CV Mean": cv_mean,
                "5-Fold CV Std": cv_std
            })

            detailed_results["models"][name] = {
                "y_pred": y_pred.tolist(),
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=unique_classes).tolist(),
                "labels": unique_classes.tolist(),
                "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            }

        summary_df = pd.DataFrame(results).sort_values(by="F1 Score (Weighted)", ascending=False)
        return summary_df, detailed_results

    def compare_feature_reduction(self,
                                  *args,
                                  **kwargs) -> pd.DataFrame:
        """
        Compare ML performance on Full Features vs. Reduced Latent Features.
        Directly fulfills IDC409 Requirement 4.

        Supports both:
          1. Leakage-safe call with raw text:
             compare_feature_reduction(raw_texts, y, n_components=25)
          2. Legacy call with pre-extracted matrices:
             compare_feature_reduction(X_full, X_reduced, y)
        """
        if len(args) == 2 and isinstance(args[0][0], (str, np.str_)):
            raw_texts, y = args
            n_components = kwargs.get("n_components", 25)
            full_df, _ = self.train_and_evaluate(raw_texts, y, reduce_dim=False)
            full_df['Feature Set'] = "Full TF-IDF (400 dims)"
            reduced_df, _ = self.train_and_evaluate(raw_texts, y, reduce_dim=True, n_components=n_components)
            reduced_df['Feature Set'] = f"Reduced LSA/SVD ({n_components} dims)"
        elif len(args) == 3 and isinstance(args[0][0], (str, np.str_)):
            raw_texts, y, n_components = args
            full_df, _ = self.train_and_evaluate(raw_texts, y, reduce_dim=False)
            full_df['Feature Set'] = "Full TF-IDF (400 dims)"
            reduced_df, _ = self.train_and_evaluate(raw_texts, y, reduce_dim=True, n_components=n_components)
            reduced_df['Feature Set'] = f"Reduced LSA/SVD ({n_components} dims)"
        elif len(args) == 3:
            X_full, X_reduced, y = args
            full_df, _ = self.train_and_evaluate(X_full, y)
            full_df['Feature Set'] = f"Full TF-IDF ({X_full.shape[1]} dims)"
            reduced_df, _ = self.train_and_evaluate(X_reduced, y)
            reduced_df['Feature Set'] = f"Reduced LSA/SVD ({X_reduced.shape[1]} dims)"
        elif "raw_texts" in kwargs and "y" in kwargs:
            raw_texts = kwargs["raw_texts"]
            y = kwargs["y"]
            n_components = kwargs.get("n_components", 25)
            full_df, _ = self.train_and_evaluate(raw_texts, y, reduce_dim=False)
            full_df['Feature Set'] = "Full TF-IDF (400 dims)"
            reduced_df, _ = self.train_and_evaluate(raw_texts, y, reduce_dim=True, n_components=n_components)
            reduced_df['Feature Set'] = f"Reduced LSA/SVD ({n_components} dims)"
        else:
            raise ValueError("Invalid arguments passed to compare_feature_reduction")

        combined = pd.concat([full_df, reduced_df], axis=0)
        return combined[['Model', 'Feature Set', 'Test Accuracy', 'F1 Score (Weighted)', '5-Fold CV Mean']]
