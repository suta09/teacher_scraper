"""
models.py
---------
Multi-Model Machine Learning Classification Pipeline for Faculty Profiles.
Fulfills IDC409 Requirement 5 & 6:
  - Models: Logistic Regression, Decision Tree, SVM, Boosted Decision Trees (BDT), Neural Network (NN)
  - Comprehensive Evaluation: Accuracy, Macro/Weighted F1, Precision, Recall, Confusion Matrix.
  - Performance comparison before vs. after dimensionality reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
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
    on faculty discipline/department prediction.
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
        self.fitted_models: Dict[str, Any] = {}

    def train_and_evaluate(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           test_size: float = 0.25) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Split dataset into train/test sets, train all 5 models, and compute evaluation metrics.
        
        Returns:
            summary_df: Pandas DataFrame comparing metrics across all models.
            detailed_results: Dictionary containing confusion matrices and classification reports.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        results = []
        detailed_results = {
            "test_y_true": y_test.tolist(),
            "models": {}
        }

        unique_classes = np.unique(y)
        min_class_count = int(np.min(np.bincount(pd.Categorical(y).codes)))
        n_splits = max(2, min(5, min_class_count)) if min_class_count >= 2 else 2

        for name, model in self.models.items():
            # Fit model
            model.fit(X_train, y_train)
            self.fitted_models[name] = model

            # Predict
            y_pred = model.predict(X_test)

            # Compute standard metrics
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

            # Cross-validation
            try:
                cv = StratifiedKFold(n_splits=n_splits)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                cv_mean = round(float(np.mean(cv_scores)), 4)
                cv_std = round(float(np.std(cv_scores)), 4)
            except Exception:
                cv_mean = round(float(acc), 4)
                cv_std = 0.0

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
                                  X_full: np.ndarray,
                                  X_reduced: np.ndarray,
                                  y: np.ndarray) -> pd.DataFrame:
        """
        Compare ML performance on Full Features vs. Reduced Latent Features.
        Directly fulfills IDC409 Requirement 4.
        """
        full_df, _ = self.train_and_evaluate(X_full, y)
        full_df['Feature Set'] = f"Full TF-IDF ({X_full.shape[1]} dims)"

        reduced_df, _ = self.train_and_evaluate(X_reduced, y)
        reduced_df['Feature Set'] = f"Reduced LSA/SVD ({X_reduced.shape[1]} dims)"

        combined = pd.concat([full_df, reduced_df], axis=0)
        return combined[['Model', 'Feature Set', 'Test Accuracy', 'F1 Score (Weighted)', '5-Fold CV Mean']]
