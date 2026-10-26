"""
run_pipeline.py
---------------
Master Orchestrator for IDC409 Faculty Profile Project.
Executes End-to-End Pipeline:
  1. Ethical Scraping & Data Ingestion (IISER Mohali + Complementary Institutes)
  2. SQLite Database Setup & CRUD Operations Demonstration (Read, Write, Modify)
  3. Preprocessing, Text Sanitization, & Normalization
  4. Exploratory Data Analysis & Visualization Generation
  5. Feature Engineering, Discriminative Term Analysis & Dimensionality Reduction (PCA/LSA)
  6. Machine Learning Model Benchmarking (5 Models + Feature Set Comparison)
  7. Unsupervised Research Clustering & Silhouette Evaluation
  8. Cosine Similarity Expertise Matching & Search
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

# Force UTF-8 on Windows terminal output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.database import FacultyDatabase
from src.scraper import EthicalFacultyScraper
from src.preprocessor import prepare_profile_dataframe
from src.eda import FacultyProfileEDA
from src.feature_engineering import FeaturePipeline
from src.models import DisciplineClassificationBenchmark
from src.clustering import ResearchClusterAnalyzer
from src.similarity_search import FacultySimilarityEngine


def run_pipeline(scrape: bool = True,
                 max_scrape: int = None,
                 use_cache: bool = True) -> None:
    print("=" * 70)
    print("🚀 Web Scraping II: Faculty Profiles — SQLite CRUD & ML Pipeline")
    print("=" * 70)

    project_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Ethical Web Scraping & Ingestion
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Data Ingestion & Ethical Web Scraping...")
    scraper = EthicalFacultyScraper(base_url="https://www.iisermohali.ac.in", delay=0.4)
    
    if scrape:
        iiser_profiles = scraper.scrape_all_iiser_mohali(max_limit=max_scrape)
    else:
        print("  Skipping live scrape; using cached/discovered profiles.")
        iiser_profiles = []

    # Complementary institutional profiles for cross-institution comparative analysis
    comp_profiles = scraper.get_complementary_institution_profiles()
    all_profiles = iiser_profiles + comp_profiles
    print(f"  Total Profiles Ingested: {len(all_profiles)} ({len(iiser_profiles)} IISER Mohali, {len(comp_profiles)} Complementary/IIT)")

    # -------------------------------------------------------------------------
    # STEP 2: SQLite Database & CRUD Demonstration (Rule 2)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] SQLite Database Storage & CRUD Verification...")
    db_path = os.path.join(data_dir, "faculty.db")
    db = FacultyDatabase(db_path)
    
    # Write / Insert records
    written_count = db.bulk_write_profiles(all_profiles)
    print(f"  [CREATE/WRITE] Inserted/Synchronized {written_count} profiles into SQLite: {db_path}")

    # Read demonstration
    sample_id = all_profiles[0]['faculty_id'] if all_profiles else "iiserm_dps_abhishek"
    read_record = db.read_profile(sample_id)
    if read_record:
        print(f"  [READ] Verified retrieval of '{read_record['name']}' ({read_record['department']})")

    # Modify / Update demonstration
    temp_test_id = "test_crud_faculty"
    db.write_profile({
        "faculty_id": temp_test_id,
        "name": "Dr. Temporary Profiler",
        "institution": "IISER Mohali",
        "department": "Department of Physical Sciences",
        "department_code": "DPS",
        "designation": "Assistant Professor",
        "research_area": "Computational Physics"
    })
    db.modify_profile(temp_test_id, {"designation": "Associate Professor"})
    modified_check = db.read_profile(temp_test_id)
    print(f"  [MODIFY] Updated designation: '{modified_check['designation']}'")
    db.delete_profile(temp_test_id)
    print(f"  [DELETE] Deleted temporary profile '{temp_test_id}' successfully.")

    # Export to Pandas DataFrame (Rule 1: numpy/pandas)
    df_raw = db.to_dataframe()
    print(f"  [PANDAS EXPORT] DataFrame shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # -------------------------------------------------------------------------
    # STEP 3: Preprocessing & Text Normalization
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Preprocessing Text & Feature Synthesis...")
    df_clean = prepare_profile_dataframe(df_raw)
    cleaned_csv_path = os.path.join(processed_dir, "faculty_cleaned.csv")
    df_clean.to_csv(cleaned_csv_path, index=False)
    print(f"  Cleaned dataset saved to: {cleaned_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 4: Exploratory Data Analysis & Visualizations (Rule 3)
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Executing Exploratory Data Analysis (EDA)...")
    eda = FacultyProfileEDA(df_clean)
    plots = eda.generate_all_plots()
    print(f"  Generated {len(plots)} visualization plots in reports/figures/:")
    for name, path in plots.items():
        print(f"    - {name}: {os.path.basename(path)}")

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # STEP 5: Independent Preprocessing, Train/Test Split & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Train/Test Split & Independent Feature Preprocessing (Zero Data Leakage)...")
    
    # Filter for profiles with meaningful text and clean department codes
    valid_mask = df_clean['feature_text'].str.strip().str.len() > 10
    df_model = df_clean[valid_mask].copy()
    
    # Merge rare departments if necessary or select primary departments
    top_depts = df_model['department_code'].value_counts()
    eligible_depts = top_depts[top_depts >= 4].index.tolist()
    df_model = df_model[df_model['department_code'].isin(eligible_depts)].reset_index(drop=True)
    print(f"  Dataset for Machine Learning: {len(df_model)} profiles across {len(eligible_depts)} disciplines: {eligible_depts}")

    raw_texts = df_model['feature_text'].values
    y = df_model['department_code'].values

    # CRITICAL: Train/Test split BEFORE fitting any TF-IDF or SVD transformation
    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_texts, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"  Holdout Split (Zero-Leakage): {len(X_train_raw)} train samples, {len(X_test_raw)} test samples")

    # Fit FeaturePipeline STRICTLY on training split (no test data seen during fit)
    pipeline = FeaturePipeline(max_features=400, ngram_range=(1, 2), min_df=2, n_components=25)
    X_train_tfidf = pipeline.fit_transform(X_train_raw.tolist())
    X_test_tfidf = pipeline.transform(X_test_raw.tolist())
    print(f"  Train TF-IDF Feature Matrix: shape = {X_train_tfidf.shape} (Fitted strictly on train split)")
    print(f"  Test TF-IDF Feature Matrix:  shape = {X_test_tfidf.shape} (Transform only, zero leakage)")

    # Feature Understanding: Chi-Square Discriminative Terms strictly on training data
    print("\n  Top Discriminative Terms by Discipline (Chi-Square Analysis on Training Set):")
    top_terms = pipeline.analyze_top_terms_by_department(X_train_tfidf, y_train.tolist(), top_k=5)
    for dept, terms in top_terms.items():
        terms_str = ", ".join([f"{t} ({score})" for t, score in terms])
        print(f"    [{dept}]: {terms_str}")

    # Dimensionality Reduction: SVD / PCA fitted strictly on training data
    X_train_reduced = pipeline.reduce_dimensions(X_train_tfidf)
    X_test_reduced = pipeline.transform_dimensions(X_test_tfidf)
    var_info = pipeline.get_explained_variance()
    print(f"  Reduced Latent Semantic Components: shape = {X_train_reduced.shape}")
    print(f"  Cumulative Variance Explained by {var_info.get('n_components')} components: {var_info.get('cumulative_variance', 0):.2%}")

    # -------------------------------------------------------------------------
    # STEP 6: Machine Learning Models & Scikit-Learn Pipeline Evaluation (Rule 5 & 6)
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Machine Learning Classification Benchmarking (Scikit-Learn Pipelines)...")
    bench = DisciplineClassificationBenchmark(random_state=42)

    # Scikit-learn Pipeline encapsulates TfidfVectorizer & TruncatedSVD for 100% leakage-free CV and holdout testing
    comparison_df = bench.compare_feature_reduction(raw_texts, y, n_components=25)
    print("\n  --- MODEL BENCHMARK RESULTS (Accuracy & F1 Score) ---")
    print(comparison_df.to_string(index=False))
    
    benchmark_csv = os.path.join(processed_dir, "ml_benchmark_results.csv")
    comparison_df.to_csv(benchmark_csv, index=False)
    print(f"\n  Detailed benchmark metrics saved to: {benchmark_csv}")

    # -------------------------------------------------------------------------
    # STEP 7: Unsupervised Clustering & Silhouette Evaluation
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Unsupervised Clustering & Silhouette Score Analysis...")
    # For full unsupervised corpus exploration and semantic search index:
    full_pipeline = FeaturePipeline(max_features=400, ngram_range=(1, 2), min_df=2, n_components=25)
    X_full_tfidf = full_pipeline.fit_transform(raw_texts.tolist())

    clusterer = ResearchClusterAnalyzer(random_state=42)
    clustering_eval_df = clusterer.find_optimal_k(X_full_tfidf, k_range=range(3, min(7, len(df_model) // 5)))
    print(clustering_eval_df.to_string(index=False))

    topics = clusterer.extract_cluster_topics(full_pipeline.feature_names, top_n=5)
    print(f"\n  Extracted Research Topic Themes (Optimal k={clusterer.best_k}, Silhouette={clusterer.best_silhouette:.4f}):")
    for c_id, info in topics.items():
        print(f"    Cluster {c_id} [{info['topic_label']}]: {', '.join(info['top_keywords'])}")

    # -------------------------------------------------------------------------
    # STEP 8: Cosine Similarity Expertise Matching (AI/ML Task)
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Similarity Search & Faculty Expertise Retrieval...")
    engine = FacultySimilarityEngine(df_model, X_full_tfidf, full_pipeline.vectorizer)

    test_queries = [
        "quantum entanglement and condensed matter physics",
        "microbial genomics, bioinformatics and protein structure",
        "differential equations and algebraic geometry",
        "climate change, environmental geochemistry and hydrology"
    ]

    for q in test_queries:
        print(f"\n  Query: '{q}'")
        matches = engine.search_by_query(q, top_n=2)
        for m in matches:
            print(f"    -> {m['name']} ({m['institution']} - {m['department']}) [Similarity: {m['similarity_score']:.4f}]")
            print(f"       Area: {m['research_area'][:90]}...")

    print("\n" + "=" * 70)
    print("✅ Pipeline Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IDC409 Faculty Profile Analysis Pipeline")
    parser.add_argument("--no-scrape", dest="scrape", action="store_false", help="Skip live web scraping")
    parser.add_argument("--max-scrape", type=int, default=None, help="Maximum profiles to scrape (for testing)")
    parser.set_defaults(scrape=True)
    args = parser.parse_args()

    run_pipeline(scrape=args.scrape, max_scrape=args.max_scrape)
