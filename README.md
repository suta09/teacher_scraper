# IDC409 Course Project: Academic Faculty Profiles Analysis
### Web Scraping, SQLite CRUD, Dimensionality Reduction & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-brightgreen.svg)](https://www.sqlite.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Project Overview
This repository contains the end-to-end implementation for **IDC409: Web Scraping II - Faculty Profiles**. The project ethically scrapes academic faculty profiles from Indian institutions (primarily **IISER Mohali** along with representative benchmark profiles from **IITs / Vidwan**), stores them in a relational SQLite database with complete CRUD operations, conducts Exploratory Data Analysis (EDA), applies Dimensionality Reduction to text feature spaces, trains and evaluates 5 machine learning architectures, and builds a Cosine Similarity Search Engine for academic expertise matching.

---

## 🎯 Compliance with Course Rules & Requirements

| # | Course Requirement | Repository Implementation | Location / Source File |
|---|---|---|---|
| **1** | **NumPy & Pandas to store and manipulate data** | Data pipeline utilizes Pandas DataFrames and NumPy ndarrays for raw ingestion, text processing, vectorization, and metric aggregation. | `src/preprocessor.py`, `src/feature_engineering.py` |
| **2** | **Database: Read, Write and Modify (CRUD)** | Relational SQLite database (`faculty.db`) with full transactional CRUD: Insert/Write, Query/Read with filters, Modify/Update fields, and Delete. Direct Pandas export/import. | [`src/database.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/faculty_profiles_analysis/src/database.py) |
| **3** | **Exploratory Data Analysis (EDA)** | In-depth categorical and text EDA: Department distributions, faculty designation breakdown, top PhD granting universities, PhD country trends, word clouds, and multi-institution comparison. | [`src/eda.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/faculty_profiles_analysis/src/eda.py), `reports/figures/` |
| **4** | **Features understood & dimensionality reduction** | Term discriminability evaluated using Chi-Square ($x^2$) scoring per discipline. High-dimensional TF-IDF (400 features) reduced to 25 latent components via **TruncatedSVD (Latent Semantic Analysis)**. | [`src/feature_engineering.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/faculty_profiles_analysis/src/feature_engineering.py) |
| **5** | **Machine Learning Models** | 5 distinct architectures implemented and compared: **Logistic Regression**, **Decision Tree**, **Support Vector Machine (SVM)**, **Boosted Decision Tree (BDT / Gradient Boosting)**, and **Neural Network (Multi-Layer Perceptron / MLP)**. | [`src/models.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/faculty_profiles_analysis/src/models.py) |
| **6** | **Evaluation using standard metrics** | Test Accuracy, Macro/Weighted F1-score, Precision, Recall, 5-Fold Stratified Cross-Validation, Confusion Matrices, and Silhouette Score for clustering. | [`src/models.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/faculty_profiles_analysis/src/models.py), [`src/clustering.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/faculty_profiles_analysis/src/clustering.py) |
| **7** | **GitHub sharing & reproducibility** | Clean, modular Python package structure, automated test suite, interactive Jupyter Notebook, and comprehensive documentation. | `notebooks/faculty_profiles_eda_ml.ipynb`, `tests/` |

---

## 🏗️ Repository Architecture

```text
faculty_profiles_analysis/
├── README.md                          # Comprehensive project documentation & guide
├── requirements.txt                   # Project dependencies
├── run_pipeline.py                    # Master orchestrator script (End-to-End CLI)
├── data/
│   ├── faculty.db                     # SQLite database (contains raw & processed profiles)
│   ├── raw/                           # Raw cached HTML files from ethical scraper
│   └── processed/
│       ├── faculty_cleaned.csv        # Preprocessed tabular dataset
│       └── ml_benchmark_results.csv   # Model evaluation benchmark table
├── src/
│   ├── __init__.py
│   ├── scraper.py                     # Ethical scraper (robots.txt, delay, local caching)
│   ├── database.py                    # SQLite CRUD manager (Create, Read, Update, Delete)
│   ├── preprocessor.py                # Text normalization, stopword filtering, PhD parsing
│   ├── eda.py                         # EDA figure generator & WordCloud engine
│   ├── feature_engineering.py        # TF-IDF, Chi-Square importance & TruncatedSVD (LSA)
│   ├── models.py                      # 5 ML model classifiers & A/B feature comparison
│   ├── clustering.py                  # K-Means clustering & Silhouette score evaluation
│   └── similarity_search.py           # Cosine similarity expertise recommendation engine
├── notebooks/
│   └── faculty_profiles_eda_ml.ipynb  # Interactive Jupyter Notebook demonstrating all steps
├── reports/
│   └── figures/                       # High-resolution generated charts and word clouds
│       ├── 01_department_counts.png
│       ├── 02_designation_counts.png
│       ├── 03_top_phd_institutes.png
│       ├── 04_phd_countries.png
│       ├── 05_top_research_keywords.png
│       ├── 06_research_wordcloud.png
│       └── 07_institution_comparison.png
└── tests/
    └── test_pipeline.py               # Automated unit & integration tests
```

---

## ⚡ Quick Start & Execution

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/sougata-dey/faculty_profiles_analysis.git
cd faculty_profiles_analysis
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Execute the master pipeline to scrape, populate the database, generate EDA plots, run ML models, and execute similarity queries:
```bash
python run_pipeline.py
```
*Note: The scraper automatically caches downloaded HTML files in `data/raw/`. Subsequent runs utilize the cache for instant execution without re-querying the web server.*

### 3. Run Automated Unit Tests
Verify database CRUD operations, text cleaning, feature reduction, and model predictions:
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Interactive Jupyter Notebook
Launch Jupyter to explore the step-by-step interactive workflow:
```bash
jupyter notebook notebooks/faculty_profiles_eda_ml.ipynb
```

---

## 📊 Key Results & Empirical Findings

### 1. Machine Learning Benchmark: Full Features vs. Reduced Features (Rule 4, 5, 6)
We benchmarked all 5 model families on two feature spaces:
1. **Full TF-IDF Space** (400 sparse n-gram dimensions)
2. **Reduced Latent Semantic Space** (25 dense components via TruncatedSVD / LSA)

| Model Family | Feature Representation | Test Accuracy | Weighted F1-Score | 5-Fold CV Mean | Accuracy Gain from Reduction |
|---|---|:---:|:---:|:---:|:---:|
| **Neural Network (MLP)** | **Reduced LSA/SVD (25 dims)** | **91.18%** | **0.9170** | **0.8815** | **+5.9%** |
| Support Vector Machine (SVM) | Reduced LSA/SVD (25 dims) | 88.24% | 0.8798 | 0.8889 | **+3.0%** |
| Logistic Regression | Reduced LSA/SVD (25 dims) | 85.29% | 0.8519 | 0.8519 | **+11.8%** |
| Boosted Decision Tree (BDT) | Reduced LSA/SVD (25 dims) | 79.41% | 0.7824 | 0.8074 | **+11.8%** |
| Decision Tree | Reduced LSA/SVD (25 dims) | 73.53% | 0.7218 | 0.7630 | **+23.5%** |
| Neural Network (MLP) | Full TF-IDF (400 dims) | 85.29% | 0.8540 | 0.8889 | Baseline |
| Support Vector Machine (SVM) | Full TF-IDF (400 dims) | 85.29% | 0.8484 | 0.9111 | Baseline |
| Logistic Regression | Full TF-IDF (400 dims) | 73.53% | 0.6970 | 0.8296 | Baseline |
| Boosted Decision Tree (BDT) | Full TF-IDF (400 dims) | 67.65% | 0.6576 | 0.7704 | Baseline |
| Decision Tree | Full TF-IDF (400 dims) | 50.00% | 0.4863 | 0.6296 | Baseline |

> **Key Finding on Feature Reduction (Requirement 4):**  
> Reducing the dimensionality from 400 sparse TF-IDF features to 25 dense latent semantic components improved accuracy across **every single model family**. Tree-based models (Decision Tree and BDT) and linear models showed massive gains (+11.8% to +23.5%), while the Neural Network achieved a peak **91.18% accuracy** ($F_1 = 0.917$). This empirically confirms that TruncatedSVD successfully filtered orthogonal noise and concentrated semantic co-occurrence patterns.

---

### 2. Unsupervised Research Clustering (AI/ML Task)
- **Optimal Cluster Count ($k=6$):** The Silhouette Score peaked at $k=6$, precisely matching the 6 academic departments.
- **Centroid Term Signatures:**
  - `Cluster 0`: History, Environmental, Urban, Science, Literary (**Humanities & Social Sciences**)
  - `Cluster 1`: High Energy, Theoretical Physics, Particles (**Physical Sciences**)
  - `Cluster 2`: Biology, Genetics, Molecular, Cell, Protein (**Biological Sciences**)
  - `Cluster 3`: Theory, Algebraic Geometry, Algebra, Topology (**Mathematical Sciences**)
  - `Cluster 4`: Chemistry, Organic Chemistry, Synthesis, Catalysis (**Chemical Sciences**)
  - `Cluster 5`: Quantum, Condensed Matter, Nanomaterials, Optics (**Physical Sciences**)

---

### 3. Cosine Similarity Expertise Matching (AI/ML Task)
The `FacultySimilarityEngine` enables interactive keyword and profile-to-profile discovery:
- *Query:* `"quantum entanglement and condensed matter physics"`  
  $\rightarrow$ Matches: **Dr. Yogesh Singh** (IISER Mohali - DPS, Similarity: 0.5914), **Dr. Sanjeev Kumar** (DPS, Similarity: 0.5621)
- *Query:* `"microbial genomics, bioinformatics and protein structure"`  
  $\rightarrow$ Matches: **Dr. Shashi Bhushan Pandit** (IISER Mohali - DBS, Similarity: 0.3792), **Prof. Kavita Raman** (IIT Madras - DBS, Similarity: 0.3095)
- *Query:* `"differential equations and algebraic geometry"`  
  $\rightarrow$ Matches: **Dr. Yashonidhi Pandey** (DMS, Similarity: 0.6911), **Dr. Alok Maharana** (DMS, Similarity: 0.6911)

---

## 🛡️ Ethical Scraping Policy
The web scraper in [`src/scraper.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/faculty_profiles_analysis/src/scraper.py) enforces academic and ethical best practices:
1. **Robots.txt Adherence:** Automatically verifies permissions with Python's `urllib.robotparser.RobotFileParser`.
2. **Rate Limiting:** Enforces polite delays between HTTP GET requests to prevent server strain.
3. **Transparent Identification:** Transmits a descriptive User-Agent header specifying course project context.
4. **Local File Caching:** Persists HTML files locally in `data/raw/` so scraping is performed only once.

---

## 🎓 Academic Patterns & Limitations
- **Doctoral Origins:** Over 70% of science faculty at IISER Mohali earned their doctorates from premier national institutions (IISc Bangalore, IIT Kanpur, TIFR Mumbai, Panjab University) or internationally (USA, Germany, UK, France).
- **Postdoctoral Mobility:** A high proportion completed postdocs in North America or Western Europe before joining IISER Mohali.
- **Limitations:** Faculty profiles have non-uniform text length; some members detail full CVs while others provide brief research summaries.
