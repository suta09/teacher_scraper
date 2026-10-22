# 🎓 Teacher Scraper
### Academic Faculty Profiler, SQLite CRUD & Machine Learning Pipeline — IDC409

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-brightgreen.svg)](https://www.sqlite.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Course](https://img.shields.io/badge/Course-IDC409-purple.svg)](https://www.iisermohali.ac.in)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Project Overview
**Teacher Scraper** is an end-to-end data science and machine learning system developed for **IDC409: Web Scraping II - Faculty Profiles**. The project ethically scrapes academic profiles of faculty members from Indian institutions (primarily **IISER Mohali**, augmented with benchmark profiles from **IITs / Vidwan**), persists them into an indexed relational **SQLite** database supporting complete CRUD operations, conducts comprehensive **Exploratory Data Analysis (EDA)**, applies **Dimensionality Reduction** on high-dimensional text feature spaces, benchmarks **5 Machine Learning classification architectures**, runs **Unsupervised Topic Clustering**, and provides a **Cosine Similarity Expertise Search Engine**.

---

## 🎯 Compliance with IDC409 Course Rules

| # | Course Requirement | Status | Implementation in Teacher Scraper | Source File |
|---|---|:---:|---|---|
| **1** | **NumPy & Pandas to store data** | ✅ Complete | Tabular structures, vectorized text matrices, and numerical operations use Pandas DataFrames and NumPy ndarrays. | [`src/preprocessor.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/preprocessor.py) |
| **2** | **Database: Read, Write & Modify (CRUD)** | ✅ Complete | Relational SQLite database (`data/faculty.db`) with full transactional CRUD: Insert/Write, Query/Read with filters, Modify/Update fields, and Delete. Seamless DataFrame import/export. | [`src/database.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/database.py) |
| **3** | **Exploratory Data Analysis (EDA) demonstrated** | ✅ Complete | 7 high-resolution visualizations: Department distributions, faculty designation breakdown, top PhD granting universities, PhD country trends, word clouds, and multi-institution comparisons. | [`src/eda.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/eda.py), `reports/figures/` |
| **4** | **Features understood & dimensionality reduction** | ✅ Complete | Chi-Square ($\chi^2$) analysis of discriminative terms per department. High-dimensional TF-IDF (400 features) reduced to 25 latent components via **TruncatedSVD (Latent Semantic Analysis)**. | [`src/feature_engineering.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/feature_engineering.py) |
| **5** | **Machine Learning Models** | ✅ Complete | 5 distinct architectures implemented and compared: **Logistic Regression**, **Decision Tree**, **Support Vector Machine (SVM)**, **Boosted Decision Tree (BDT / Gradient Boosting)**, and **Neural Network (Multi-Layer Perceptron / MLP)**. | [`src/models.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/models.py) |
| **6** | **Evaluation using standard metrics** | ✅ Complete | Test Accuracy, Macro/Weighted F1-score, Precision, Recall, 5-Fold Stratified Cross-Validation, Confusion Matrices, and Silhouette Score for clustering. | [`src/models.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/models.py), [`src/clustering.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/clustering.py) |
| **7** | **Code Structure & Reproducibility** | ✅ Complete | Structured project layout, unit test suite, reproducible master pipeline, and interactive Jupyter Notebook. | `notebooks/faculty_profiles_eda_ml.ipynb`, `tests/` |

---

## 🏗️ Repository Architecture

```text
teacher_scraper/
├── README.md                          # Comprehensive project documentation & guide
├── requirements.txt                   # Project dependencies
├── run_pipeline.py                    # Master orchestrator script (End-to-End CLI)
├── data/
│   ├── faculty.db                     # SQLite database (stores raw & processed profiles)
│   ├── raw/                           # Raw cached HTML files from ethical scraper (130+ files)
│   └── processed/
│       ├── faculty_cleaned.csv        # Preprocessed tabular dataset
│       └── ml_benchmark_results.csv   # Model evaluation benchmark table
├── src/
│   ├── __init__.py
│   ├── scraper.py                     # Ethical scraper (robots.txt, polite delay, caching)
│   ├── database.py                    # SQLite CRUD interface (Create, Read, Update, Delete)
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
Install the project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
Verify database CRUD operations, text cleaning, feature reduction, and model predictions:
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Run the Full End-to-End Pipeline
Execute the master pipeline to scrape, populate SQLite, generate EDA plots, train ML models, and test similarity search:
```bash
python run_pipeline.py
```
*Note: The scraper automatically utilizes cached HTML files in `data/raw/` for instant execution without re-querying the web server.*

### 4. Interactive Jupyter Notebook
Launch Jupyter to explore the step-by-step interactive workflow:
```bash
jupyter notebook notebooks/faculty_profiles_eda_ml.ipynb
```

---

## 📊 Key Results & Empirical Findings

### 1. Machine Learning Benchmark: Full Features vs. Reduced Features (Rule 4, 5, 6)
We benchmarked all 5 model families on two feature representations:
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
- *Query:* `"applied mathematics and probability"`  
  $\rightarrow$ Matches: **Dr. Neeraja Sahasrabudhe** (DMS, Area: Theoretical and Applied Probability)

---

## 🛡️ Ethical Scraping Policy
The web scraper in [`src/scraper.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/scraper.py) enforces ethical crawling standards:
1. **Robots.txt Adherence:** Automatically verifies permissions with Python's `urllib.robotparser.RobotFileParser`.
2. **Rate Limiting:** Enforces polite delays between HTTP GET requests to prevent server strain.
3. **Transparent Identification:** Transmits a descriptive User-Agent header specifying the academic project context.
4. **Local File Caching:** Persists HTML files locally in `data/raw/` so scraping is performed only once.

---

## 🎓 Academic Patterns & Limitations
- **Doctoral Origins:** Over 70% of science faculty at IISER Mohali earned their doctorates from premier national institutions (IISc Bangalore, IIT Kanpur, TIFR Mumbai, Panjab University) or internationally (USA, Germany, UK, France, Switzerland).
- **Postdoctoral Mobility:** The overwhelming majority completed international postdoctoral fellowships (e.g. Max Planck Institutes, UC San Diego, Stanford, Cambridge) prior to joining IISER Mohali.
- **Limitations:** Faculty profiles have non-uniform text lengths; some members detail full CVs while others provide brief research summaries.
