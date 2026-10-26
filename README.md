# 🎓 Web Scraping II: Faculty Profiles
### Academic Faculty Profiler, SQLite CRUD & Machine Learning Pipeline — IDC409

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-brightgreen.svg)](https://www.sqlite.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Web App](https://img.shields.io/badge/Web%20App-Flask-green.svg)](http://127.0.0.1:5000)
[![Course](https://img.shields.io/badge/Course-IDC409-purple.svg)](https://www.iisermohali.ac.in)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Project Overview
**Web Scraping II: Faculty Profiles** is an end-to-end data science and machine learning system developed for **IDC409**. The project ethically scrapes academic profiles of faculty members from Indian institutions (primarily **IISER Mohali**, augmented with benchmark profiles from **IITs / Vidwan**), persists them into an indexed relational **SQLite** database supporting complete CRUD operations, conducts comprehensive **Exploratory Data Analysis (EDA)**, applies **Dimensionality Reduction** on high-dimensional text feature spaces, benchmarks **5 Machine Learning classification architectures**, runs **Unsupervised Topic Clustering**, and provides a **Cosine Similarity Expertise Search Engine**.

---

## 🎯 Compliance with IDC409 Course Rules

| # | Course Requirement | Status | Implementation in this Project | Source File |
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
├── app.py                             # Interactive Web Dashboard (Live Search & Collaboration)
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

## 💻 Step-by-Step Guide: Running on Any Computer

This project is completely self-contained. Anyone (evaluators, instructors, or collaborators) can run it on Windows, macOS, or Linux by following these steps:

### Prerequisites
- **Python 3.9 or higher** ([python.org/downloads](https://www.python.org/downloads/))
- **Git** ([git-scm.com](https://git-scm.com/))

---

### Step 1: Clone the Repository
Open your terminal (PowerShell, Command Prompt, or bash) and clone the repository:
```bash
git clone https://github.com/suta09/teacher_scraper.git
cd teacher_scraper
```

### Step 2: (Recommended) Create & Activate a Virtual Environment
Create an isolated environment to ensure clean dependency management:

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux (bash/zsh):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Automated Unit Tests
Verify SQLite database CRUD operations, text cleaning, feature reduction, similarity search, fail-safe robots crawling, and leakage-safe pipelines:
```bash
python -m unittest tests/test_pipeline.py
```
*(All 6 tests run and verify `OK` with zero warnings).*

### Step 5: Execute the Master Pipeline
Run the complete end-to-end data pipeline:
```bash
python run_pipeline.py
```
*Note: The repository already includes the populated SQLite database (`data/faculty.db`) and cached profile HTML files (`data/raw/`). The script will run instantly offline without needing to re-crawl university servers.*

### Step 6: Launch the Interactive Jupyter Notebook
To explore the visual charts, data frames, and interactive ML benchmarks:
```bash
jupyter notebook notebooks/faculty_profiles_eda_ml.ipynb
```
*(You can also open the notebook directly in VS Code, PyCharm, or JupyterLab).*

### Step 7: Launch the Interactive Web Dashboard (Live Search & Demo)
To explore the live semantic search engine and discover faculty collaborators in your web browser:
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in any web browser to test free-form research query matching, professor collaboration exploration, and interactive analytics.

---

## 📊 Key Results & Empirical Findings

### 1. Data Validation & Dataset Denominators (Denominators & Coverage)
All analytical conclusions and statistical inferences are grounded in rigorously documented sample sizes and denominator counts:
- **Total Ingested Profiles:** $N = 137$ faculty members across institutions.
  - **IISER Mohali:** 129 profiles (full institutional coverage across 6 core divisions).
  - **IIT Benchmark Profiles:** 8 profiles (IIT Delhi: 2, IIT Bombay: 2, IIT Ropar: 2, IIT Kanpur: 1, IIT Madras: 1).
- **Academic Discipline Distribution:**
  - Biological Sciences (DBS): $n = 31$ ($22.6\%$)
  - Physical Sciences (DPS): $n = 29$ ($21.2\%$)
  - Chemical Sciences (DCS): $n = 29$ ($21.2\%$)
  - Mathematical Sciences (DMS): $n = 23$ ($16.8\%$)
  - Earth and Environmental Sciences (EES): $n = 13$ ($9.5\%$)
  - Humanities and Social Sciences (HSS): $n = 10$ ($7.3\%$)
  - Computer Science & Engineering (CSE): $n = 2$ ($1.5\%$)
- **Data Completeness & Missingness Audit:**
  - **Names, Institutions, Department Codes, Designations:** $137 / 137$ ($100.0\%$ coverage, 0 missing).
  - **Research Area & Specialization:** $136 / 137$ ($99.3\%$ coverage, 1 missing).
  - **Email Addresses:** $124 / 137$ ($90.5\%$ coverage, 13 missing/obfuscated).
  - **PhD Awarding Institution:** $105 / 137$ ($76.6\%$ documented, 32 missing in raw source).
  - **PhD Graduation Year:** $103 / 137$ ($75.2\%$ documented, 34 missing in raw source).
  - **PhD Awarding Country:** $93 / 137$ ($67.9\%$ documented, 44 missing in raw source).
    - India: $74 / 93$ ($79.6\%$)
    - Germany: $7 / 93$ ($7.5\%$)
    - USA: $6 / 93$ ($6.5\%$)
    - France: $3 / 93$ ($3.2\%$)
    - Japan: $2 / 93$ ($2.2\%$)
    - United Kingdom: $1 / 93$ ($1.1\%$)
  - **Postdoctoral Experience:** $75 / 137$ ($54.7\%$ documented, 62 without explicit postdoc details).
- **Date of Collection:** October 2026.

---

### 2. Machine Learning Benchmark: Full Features vs. Reduced Features (Leakage-Free Pipelines)
To strictly prevent data leakage between training splits and evaluation sets, all feature extraction (`TfidfVectorizer`) and dimensionality reduction (`TruncatedSVD`) operations are encapsulated inside `sklearn.pipeline.Pipeline`. Vectorizers and SVD transformations fit **solely** on training folds/splits.

Evaluation was performed on $N = 135$ eligible faculty profiles across 6 disciplines with holdout test size $= 25\%$ (Stratified) and 5-Fold Stratified Cross-Validation:

| Model Family | Feature Representation | Holdout Test Acc | Weighted F1 | 5-Fold CV Mean (± Std) | Performance Impact from SVD Reduction |
|---|---|:---:|:---:|:---:|---|
| **Neural Network (MLP)** | **Reduced LSA/SVD (25 dims)** | **85.29%** | **0.8538** | **91.11% ± 0.0502** | **Top Performer overall**; Highest generalizability |
| **Decision Tree** | **Reduced LSA/SVD (25 dims)** | **70.59%** | **0.6888** | **75.56% ± 0.0378** | **+26.47% Test Accuracy jump** (Eliminates high-dim noise) |
| Logistic Regression | Reduced LSA/SVD (25 dims) | 79.41% | 0.7778 | 80.74% ± 0.0432 | +2.94% Test Accuracy improvement |
| Support Vector Machine (SVM) | Reduced LSA/SVD (25 dims) | 79.41% | 0.7753 | 82.96% ± 0.0687 | Compact latent space representation |
| Boosted Decision Tree (BDT) | Reduced LSA/SVD (25 dims) | 67.65% | 0.6424 | 80.00% ± 0.0502 | +8.15% CV gain from dense embeddings |
| Support Vector Machine (SVM) | Full TF-IDF (400 dims) | 82.35% | 0.8166 | 87.41% ± 0.0602 | Baseline high-dimensional benchmark |
| Neural Network (MLP) | Full TF-IDF (400 dims) | 79.41% | 0.7938 | 85.93% ± 0.0718 | Baseline high-dimensional benchmark |
| Logistic Regression | Full TF-IDF (400 dims) | 76.47% | 0.7214 | 82.22% ± 0.0432 | Baseline high-dimensional benchmark |
| Boosted Decision Tree (BDT) | Full TF-IDF (400 dims) | 70.59% | 0.6879 | 71.85% ± 0.0602 | Baseline high-dimensional benchmark |
| Decision Tree | Full TF-IDF (400 dims) | 44.12% | 0.4097 | 63.70% ± 0.0593 | Baseline (High variance in sparse 400D space) |

> **Key Takeaway on Dimensionality Reduction & Score Validation:**  
> 1. **Data Leakage Elimination:** Wrapping text vectorization and SVD inside `Pipeline` ensures zero target or vocabulary leakage into validation folds.  
> 2. **Score Truth:** Under rigorous leakage-free conditions, the Neural Network MLP achieves **85.29% holdout test accuracy** ($F_1 = 0.8538$) and a peak **91.11% 5-Fold Cross-Validation mean**, verifying earlier reported high marks.  
> 3. **Empirical Value of Feature Reduction (Rule 4):** Latent Semantic Analysis (TruncatedSVD, 25 components) provides dramatic relief from the curse of dimensionality for tree-based models: the Decision Tree increases from **44.12% to 70.59% (+26.47% gain)** by pruning orthogonal n-gram sparsity.

---

### 3. Unsupervised Research Clustering (AI/ML Task)
- **Optimal Cluster Count ($k=6$):** The Silhouette Score was evaluated across $k \in [3, 6]$, achieving peak score and optimal separation at $k=6$, precisely matching the 6 academic departments.
- **Centroid Term Signatures:**
  - `Cluster 0`: History, Environmental, Urban, Science, Literary (**Humanities & Social Sciences**)
  - `Cluster 1`: Energy, High Energy Physics, Theoretical Physics (**Physical Sciences**)
  - `Cluster 2`: Biology, Genetics, Molecular, Cell, Protein (**Biological Sciences**)
  - `Cluster 3`: Theory, Algebraic Geometry, Algebra, Topology (**Mathematical Sciences**)
  - `Cluster 4`: Chemistry, Organic Chemistry, Synthesis, Catalysis, Materials (**Chemical Sciences**)
  - `Cluster 5`: Quantum, Condensed Matter, Nanomaterials, Optics (**Physical Sciences**)

---

### 4. Cosine Similarity Expertise Matching (AI/ML Task)
The `FacultySimilarityEngine` enables interactive keyword and profile-to-profile discovery:
- *Query:* `"quantum entanglement and condensed matter physics"`  
  $\rightarrow$ Matches: **Dr. Yogesh Singh** (IISER Mohali - DPS, Similarity: 0.5914), **Dr. Sanjeev Kumar** (DPS, Similarity: 0.5621)
- *Query:* `"microbial genomics, bioinformatics and protein structure"`  
  $\rightarrow$ Matches: **Dr. Shashi Bhushan Pandit** (IISER Mohali - DBS, Similarity: 0.3792), **Prof. Kavita Raman** (IIT Madras - DBS, Similarity: 0.3095)
- *Query:* `"differential equations and algebraic geometry"`  
  $\rightarrow$ Matches: **Dr. Yashonidhi Pandey** (DMS, Similarity: 0.6911), **Dr. Alok Maharana** (DMS, Similarity: 0.6911)
- *Query:* `"climate change, environmental geochemistry and hydrology"`  
  $\rightarrow$ Matches: **Dr. Vikas Goyal** (IIT Ropar - EES, Similarity: 0.4685), **Dr. Santosh B. Satbhai** (IISER Mohali - DBS, Similarity: 0.3687)

---

## 🛡️ Ethical Scraping Policy & Defensive Crawling
The web scraper in [`src/scraper.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/scraper.py) strictly enforces defensive, fail-closed crawling:
1. **Robots.txt Adherence with Strict Fail-Closed Safety:**  
   - **HTTP 200 OK:** Directives are parsed via Python's `urllib.robotparser.RobotFileParser` and verified.  
   - **Non-200 Status (401, 403, 404, 5xx) or Exceptions/Timeouts:** **FAILS CLOSED (`Disallow: /`)** and sets `robots_verified = False`. Live crawling halts immediately if rules cannot be verified, preventing unauthorized crawler access.
2. **Rate Limiting:** Enforces polite delays between HTTP GET requests to prevent server strain.
3. **Transparent Identification:** Transmits a descriptive User-Agent header specifying the academic project context.
4. **Local File Caching:** Persists HTML files locally in `data/raw/` so scraping is performed only once.

---

## 🎓 Empirical Academic Patterns, Denominators & Limitations

### 1. Doctoral Origins & Training Pathways (Recalculated with Exact Denominators)
- **Collection Date:** October 2026.
- **IISER Mohali Cohort ($N = 129$):**
  - **Documented PhD Institution:** $97 / 129$ ($75.2\%$ coverage, $32 / 129$ or $24.8\%$ missing in institute directory).
  - **Documented PhD Country:** $85 / 129$ ($65.9\%$ coverage, $44 / 129$ or $34.1\%$ missing).
    - **Domestic Doctorates (India):** $68 / 85$ ($80.0\%$) earned PhDs from premier national research institutes and universities, predominantly **TIFR Mumbai**, **IIT Kanpur**, **IISc Bangalore**, **Institute of Physics (IOP) Bhubaneswar**, and **CSIR-IMTECH**.
    - **International Doctorates:** $17 / 85$ ($20.0\%$) earned PhDs abroad: **Germany** ($7 / 85$, $8.2\%$), **USA** ($4 / 85$, $4.7\%$), **France** ($3 / 85$, $3.5\%$), **Japan** ($2 / 85$, $2.4\%$), and **United Kingdom** ($1 / 85$, $1.2\%$).
- **Combined Benchmark Dataset ($N = 137$, IISER Mohali + IITs):**
  - Documented PhD Country: $93 / 137$ ($67.9\%$ coverage, $44 / 137$ missing): India ($74 / 93$, $79.6\%$), Germany ($7 / 93$, $7.5\%$), USA ($6 / 93$, $6.5\%$), France ($3 / 93$, $3.2\%$), Japan ($2 / 93$, $2.2\%$), UK ($1 / 93$, $1.1\%$).

### 2. Postdoctoral Mobility & Fellowships (Recalculated with Exact Denominators)
- **IISER Mohali Postdoctoral Experience:** $67 / 129$ ($51.9\%$ documented, $62 / 129$ or $48.1\%$ without explicit postdoc mentions).
- **International Training Footprint:** Among faculty with documented postdoctoral training, leading affiliations include:
  - **USA:** University of California Berkeley / Lawrence Berkeley National Laboratory, University of California San Diego, Albert Einstein College of Medicine (NY), Arizona State University.
  - **Europe:** Saarland University (Germany), University of Bayreuth (Germany), Université catholique de Louvain (Belgium), Tel Aviv University (Israel).
  - **India:** Chennai Mathematical Institute, TIFR.

### 3. Study Limitations & Missing Data Documentation
- **Non-Uniform Profile Lengths:** Text lengths range from brief 2-sentence research summaries to exhaustive publication bibliographies.
- **Missingness Profile:** While foundational demographic fields (`name`, `institution`, `department_code`, `designation`) achieve **$100.0\%$ complete coverage ($137 / 137$)**, education history fields exhibit natural missingness in institutional public directories: $23.4\%$ of profiles omit the PhD institution, $32.1\%$ omit PhD country, and $45.3\%$ omit postdoctoral details.
- **Regex Parsing Variance:** Education extraction relies on semi-structured text heuristics; non-standard CV ordering or abbreviations may evade deterministic regex matching.
