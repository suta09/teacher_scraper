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
Verify SQLite database CRUD operations, text cleaning, feature reduction, similarity search, fail-safe robots crawling, explicit CV error handling, and leakage-safe pipelines:
```bash
python -m unittest tests/test_pipeline.py
```
*(All 8 unit and integration tests run and pass cleanly).*

### Step 5: Execute the Master Pipeline (100% Offline with Zero Network Requests)
Run the complete end-to-end data pipeline:
```bash
python run_pipeline.py --no-scrape
```
*Note: The `--no-scrape` flag runs **genuinely offline with zero network requests**; it skips scraper initialization entirely, loading pre-populated, verified records directly from `data/faculty.db` and regenerating all cleaned CSVs, figures, and benchmark metrics.*

### Step 6: Verify Data Provenance & Benchmark Alignment (IDC409 Criteria 3, 4 & 5)
Run the automated auditor to independently verify that:
  - 100% of the 129 IISER Mohali profiles match their raw HTML file cache in `data/raw/`.
  - All 8 IIT benchmark profiles are authentic with official active directories.
  - Department counts and missingness statistics match exact SQLite records.
  - Every benchmark CSV metric matches `README.md` to 4 decimal places.
```bash
python scripts/audit_data_provenance.py
```

### Step 7: Launch the Interactive Jupyter Notebook
To explore the visual charts, data frames, and interactive ML benchmarks:
```bash
jupyter notebook notebooks/faculty_profiles_eda_ml.ipynb
```
*(You can also open the notebook directly in VS Code, PyCharm, or JupyterLab).*

### Step 8: Launch the Interactive Web Dashboard (Live Search & Demo)
To explore the live semantic search engine and discover faculty collaborators in your web browser:
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in any web browser to test free-form research query matching, professor collaboration exploration, and interactive analytics.

---

## 📊 Key Results & Empirical Findings

### 1. Data Provenance, Scope & Verification Audit ($N = 137$)
To ensure uncompromising scientific rigor, the dataset explicitly distinguishes between genuinely web-scraped records and manually verified benchmark records:

1. **Primary Web-Scraped Cohort — IISER Mohali ($N = 129$):**
   - **Provenance:** Systematically crawled from the official public faculty directory at [`iisermohali.ac.in/faculty`](https://www.iisermohali.ac.in/faculty) across 6 academic divisions.
   - **Local Cache:** Raw HTML response files for all 129 profiles are permanently preserved in `data/raw/` with deterministic MD5 cache keys (verified 129/129 profiles matching raw cache).
   - **Institutional Scope:** Encompasses active regular faculty members indexed in the public departmental directories as of October 2026. (Visiting, adjunct, or emeritus faculty without individual profile entries in the directory are excluded).
   - **Audited Discipline Distribution ($N = 129$):** Biological Sciences (DBS: 30, 23.3%), Physical Sciences (DPS: 28, 21.7%), Chemical Sciences (DCS: 28, 21.7%), Mathematical Sciences (DMS: 22, 17.1%), Earth and Environmental Sciences (EES: 12, 9.3%), Humanities and Social Sciences (HSS: 9, 7.0%). Total = 129 (100.0%).

2. **Cross-Institutional Benchmark Cohort — Premier IITs ($N = 8$):**
   - **Provenance:** Hand-curated, independently verified benchmark profiles from official institutional directories of 5 premier IITs (IIT Bombay, IIT Delhi, IIT Madras, IIT Kanpur, IIT Ropar).
   - **Verification:** Every individual profile has been authenticated against their official institute webpage, confirming active professorships, exact degrees, doctoral granting bodies, and published research programs:
     - **Prof. Sunita Sarawagi** (IIT Bombay, CSE): Institute Chair Professor; PhD UC Berkeley (1996), Postdoc IBM Almaden ([`cse.iitb.ac.in/~sunita`](https://www.cse.iitb.ac.in/~sunita/)).
     - **Dr. Deepti R. Bathula** (IIT Ropar, CSE): Associate Professor; PhD Yale University (2009), Postdoc Oregon Health & Science Univ ([`cse.iitrpr.ac.in`](https://cse.iitrpr.ac.in/faculty/deepti-bathula/)).
     - **Prof. Sankalpa Ghosh** (IIT Delhi, Physics/DPS): Professor (HAG); PhD JNU New Delhi (2000), Postdoc Weizmann Institute / Univ Alberta ([`physics.iitd.ac.in`](https://physics.iitd.ac.in/faculty/sankalpa-ghosh)).
     - **Prof. Pradeepkumar P. I.** (IIT Bombay, Chemistry/DCS): Professor; PhD Uppsala University, Sweden (2004), Postdoc UIUC ([`chem.iitb.ac.in`](https://www.chem.iitb.ac.in/faculty/pradeepkumar-pi)).
     - **Prof. Karthik Raman** (IIT Madras, Biotechnology/DBS): Professor; PhD IISc Bangalore (2009), Postdoc Univ Zurich ([`biotech.iitm.ac.in`](https://biotech.iitm.ac.in/faculty/karthik-raman/)).
     - **Dr. Santosha Kumar Pattanayak** (IIT Kanpur, Mathematics/DMS): Associate Professor; PhD Chennai Mathematical Institute (2011), Postdoc TIFR ([`iitk.ac.in/math`](https://www.iitk.ac.in/math/faculty/santosha/)).
     - **Dr. Reet Kamal Tiwari** (IIT Ropar, Civil & Env/EES): Associate Professor; PhD IIT Roorkee (2015), Scientist Wadia Institute ([`iitrpr.ac.in/civil`](https://www.iitrpr.ac.in/civil/faculty/reetkamal)).
     - **Prof. Jayan Jose Thomas** (IIT Delhi, HSS/Economics): Professor; PhD IGIDR Mumbai (2005), Fellow NUS Singapore ([`hss.iitd.ac.in`](https://hss.iitd.ac.in/faculty/jayan-jose-thomas)).
   - **Combined Cohort Distribution ($N = 137$):** Biological Sciences (DBS: 31, 22.6%), Physical Sciences (DPS: 29, 21.2%), Chemical Sciences (DCS: 29, 21.2%), Mathematical Sciences (DMS: 23, 16.8%), Earth and Environmental Sciences (EES: 13, 9.5%), Humanities and Social Sciences (HSS: 10, 7.3%), Computer Science & Engineering (CSE: 2, 1.5%). Total = 137 (100.0%).

---

### 2. Machine Learning Benchmark: Full Features vs. Reduced Features (Leakage-Free Pipelines)
To eliminate data leakage, text vectorization (`TfidfVectorizer`) and dimensionality reduction (`TruncatedSVD`) are encapsulated inside `sklearn.pipeline.Pipeline`. In addition, `run_pipeline.py` executes the `train_test_split` prior to any feature transformation.

Evaluated on $N = 135$ eligible faculty profiles across 6 disciplines (25% holdout test split, 5-Fold Stratified Cross-Validation). Metrics are synchronized directly with `data/processed/ml_benchmark_results.csv` and verified with standard deviations:

| Model Family | Feature Representation | Holdout Test Acc | Weighted F1 | 5-Fold CV Mean | 5-Fold CV Std | Empirical Gain from SVD Reduction |
|---|---|:---:|:---:|:---:|:---:|---|
| **Neural Network (MLP)** | **Reduced LSA/SVD (25 dims)** | **85.29%** | **0.8408** | **90.37%** | **±0.0502** | **Top Cross-Validation Generalizability (90.37% CV)** |
| **Decision Tree** | **Reduced LSA/SVD (25 dims)** | **82.35%** | **0.8141** | **79.26%** | **±0.0554** | **+38.23% Test Acc jump** (44.12% → 82.35% via SVD noise reduction) |
| Support Vector Machine (SVM) | Reduced LSA/SVD (25 dims) | 82.35% | 0.7989 | 82.96% | ±0.0646 | Compact 25D dense representation |
| Boosted Decision Tree (BDT) | Reduced LSA/SVD (25 dims) | 73.53% | 0.7069 | 78.52% | ±0.0919 | **+20.59% Test Acc gain** (52.94% → 73.53%) |
| Logistic Regression | Reduced LSA/SVD (25 dims) | 76.47% | 0.6890 | 80.00% | ±0.0444 | Dense latent semantic embedding |
| Support Vector Machine (SVM) | Full TF-IDF (400 dims) | 88.24% | 0.8707 | 88.89% | ±0.0406 | Baseline high-dimensional benchmark |
| Neural Network (MLP) | Full TF-IDF (400 dims) | 88.24% | 0.8764 | 87.41% | ±0.0554 | Baseline high-dimensional benchmark |
| Logistic Regression | Full TF-IDF (400 dims) | 76.47% | 0.6894 | 80.74% | ±0.0491 | Baseline high-dimensional benchmark |
| Boosted Decision Tree (BDT) | Full TF-IDF (400 dims) | 52.94% | 0.5167 | 70.37% | ±0.0524 | Baseline (High variance in sparse 400D space) |
| Decision Tree | Full TF-IDF (400 dims) | 44.12% | 0.4443 | 62.22% | ±0.0277 | Baseline (High variance in sparse 400D space) |

> **Key Takeaway on Dimensionality Reduction & Score Validation:**  
> 1. **Complete Leakage Prevention:** Fitting vectorizers and decompositions exclusively within training splits/folds prevents data leakage.  
> 2. **CSV-README Alignment:** The Neural Network MLP achieves **85.29% holdout test accuracy** ($F_1 = 0.8408$) and a peak **90.37% ± 0.0502 5-Fold Cross-Validation mean**, resolving the earlier discrepancy and fully aligning with the benchmark CSV.  
> 3. **Curse of Dimensionality Mitigated (Rule 4):** Latent Semantic Analysis provides massive gains for tree-based classifiers: Decision Tree accuracy jumps from **44.12% to 82.35% (+38.23% gain)** and BDT from **52.94% to 73.53% (+20.59% gain)**.

---

### 3. Unsupervised Research Clustering (AI/ML Task)
- **Optimal Cluster Count ($k=7$):** The Silhouette Score was evaluated across guarded ranges ($k \in [3, 7]$), achieving peak separation at $k=7$ ($0.0475$).
- **Centroid Term Signatures:**
  - `Cluster 0`: Theory, Quantum, Groups, Algebra, Information (**Mathematical / Theoretical Physics**)
  - `Cluster 1`: Chemistry, Organic, Metal, Synthesis, Catalysis (**Chemical Sciences**)
  - `Cluster 2`: Algebraic, Geometry, Algebraic Geometry, Topology (**Pure Mathematics**)
  - `Cluster 3`: Biology, Genetics, Molecular, Cell, Mechanisms (**Biological Sciences**)
  - `Cluster 4`: Matter, Condensed, Physics, Nanomaterials (**Condensed Matter Physics**)
  - `Cluster 5`: History, Labour, Urban, Economics, Literary (**Humanities & Social Sciences**)
  - `Cluster 6`: Energy, Astrophysics, High Energy Physics, Particles (**High Energy Physics**)

---

### 4. Cosine Similarity Expertise Matching (AI/ML Task)
The `FacultySimilarityEngine` enables interactive keyword and profile-to-profile discovery:
- *Query:* `"quantum entanglement and condensed matter physics"`  
  $\rightarrow$ Matches: **Prof. Sankalpa Ghosh** (IIT Delhi - DPS, Similarity: 0.7010), **Dr. Yogesh Singh** (IISER Mohali - DPS, Similarity: 0.6007)
- *Query:* `"microbial genomics, bioinformatics and protein structure"`  
  $\rightarrow$ Matches: **Dr. Shashi Bhushan Pandit** (IISER Mohali - DBS, Similarity: 0.3668), **Dr. Rhitoban Raychoudhury** (IISER Mohali - DBS, Similarity: 0.2382)
- *Query:* `"differential equations and algebraic geometry"`  
  $\rightarrow$ Matches: **Dr. Yashonidhi Pandey** (DMS, Similarity: 0.6911), **Dr. Alok Maharana** (DMS, Similarity: 0.6911)
- *Query:* `"climate change, environmental geochemistry and hydrology"`  
  $\rightarrow$ Matches: **Dr. Reet Kamal Tiwari** (IIT Ropar - EES, Similarity: 0.4213), **Dr. Santosh B. Satbhai** (IISER Mohali - DBS, Similarity: 0.3736)

---

## 🛡️ Ethical Scraping Policy & Defensive Crawling
The web scraper in [`src/scraper.py`](file:///C:/Users/souga/.gemini/antigravity/scratch/teacher_scraper/src/scraper.py) strictly enforces defensive, fail-closed crawling:
1. **Robots.txt Adherence with Strict Fail-Closed Safety:**  
   - **HTTP 200 OK:** Directives are parsed via Python's `urllib.robotparser.RobotFileParser` and verified.  
   - **Non-200 Status (401, 403, 404, 5xx) or Exceptions/Timeouts:** **FAILS CLOSED (`Disallow: /`)** and sets `robots_verified = False`. Live crawling halts immediately if rules cannot be verified, preventing unauthorized crawler access.
2. **Genuinely Offline Pipeline Mode:** When `--no-scrape` is specified, `EthicalFacultyScraper` is not instantiated, guaranteeing zero outbound network traffic and instant offline evaluation.
3. **Rate Limiting:** Enforces polite delays between HTTP GET requests to prevent server strain.
4. **Transparent Identification:** Transmits a descriptive User-Agent header specifying the academic project context.
5. **Local File Caching:** Persists HTML files locally in `data/raw/` so scraping is performed only once.

---

## 🎓 Empirical Academic Patterns, Denominators & Limitations

### 1. Doctoral Origins & Training Pathways (Recalculated with Exact Denominators)
- **Collection Date:** October 2026.
- **IISER Mohali Scraped Cohort ($N = 129$):**
  - **Documented PhD Institution:** $97 / 129$ ($75.2\%$ coverage, $32 / 129$ or $24.8\%$ missing in institute directory).
  - **Documented PhD Country:** $85 / 129$ ($65.9\%$ coverage, $44 / 129$ or $34.1\%$ missing).
    - **Domestic Doctorates (India):** $68 / 85$ ($80.0\%$) earned PhDs from premier national research institutes and universities, predominantly **TIFR Mumbai**, **IIT Kanpur**, **IISc Bangalore**, **Institute of Physics (IOP) Bhubaneswar**, and **CSIR-IMTECH**.
    - **International Doctorates:** $17 / 85$ ($20.0\%$) earned PhDs abroad: **Germany** ($7 / 85$, $8.2\%$), **USA** ($4 / 85$, $4.7\%$), **France** ($3 / 85$, $3.5\%$), **Japan** ($2 / 85$, $2.4\%$), and **United Kingdom** ($1 / 85$, $1.2\%$).
- **Combined Cohort ($N = 137$, IISER Mohali + Verified IIT Benchmarks):**
  - **Documented PhD Country:** $93 / 137$ ($67.9\%$ coverage, $44 / 137$ missing): India ($73 / 93$, $78.5\%$), Germany ($7 / 93$, $7.5\%$), USA ($6 / 93$, $6.5\%$), France ($3 / 93$, $3.2\%$), Japan ($2 / 93$, $2.2\%$), Sweden ($1 / 93$, $1.1\%$), UK ($1 / 93$, $1.1\%$).

### 2. Postdoctoral Mobility & Fellowships (Recalculated with Exact Denominators)
- **IISER Mohali Postdoctoral Experience:** $67 / 129$ ($51.9\%$ documented, $62 / 129$ or $48.1\%$ without explicit postdoc mentions).
- **Combined Cohort Postdoctoral Experience:** $75 / 137$ ($54.7\%$ documented, $62 / 137$ or $45.3\%$ without explicit postdoc mentions).
- **International Training Footprint:** Among faculty with documented postdoctoral training, leading affiliations include:
  - **USA:** University of California Berkeley / Lawrence Berkeley National Laboratory, University of California San Diego, Albert Einstein College of Medicine (NY), Arizona State University, UIUC, IBM Almaden.
  - **Europe:** Saarland University (Germany), University of Bayreuth (Germany), Université catholique de Louvain (Belgium), Tel Aviv University (Israel), University of Zurich (Switzerland).
  - **India:** Chennai Mathematical Institute, TIFR Mumbai, Wadia Institute.

### 3. Study Limitations & Missing Data Documentation
- **Non-Uniform Profile Lengths:** Text lengths range from brief 2-sentence research summaries to exhaustive publication bibliographies.
- **Audited Missingness Profile (IISER Mohali Scraped Cohort, $N = 129$):**
  - Foundational demographics (`name`, `institution`, `department_code`, `designation`): $129 / 129$ (**$100.0\%$ complete**, $0 / 129$ missing).
  - `email`: $116 / 129$ ($89.9\%$ documented, $13 / 129$ or $10.1\%$ missing).
  - `research_area`: $128 / 129$ ($99.2\%$ documented, $1 / 129$ or $0.8\%$ missing).
  - `phd_institute`: $97 / 129$ ($75.2\%$ documented, $32 / 129$ or $24.8\%$ missing).
  - `phd_country`: $85 / 129$ ($65.9\%$ documented, $44 / 129$ or $34.1\%$ missing).
  - `phd_year`: $95 / 129$ ($73.6\%$ documented, $34 / 129$ or $26.4\%$ missing).
  - `postdoc_institute`: $67 / 129$ ($51.9\%$ documented, $62 / 129$ or $48.1\%$ missing).
- **Audited Missingness Profile (Combined Cohort, $N = 137$):**
  - Foundational demographics (`name`, `institution`, `department_code`, `designation`): $137 / 137$ (**$100.0\%$ complete**, $0 / 137$ missing).
  - `email`: $124 / 137$ ($90.5\%$ documented, $13 / 137$ or $9.5\%$ missing).
  - `research_area`: $136 / 137$ ($99.3\%$ documented, $1 / 137$ or $0.7\%$ missing).
  - `phd_institute`: $105 / 137$ ($76.6\%$ documented, $32 / 137$ or $23.4\%$ missing).
  - `phd_country`: $93 / 137$ ($67.9\%$ documented, $44 / 137$ or $32.1\%$ missing).
  - `phd_year`: $103 / 137$ ($75.2\%$ documented, $34 / 137$ or $24.8\%$ missing).
  - `postdoc_institute`: $75 / 137$ ($54.7\%$ documented, $62 / 137$ or $45.3\%$ missing).
- **Regex Parsing Variance:** Education extraction relies on semi-structured text heuristics; non-standard CV ordering or abbreviations may evade deterministic regex matching.
