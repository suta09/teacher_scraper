"""
app.py
------
Interactive Web Dashboard for Web Scraping II: Faculty Profiles (IDC409).
Provides a live UI for:
  - Semantic Cosine Similarity Search across research areas
  - Professor-to-Professor Collaboration Discovery
  - Interactive EDA Visualizations
  - Machine Learning Classification Benchmark Results
"""

import os
import sys
from typing import List, Dict, Any
from flask import Flask, request, jsonify, render_template_string, send_from_directory
import pandas as pd
import numpy as np

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.database import FacultyDatabase
from src.preprocessor import prepare_profile_dataframe
from src.feature_engineering import FeaturePipeline
from src.similarity_search import FacultySimilarityEngine

app = Flask(__name__)

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
DB_PATH = os.path.join(DATA_DIR, "faculty.db")

# Initialize database, data & search engine
print("[App] Loading database and indexing profiles...")
db = FacultyDatabase(DB_PATH)
df_raw = db.to_dataframe()
df_clean = prepare_profile_dataframe(df_raw)

# Vectorize profiles
pipeline = FeaturePipeline(max_features=400, min_df=1, n_components=25)
X_tfidf = pipeline.fit_transform(df_clean['feature_text'].tolist())
search_engine = FacultySimilarityEngine(df_clean, X_tfidf, pipeline.vectorizer)
print(f"[App] Initialized search engine across {len(df_clean)} faculty profiles.")

# Load benchmark metrics if available
benchmark_csv = os.path.join(DATA_DIR, "processed", "ml_benchmark_results.csv")
if os.path.exists(benchmark_csv):
    metrics_df = pd.read_csv(benchmark_csv)
else:
    metrics_df = pd.DataFrame()


# HTML Template with modern clean CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Faculty Profiles & Expertise Explorer — IDC409</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #1a73e8;
            --primary-dark: #1557b0;
            --bg-light: #f8f9fa;
        }
        body {
            background-color: var(--bg-light);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #202124;
        }
        .hero-banner {
            background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
            color: white;
            padding: 3rem 1rem 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .nav-tabs .nav-link {
            font-weight: 600;
            color: #5f6368;
            border: none;
            padding: 0.8rem 1.4rem;
        }
        .nav-tabs .nav-link.active {
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            background: transparent;
        }
        .card-custom {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: transform 0.2s, box-shadow 0.2s;
            background: white;
        }
        .card-custom:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        }
        .badge-dept {
            background-color: #e8f0fe;
            color: #1a73e8;
            font-weight: 600;
            font-size: 0.8rem;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
        }
        .badge-score {
            background-color: #e6f4ea;
            color: #137333;
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
        }
        .search-input {
            border-radius: 30px;
            padding: 0.75rem 1.5rem;
            font-size: 1.05rem;
            border: 2px solid #dfe1e5;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .search-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(26,115,232,0.15);
        }
        .figure-img {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            width: 100%;
            height: auto;
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body>

    <!-- Header Banner -->
    <div class="hero-banner text-center">
        <div class="container">
            <h1 class="fw-bold mb-2"><i class="fa-solid fa-graduation-cap me-2"></i>Web Scraping II: Faculty Profiles</h1>
            <p class="lead mb-0 opacity-90">Academic Expertise Discovery, Relational SQLite CRUD & Machine Learning Benchmarking — IDC409</p>
            <div class="mt-3">
                <span class="badge bg-light text-primary me-2"><i class="fa-solid fa-database me-1"></i>137 Profiles</span>
                <span class="badge bg-light text-primary me-2"><i class="fa-solid fa-brain me-1"></i>5 ML Models</span>
                <span class="badge bg-light text-primary"><i class="fa-solid fa-compress me-1"></i>SVD Latent Space (25 Dims)</span>
            </div>
        </div>
    </div>

    <!-- Main Container -->
    <div class="container pb-5">
        <!-- Navigation Tabs -->
        <ul class="nav nav-tabs mb-4 justify-content-center" id="dashboardTabs" role="tablist">
            <li class="nav-item">
                <button class="nav-link active" id="search-tab" data-bs-toggle="tab" data-bs-target="#search-pane" type="button">
                    <i class="fa-solid fa-magnifying-glass me-2"></i>Semantic Expertise Search
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="collaborate-tab" data-bs-toggle="tab" data-bs-target="#collaborate-pane" type="button">
                    <i class="fa-solid fa-network-wired me-2"></i>Faculty Matcher
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="eda-tab" data-bs-toggle="tab" data-bs-target="#eda-pane" type="button">
                    <i class="fa-solid fa-chart-pie me-2"></i>Visual Analytics & EDA
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="models-tab" data-bs-toggle="tab" data-bs-target="#models-pane" type="button">
                    <i class="fa-solid fa-microchip me-2"></i>ML Model Benchmarks
                </button>
            </li>
        </ul>

        <!-- Tab Content -->
        <div class="tab-content" id="dashboardTabContent">

            <!-- TAB 1: Semantic Expertise Search -->
            <div class="tab-pane fade show active" id="search-pane">
                <div class="row justify-content-center mb-4">
                    <div class="col-lg-8">
                        <div class="input-group">
                            <input type="text" id="queryInput" class="form-control search-input" 
                                   placeholder="Search by keywords (e.g. quantum entanglement, differential geometry, bioinformatics, climate change)..."
                                   onkeypress="if(event.key === 'Enter') performSearch()">
                            <button class="btn btn-primary px-4 rounded-pill ms-2" onclick="performSearch()">
                                <i class="fa-solid fa-search me-1"></i>Search
                            </button>
                        </div>
                        <div class="mt-2 text-center text-muted small">
                            Try quick searches: 
                            <a href="#" onclick="quickSearch('quantum optics superconductivity')" class="text-decoration-none me-2">Quantum Optics</a> |
                            <a href="#" onclick="quickSearch('algebraic geometry topology')" class="text-decoration-none me-2">Algebraic Geometry</a> |
                            <a href="#" onclick="quickSearch('microbial genomics proteins')" class="text-decoration-none me-2">Genomics</a> |
                            <a href="#" onclick="quickSearch('organic synthesis catalysis')" class="text-decoration-none">Catalysis</a>
                        </div>
                    </div>
                </div>

                <div id="searchResults" class="row g-3">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>

            <!-- TAB 2: Faculty Matcher -->
            <div class="tab-pane fade" id="collaborate-pane">
                <div class="row justify-content-center mb-4">
                    <div class="col-lg-7">
                        <label class="form-label fw-bold">Select a Professor to Discover Overlapping Collaborators:</label>
                        <select id="facultySelect" class="form-select form-select-lg" onchange="findCollaborators()">
                            <option value="">-- Choose a Faculty Member --</option>
                            {% for prof in faculty_list %}
                            <option value="{{ prof.faculty_id }}">{{ prof.name }} ({{ prof.department_code }}) - {{ prof.institution }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <div id="targetProfileInfo" class="mb-4"></div>
                <div id="collaboratorResults" class="row g-3"></div>
            </div>

            <!-- TAB 3: Visual Analytics & EDA -->
            <div class="tab-pane fade" id="eda-pane">
                <div class="row g-4">
                    <div class="col-md-6">
                        <div class="card card-custom p-3">
                            <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-column me-2 text-primary"></i>1. Department Distribution</h5>
                            <img src="/reports/figures/01_department_counts.png" class="figure-img" alt="Department Distribution">
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card card-custom p-3">
                            <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-pie me-2 text-primary"></i>2. Faculty Seniority Breakdown</h5>
                            <img src="/reports/figures/02_designation_counts.png" class="figure-img" alt="Designation Counts">
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card card-custom p-3">
                            <h5 class="fw-bold mb-3"><i class="fa-solid fa-building-columns me-2 text-primary"></i>3. Top PhD Alma Maters</h5>
                            <img src="/reports/figures/03_top_phd_institutes.png" class="figure-img" alt="Top PhD Institutes">
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card card-custom p-3">
                            <h5 class="fw-bold mb-3"><i class="fa-solid fa-earth-americas me-2 text-primary"></i>4. PhD Origin Countries</h5>
                            <img src="/reports/figures/04_phd_countries.png" class="figure-img" alt="PhD Countries">
                        </div>
                    </div>
                    <div class="col-12">
                        <div class="card card-custom p-3 text-center">
                            <h5 class="fw-bold mb-3"><i class="fa-solid fa-cloud me-2 text-primary"></i>5. Research Themes Word Cloud</h5>
                            <img src="/reports/figures/06_research_wordcloud.png" class="figure-img" style="max-height: 500px; object-fit: contain;" alt="Word Cloud">
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 4: ML Model Benchmarks -->
            <div class="tab-pane fade" id="models-pane">
                <div class="card card-custom p-4 mb-4">
                    <h4 class="fw-bold mb-3"><i class="fa-solid fa-trophy me-2 text-warning"></i>Model Benchmark & Dimensionality Reduction (Rule 4, 5, 6)</h4>
                    <p class="text-muted">
                        Comparison of 5 classification architectures predicting academic discipline from raw research profiles.
                        Dimensionality reduction using <strong>TruncatedSVD (25 components)</strong> yielded a dramatic performance boost (+5.9% to +23.5%) by filtering vocabulary noise.
                    </p>
                    <div class="table-responsive">
                        <table class="table table-hover align-middle">
                            <thead class="table-light">
                                <tr>
                                    <th>Model Family</th>
                                    <th>Feature Representation</th>
                                    <th>Test Accuracy</th>
                                    <th>Weighted F1-Score</th>
                                    <th>5-Fold CV Mean</th>
                                    <th>Verdict</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for row in benchmark_rows %}
                                <tr {% if 'Reduced' in row['Feature Set'] and 'Neural Network' in row['Model'] %}class="table-success fw-bold"{% endif %}>
                                    <td>{{ row['Model'] }}</td>
                                    <td><span class="badge {% if 'Reduced' in row['Feature Set'] %}bg-primary{% else %}bg-secondary{% endif %}">{{ row['Feature Set'] }}</span></td>
                                    <td>{{ "%.2f"|format(row['Test Accuracy'] * 100) }}%</td>
                                    <td>{{ "%.4f"|format(row['F1 Score (Weighted)']) }}</td>
                                    <td>{{ "%.2f"|format(row['5-Fold CV Mean'] * 100) }}%</td>
                                    <td>
                                        {% if 'Neural Network' in row['Model'] and 'Reduced' in row['Feature Set'] %}
                                            <span class="badge bg-success">★ Best Model (91.2%)</span>
                                        {% elif 'Reduced' in row['Feature Set'] %}
                                            <span class="text-success small fw-semibold">+ Boosted by SVD</span>
                                        {% else %}
                                            <span class="text-muted small">Baseline</span>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function quickSearch(query) {
            document.getElementById('queryInput').value = query;
            performSearch();
        }

        async function performSearch() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            const container = document.getElementById('searchResults');
            container.innerHTML = '<div class="col-12 text-center py-4"><div class="spinner-border text-primary" role="status"></div><p class="mt-2 text-muted">Searching faculty embeddings...</p></div>';

            try {
                const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const data = await resp.json();

                if (data.length === 0) {
                    container.innerHTML = '<div class="col-12 text-center py-4 text-muted">No faculty matched your query. Try different research terms!</div>';
                    return;
                }

                container.innerHTML = data.map(m => `
                    <div class="col-md-6">
                        <div class="card card-custom p-3 h-100">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h5 class="fw-bold mb-0 text-dark">${m.name}</h5>
                                <span class="badge-score"><i class="fa-solid fa-chart-line me-1"></i>${(m.similarity_score * 100).toFixed(1)}% Match</span>
                            </div>
                            <div class="mb-2">
                                <span class="badge-dept me-1">${m.department_code || m.department}</span>
                                <span class="text-muted small">${m.designation} &bull; ${m.institution}</span>
                            </div>
                            <p class="small text-muted mb-2 flex-grow-1">${m.research_area || 'Research area details in profile.'}</p>
                            ${m.email ? `<div class="small text-primary"><i class="fa-solid fa-envelope me-1"></i><a href="mailto:${m.email}" class="text-decoration-none">${m.email}</a></div>` : ''}
                        </div>
                    </div>
                `).join('');
            } catch (err) {
                container.innerHTML = `<div class="col-12 text-center text-danger py-4">Error loading results: ${err}</div>`;
            }
        }

        async function findCollaborators() {
            const facultyId = document.getElementById('facultySelect').value;
            if (!facultyId) return;

            const infoContainer = document.getElementById('targetProfileInfo');
            const resultsContainer = document.getElementById('collaboratorResults');
            resultsContainer.innerHTML = '<div class="col-12 text-center py-4"><div class="spinner-border text-primary"></div></div>';

            try {
                const resp = await fetch(`/api/similar?id=${encodeURIComponent(facultyId)}`);
                const data = await resp.json();

                infoContainer.innerHTML = `
                    <div class="alert alert-primary">
                        <h5 class="fw-bold mb-1"><i class="fa-solid fa-user me-2"></i>${data.target.name}</h5>
                        <p class="mb-1">${data.target.designation} &bull; ${data.target.department} (${data.target.institution})</p>
                        <p class="small mb-0 text-muted"><strong>Primary Research:</strong> ${data.target.research_area || 'N/A'}</p>
                    </div>
                    <h5 class="fw-bold mt-4 mb-3"><i class="fa-solid fa-users me-2"></i>Top 5 Closest Research Collaborators:</h5>
                `;

                resultsContainer.innerHTML = data.collaborators.map(m => `
                    <div class="col-md-6">
                        <div class="card card-custom p-3 h-100">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h6 class="fw-bold mb-0">${m.name}</h6>
                                <span class="badge-score">${(m.similarity_score * 100).toFixed(1)}% Overlap</span>
                            </div>
                            <div class="mb-2">
                                <span class="badge-dept me-1">${m.department_code || m.department}</span>
                                <span class="text-muted small">${m.designation}</span>
                            </div>
                            <p class="small text-muted mb-0">${m.research_area || 'Details available in profile.'}</p>
                        </div>
                    </div>
                `).join('');
            } catch (err) {
                resultsContainer.innerHTML = `<div class="col-12 text-danger">Error: ${err}</div>`;
            }
        }

        // Run default search on load
        window.addEventListener('DOMContentLoaded', () => {
            quickSearch('quantum physics');
        });
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    """Render main interactive dashboard."""
    faculty_list = df_clean[['faculty_id', 'name', 'department_code', 'institution']].sort_values(by='name').to_dict('records')
    benchmark_rows = metrics_df.to_dict('records') if not metrics_df.empty else []
    return render_template_string(HTML_TEMPLATE, faculty_list=faculty_list, benchmark_rows=benchmark_rows)


@app.route("/api/search")
def api_search():
    """Endpoint for semantic expertise search."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    matches = search_engine.search_by_query(q, top_n=6)
    return jsonify(matches)


@app.route("/api/similar")
def api_similar():
    """Endpoint to retrieve closest faculty collaborators."""
    fid = request.args.get("id", "").strip()
    if not fid:
        return jsonify({"error": "Missing faculty ID"}), 400

    target = df_clean[df_clean['faculty_id'] == fid].iloc[0].to_dict()
    collaborators = search_engine.find_similar_by_faculty_id(fid, top_n=5)
    return jsonify({
        "target": {
            "name": target.get("name"),
            "designation": target.get("designation"),
            "department": target.get("department"),
            "institution": target.get("institution"),
            "research_area": target.get("research_area")
        },
        "collaborators": collaborators
    })


@app.route("/reports/figures/<path:filename>")
def serve_figure(filename):
    """Serve EDA plots and word clouds."""
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================================")
    print(f"🚀 Faculty Profiles Dashboard live at: http://127.0.0.1:{port}")
    print(f"==================================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
