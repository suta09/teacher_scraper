"""
test_pipeline.py
----------------
Unit & Integration Tests for IDC409 Faculty Profile Analysis.
Verifies SQLite CRUD, text preprocessing, feature reduction, model training,
clustering silhouette scores, and similarity search.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from src.database import FacultyDatabase
from src.preprocessor import clean_text, normalize_institution_name, prepare_profile_dataframe
from src.feature_engineering import FeaturePipeline
from src.models import DisciplineClassificationBenchmark
from src.clustering import ResearchClusterAnalyzer
from src.similarity_search import FacultySimilarityEngine


class TestFacultyPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_faculty.db")
        self.db = FacultyDatabase(self.db_path)

        # Sample synthetic faculty data for testing
        self.sample_profiles = [
            {
                "faculty_id": "prof_phy_1",
                "name": "Dr. Alice Quantum",
                "institution": "IISER Mohali",
                "department": "Department of Physical Sciences",
                "department_code": "DPS",
                "designation": "Professor",
                "email": "alice@iisermohali.ac.in",
                "research_area": "Quantum Optics and Superconductivity",
                "research_interests": "Investigating photon entanglement and topological phases of matter.",
                "phd_institute": "IISc Bangalore",
                "phd_country": "India",
                "phd_year": 2010
            },
            {
                "faculty_id": "prof_phy_2",
                "name": "Dr. Bob Lasers",
                "institution": "IISER Mohali",
                "department": "Department of Physical Sciences",
                "department_code": "DPS",
                "designation": "Associate Professor",
                "email": "bob@iisermohali.ac.in",
                "research_area": "Condensed Matter Physics and Nanophotonics",
                "research_interests": "Experimental ultrafast laser spectroscopy and semiconductors.",
                "phd_institute": "Stanford University",
                "phd_country": "USA",
                "phd_year": 2014
            },
            {
                "faculty_id": "prof_bio_1",
                "name": "Dr. Carol Genes",
                "institution": "IISER Mohali",
                "department": "Department of Biological Sciences",
                "department_code": "DBS",
                "designation": "Assistant Professor",
                "email": "carol@iisermohali.ac.in",
                "research_area": "Cellular Biology and Molecular Genetics",
                "research_interests": "CRISPR gene regulation and chromosome structural dynamics.",
                "keywords": "Biology; Genetics; Molecular; Cells",
                "phd_institute": "Max Planck Institute",
                "phd_country": "Germany",
                "phd_year": 2018
            },
            {
                "faculty_id": "prof_bio_2",
                "name": "Dr. David Cells",
                "institution": "IISER Mohali",
                "department": "Department of Biological Sciences",
                "department_code": "DBS",
                "designation": "Associate Professor",
                "email": "david@iisermohali.ac.in",
                "research_area": "Molecular Biology and Cellular Microbiology",
                "research_interests": "Bacterial biofilms and molecular genetics mechanisms.",
                "keywords": "Biology; Genetics; Molecular; Cells",
                "phd_institute": "IIT Kanpur",
                "phd_country": "India",
                "phd_year": 2012
            },
            {
                "faculty_id": "prof_chm_1",
                "name": "Dr. Eva Molecules",
                "institution": "IISER Mohali",
                "department": "Department of Chemical Sciences",
                "department_code": "DCS",
                "designation": "Professor",
                "email": "eva@iisermohali.ac.in",
                "research_area": "Synthetic Organic Chemistry and Catalysis",
                "research_interests": "Developing organocatalytic reactions and total synthesis of alkaloids.",
                "phd_institute": "IIT Bombay",
                "phd_country": "India",
                "phd_year": 2008
            },
            {
                "faculty_id": "prof_chm_2",
                "name": "Dr. Frank Bonds",
                "institution": "IISER Mohali",
                "department": "Department of Chemical Sciences",
                "department_code": "DCS",
                "designation": "Assistant Professor",
                "email": "frank@iisermohali.ac.in",
                "research_area": "Inorganic Materials and Electrochemistry",
                "research_interests": "Nanomaterials for lithium ion batteries and water splitting.",
                "phd_institute": "University of Cambridge",
                "phd_country": "UK",
                "phd_year": 2019
            }
        ]

    def tearDown(self):
        import gc
        self.db = None
        gc.collect()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_database_crud(self):
        """Verify Create, Read, Update, Delete in SQLite."""
        # Create / Bulk Write
        written = self.db.bulk_write_profiles(self.sample_profiles)
        self.assertEqual(written, len(self.sample_profiles))

        # Read
        prof = self.db.read_profile("prof_phy_1")
        self.assertIsNotNone(prof)
        self.assertEqual(prof["name"], "Dr. Alice Quantum")
        self.assertEqual(prof["department_code"], "DPS")

        # Update / Modify
        modified = self.db.modify_profile("prof_phy_1", {"designation": "Distinguished Professor"})
        self.assertTrue(modified)
        updated_prof = self.db.read_profile("prof_phy_1")
        self.assertEqual(updated_prof["designation"], "Distinguished Professor")

        # Delete
        deleted = self.db.delete_profile("prof_phy_1")
        self.assertTrue(deleted)
        self.assertIsNone(self.db.read_profile("prof_phy_1"))

        # DataFrame export
        df = self.db.to_dataframe()
        self.assertEqual(len(df), len(self.sample_profiles) - 1)

    def test_text_cleaning_and_preprocessor(self):
        """Test sanitization of raw academic text."""
        raw_text = "The aim of our group is to investigate Quantum Optics & Lasers at Dr. Lab!"
        cleaned = clean_text(raw_text)
        self.assertNotIn("group", cleaned)
        self.assertNotIn("aim", cleaned)
        self.assertIn("quantum", cleaned)
        self.assertIn("optics", cleaned)

        norm_inst = normalize_institution_name("Dept of Physics, IIT Kanpur, India")
        self.assertEqual(norm_inst, "IIT Kanpur")

    def test_feature_pipeline_and_reduction(self):
        """Test TF-IDF extraction and dimensionality reduction via TruncatedSVD."""
        texts = [p["research_area"] + " " + p["research_interests"] for p in self.sample_profiles]
        pipeline = FeaturePipeline(max_features=50, min_df=1, n_components=3)
        X_tfidf = pipeline.fit_transform(texts)

        self.assertEqual(X_tfidf.shape[0], len(texts))
        self.assertGreater(X_tfidf.shape[1], 5)

        X_reduced = pipeline.reduce_dimensions(X_tfidf)
        self.assertEqual(X_reduced.shape[0], len(texts))
        self.assertEqual(X_reduced.shape[1], 3)

        evr = pipeline.get_explained_variance()
        self.assertIn("cumulative_variance", evr)
        self.assertGreater(evr["cumulative_variance"], 0.0)

    def test_similarity_search(self):
        """Test cosine similarity retrieval for faculty and queries."""
        df = pd.DataFrame(self.sample_profiles)
        df = prepare_profile_dataframe(df)
        pipeline = FeaturePipeline(max_features=50, min_df=1, n_components=3)
        X_tfidf = pipeline.fit_transform(df['feature_text'].tolist())

        engine = FacultySimilarityEngine(df, X_tfidf, pipeline.vectorizer)
        
        # Query search
        results = engine.search_by_query("quantum entanglement superconductivity", top_n=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["name"], "Dr. Alice Quantum")

        # Profile-to-profile search
        similar_to_carol = engine.find_similar_by_faculty_id("prof_bio_1", top_n=2)
        self.assertGreater(len(similar_to_carol), 0)
        # Should match David Cells from Biology
        self.assertEqual(similar_to_carol[0]["department_code"], "DBS")

    def test_robots_fail_closed_on_error(self):
        """Test that EthicalFacultyScraper fails closed (Disallow: /) on non-200 status or network error."""
        from unittest.mock import patch, MagicMock
        from src.scraper import EthicalFacultyScraper

        # Simulate 500 Internal Server Error (fail closed)
        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"
            mock_get.return_value = mock_resp

            scraper = EthicalFacultyScraper(base_url="https://test.edu", cache_dir=self.test_dir)
            self.assertFalse(scraper.robots_verified)
            self.assertFalse(scraper.can_fetch("https://test.edu/faculty/profile"))

        # Simulate non-200 status (404/403/etc.) -> fail closed
        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_get.return_value = mock_resp

            scraper = EthicalFacultyScraper(base_url="https://test.edu", cache_dir=self.test_dir)
            self.assertFalse(scraper.robots_verified)
            self.assertFalse(scraper.can_fetch("https://test.edu/faculty/profile"))

        # Simulate 200 OK with explicit permission
        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "User-agent: *\nAllow: /\n"
            mock_get.return_value = mock_resp

            scraper = EthicalFacultyScraper(base_url="https://test.edu", cache_dir=self.test_dir)
            self.assertTrue(scraper.robots_verified)
            self.assertTrue(scraper.can_fetch("https://test.edu/faculty/profile"))

    def test_leakage_free_classification_pipeline(self):
        """Test multi-model benchmarking with leakage-free sklearn Pipelines."""
        texts = [
            "quantum optics physics superconductivity photon topological matter",
            "ultrafast laser spectroscopy condensed matter semiconductors physics",
            "astrophysics gravitational waves cosmology relativity astronomy physics",
            "statistical mechanics thermodynamics computational condensed matter physics",
            "cellular molecular genetics crispr gene regulation chromosome biology",
            "microbiology biochemistry cell division enzymes pathways protein biology",
            "immunology infectious diseases virology host pathogen interactions biology",
            "plant physiology genomics photosynthesis chloroplast biology",
            "algebraic topology differential geometry manifolds lie groups mathematics",
            "probability partial differential equations stochastic analysis complex mathematics",
            "number theory modular forms elliptic curves arithmetic geometry mathematics",
            "numerical analysis scientific computing mathematical modeling algorithms mathematics"
        ]
        labels = ["DPS", "DPS", "DPS", "DPS", "DBS", "DBS", "DBS", "DBS", "DMS", "DMS", "DMS", "DMS"]
        
        bench = DisciplineClassificationBenchmark(random_state=42)
        summary_df, detailed = bench.train_and_evaluate(texts, labels, test_size=0.25, reduce_dim=True, n_components=2)
        
        self.assertGreaterEqual(len(summary_df), 5)
        self.assertIn("Model", summary_df.columns)
        self.assertIn("Test Accuracy", summary_df.columns)
        self.assertIn("5-Fold CV Mean", summary_df.columns)
        
        # Verify pipelines were instantiated and fitted
        self.assertIn("Neural Network (MLP)", bench.fitted_pipelines)
        mlp_pipe = bench.fitted_pipelines["Neural Network (MLP)"]
        self.assertTrue(hasattr(mlp_pipe.named_steps["tfidf"], "vocabulary_"))
        self.assertTrue(hasattr(mlp_pipe.named_steps["svd"], "components_"))

    def test_cv_error_reporting_not_silent(self):
        """Verify that cross-validation exceptions do NOT silently substitute test accuracy."""
        from unittest.mock import patch
        texts = [
            "quantum optics physics", "laser physics photon", "astrophysics cosmology", "mechanics physics",
            "cellular genetics biology", "microbiology protein cell", "immunology disease biology", "plant cell biology"
        ]
        labels = ["DPS", "DPS", "DPS", "DPS", "DBS", "DBS", "DBS", "DBS"]

        bench = DisciplineClassificationBenchmark(random_state=42)
        with patch("src.models.cross_val_score", side_effect=RuntimeError("Simulated CV error")):
            summary_df, _ = bench.train_and_evaluate(texts, labels, test_size=0.25, reduce_dim=False)
            # Verify cv_mean is NaN and NOT equal to Test Accuracy
            for _, row in summary_df.iterrows():
                self.assertTrue(np.isnan(row["5-Fold CV Mean"]))

    def test_small_dataset_clustering_guard(self):
        """Verify that clustering guards against small datasets and empty k_range."""
        clusterer = ResearchClusterAnalyzer(random_state=42)
        
        # Test with tiny matrix (2 samples)
        small_X = np.array([[1.0, 0.0], [0.0, 1.0]])
        df_small = clusterer.find_optimal_k(small_X, k_range=range(3, 2))
        self.assertTrue(df_small.empty)

        # Test with identical duplicate vectors
        duplicate_X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        df_dup = clusterer.find_optimal_k(duplicate_X, k_range=range(2, 4))
        self.assertTrue(df_dup.empty)

    def test_readme_links_and_anchors(self):
        """Verify README.md contains no local file:/// links, all relative paths exist, and all TOC anchors resolve."""
        import re
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        readme_path = os.path.join(repo_root, "README.md")
        self.assertTrue(os.path.exists(readme_path), "README.md not found")

        with open(readme_path, "r", encoding="utf-8") as f:
            readme_text = f.read()

        # 1. Zero local file:/// links allowed
        self.assertNotIn("file:///", readme_text, "Found prohibited local file:/// links in README.md")

        # 2. Verify all markdown file links point to existing repository files/directories
        links = re.findall(r'\[([^\]]+)\]\(([^)#\s]+)\)', readme_text)
        for text, target in links:
            if target.startswith("http") or target.startswith("mailto"):
                continue
            full_path = os.path.normpath(os.path.join(repo_root, target))
            self.assertTrue(os.path.exists(full_path), f"Broken relative link in README: [{text}]({target})")

        # 3. Verify all TOC anchors resolve to defined HTML anchors
        defined_anchors = set(re.findall(r'<a id="([^"]+)"></a>', readme_text))
        toc_anchors = re.findall(r'\[([^\]]+)\]\(#([^)\s]+)\)', readme_text)
        self.assertGreater(len(toc_anchors), 0, "No TOC anchors found in README.md")
        for text, anchor in toc_anchors:
            self.assertIn(anchor, defined_anchors, f"Broken TOC anchor in README: [{text}](#{anchor})")

    def test_license_file_exists(self):
        """Verify that the MIT LICENSE file exists in the repository root."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        license_path = os.path.join(repo_root, "LICENSE")
        self.assertTrue(os.path.exists(license_path), "LICENSE file missing from repository root")
        with open(license_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("MIT License", content)

    def test_scraped_database_and_raw_cache_integrity(self):
        """Integration test: Verify the authentic repository SQLite database and raw HTML cache."""
        import sqlite3
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        db_path = os.path.join(repo_root, "data", "faculty.db")
        raw_dir = os.path.join(repo_root, "data", "raw")

        self.assertTrue(os.path.exists(db_path), f"Database not found: {db_path}")
        self.assertTrue(os.path.exists(raw_dir), f"Raw cache dir not found: {raw_dir}")

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT faculty_id, name, institution, department_code, profile_url FROM faculty_profiles")
        rows = c.fetchall()
        conn.close()

        # Cohort total size
        self.assertEqual(len(rows), 137, f"Expected 137 profiles in authentic database, found {len(rows)}")

        # Check raw HTML cache correspondence for IISER Mohali profiles
        all_raw_files = set(os.listdir(raw_dir))
        iiser_count = 0
        iit_count = 0
        for fid, name, inst, dept, url in rows:
            self.assertTrue(name and len(name) > 2, f"Invalid name for {fid}")
            self.assertTrue(url and url.startswith("https://"), f"Invalid HTTPS URL for {fid}")
            if inst == "IISER Mohali":
                iiser_count += 1
                stem = fid.replace("iiserm_", "faculty_")
                matches = [f for f in all_raw_files if f.startswith(stem)]
                self.assertTrue(len(matches) >= 1, f"Missing raw HTML cache file for {fid} ({name})")
            else:
                iit_count += 1

        self.assertEqual(iiser_count, 129, "Expected 129 IISER Mohali profiles")
        self.assertEqual(iit_count, 8, "Expected 8 Premier IIT benchmark profiles")


if __name__ == "__main__":
    unittest.main()
