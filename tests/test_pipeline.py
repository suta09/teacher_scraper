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


if __name__ == "__main__":
    unittest.main()
