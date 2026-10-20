"""
database.py
-----------
SQLite Database Manager for Academic Faculty Profiles.
Fulfills IDC409 Requirement 1 & 2:
  - Use NumPy and Pandas to store data
  - Database usage: Read, Write (Insert), and Modify (Update/Delete) operations.
"""

import sqlite3
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np


class FacultyDatabase:
    """SQLite Database Interface providing full CRUD operations for faculty profiles."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path == ':memory:':
            self.db_path = ':memory:'
        elif db_path is None:
            # Default to data/faculty.db relative to this file's directory
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "faculty.db")
        else:
            self.db_path = os.path.abspath(db_path)
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Establish a SQLite database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self) -> None:
        """Initialize database schema with primary profiles table and indexing."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Faculty profiles main table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faculty_profiles (
                    faculty_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    institution TEXT NOT NULL,
                    department TEXT NOT NULL,
                    department_code TEXT,
                    designation TEXT,
                    email TEXT,
                    office TEXT,
                    research_area TEXT,
                    research_interests TEXT,
                    education_raw TEXT,
                    phd_institute TEXT,
                    phd_country TEXT,
                    phd_year INTEGER,
                    postdoc_institute TEXT,
                    keywords TEXT,
                    profile_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Scraping metadata & audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scraping_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    status_code INTEGER,
                    error_message TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indices for rapid querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_inst ON faculty_profiles(institution)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_dept ON faculty_profiles(department_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_desig ON faculty_profiles(designation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_phd_inst ON faculty_profiles(phd_institute)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faculty_phd_cntry ON faculty_profiles(phd_country)")

            conn.commit()

    # =========================================================================
    # CREATE / WRITE OPERATIONS
    # =========================================================================
    def write_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Insert or replace a faculty profile record.
        
        Args:
            profile: Dictionary containing faculty profile fields.
        Returns:
            bool: True if write succeeded.
        """
        now = datetime.now(timezone.utc).isoformat()
        fields = [
            'faculty_id', 'name', 'institution', 'department', 'department_code',
            'designation', 'email', 'office', 'research_area', 'research_interests',
            'education_raw', 'phd_institute', 'phd_country', 'phd_year',
            'postdoc_institute', 'keywords', 'profile_url'
        ]
        
        record = {k: profile.get(k, None) for k in fields}
        record['created_at'] = profile.get('created_at', now)
        record['updated_at'] = now

        cols = list(record.keys())
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        
        query = f"""
            INSERT INTO faculty_profiles ({col_names})
            VALUES ({placeholders})
            ON CONFLICT(faculty_id) DO UPDATE SET
                name=excluded.name,
                institution=excluded.institution,
                department=excluded.department,
                department_code=excluded.department_code,
                designation=excluded.designation,
                email=COALESCE(excluded.email, faculty_profiles.email),
                office=COALESCE(excluded.office, faculty_profiles.office),
                research_area=COALESCE(excluded.research_area, faculty_profiles.research_area),
                research_interests=COALESCE(excluded.research_interests, faculty_profiles.research_interests),
                education_raw=COALESCE(excluded.education_raw, faculty_profiles.education_raw),
                phd_institute=COALESCE(excluded.phd_institute, faculty_profiles.phd_institute),
                phd_country=COALESCE(excluded.phd_country, faculty_profiles.phd_country),
                phd_year=COALESCE(excluded.phd_year, faculty_profiles.phd_year),
                postdoc_institute=COALESCE(excluded.postdoc_institute, faculty_profiles.postdoc_institute),
                keywords=COALESCE(excluded.keywords, faculty_profiles.keywords),
                profile_url=excluded.profile_url,
                updated_at=excluded.updated_at
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, list(record.values()))
            conn.commit()
            return True

    def bulk_write_profiles(self, profiles: List[Dict[str, Any]]) -> int:
        """Bulk insert faculty profile records with transaction safety."""
        count = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            for profile in profiles:
                self.write_profile(profile)
                count += 1
            conn.commit()
        return count

    def log_scrape(self, url: str, status_code: int, error: Optional[str] = None) -> None:
        """Record scraping audit log."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO scraping_logs (url, status_code, error_message) VALUES (?, ?, ?)",
                (url, status_code, error)
            )
            conn.commit()

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================
    def read_profile(self, faculty_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single profile by its primary key faculty_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM faculty_profiles WHERE faculty_id = ?", (faculty_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def read_all_profiles(self, 
                          institution: Optional[str] = None, 
                          department_code: Optional[str] = None,
                          designation: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all profiles matching specified optional filters."""
        query = "SELECT * FROM faculty_profiles WHERE 1=1"
        params: List[Any] = []

        if institution:
            query += " AND institution = ?"
            params.append(institution)
        if department_code:
            query += " AND department_code = ?"
            params.append(department_code)
        if designation:
            query += " AND designation LIKE ?"
            params.append(f"%{designation}%")

        query += " ORDER BY department_code, name ASC"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search profiles where keyword appears in research area, interests, or keywords."""
        query = """
            SELECT * FROM faculty_profiles
            WHERE research_area LIKE ?
               OR research_interests LIKE ?
               OR keywords LIKE ?
               OR name LIKE ?
            ORDER BY name ASC
        """
        pattern = f"%{keyword}%"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (pattern, pattern, pattern, pattern))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # MODIFY / UPDATE OPERATIONS
    # =========================================================================
    def modify_profile(self, faculty_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific columns of an existing faculty profile.
        
        Args:
            faculty_id: Unique identifier of profile.
            updates: Dictionary of column-value pairs to modify.
        Returns:
            bool: True if record was updated, False if not found.
        """
        if not updates:
            return False

        allowed_cols = {
            'name', 'institution', 'department', 'department_code', 'designation',
            'email', 'office', 'research_area', 'research_interests', 'education_raw',
            'phd_institute', 'phd_country', 'phd_year', 'postdoc_institute',
            'keywords', 'profile_url'
        }
        
        valid_updates = {k: v for k, v in updates.items() if k in allowed_cols}
        if not valid_updates:
            raise ValueError(f"No valid columns provided for update. Allowed: {allowed_cols}")

        valid_updates['updated_at'] = datetime.now(timezone.utc).isoformat()
        set_clauses = [f"{col} = ?" for col in valid_updates.keys()]
        query = f"UPDATE faculty_profiles SET {', '.join(set_clauses)} WHERE faculty_id = ?"
        params = list(valid_updates.values()) + [faculty_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================
    def delete_profile(self, faculty_id: str) -> bool:
        """Delete a faculty profile from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faculty_profiles WHERE faculty_id = ?", (faculty_id,))
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # PANDAS & NUMPY INTEGRATION (Requirement 1 & 2)
    # =========================================================================
    def to_dataframe(self, query: Optional[str] = None, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Export database records into a Pandas DataFrame.
        Fulfills Requirement 1 (use NumPy, Pandas to store and manipulate data).
        """
        with self.get_connection() as conn:
            if query is None:
                query = "SELECT * FROM faculty_profiles"
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def save_dataframe(self, df: pd.DataFrame, table_name: str = "faculty_profiles", if_exists: str = "append") -> int:
        """
        Save/Append a Pandas DataFrame directly to SQLite database.
        """
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            return len(df)

    # =========================================================================
    # SUMMARY AGGREGATIONS
    # =========================================================================
    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate summary counts from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM faculty_profiles")
            total_faculty = cursor.fetchone()[0]

            cursor.execute("SELECT institution, COUNT(*) FROM faculty_profiles GROUP BY institution")
            by_institution = dict(cursor.fetchall())

            cursor.execute("SELECT department, COUNT(*) FROM faculty_profiles GROUP BY department ORDER BY COUNT(*) DESC")
            by_department = dict(cursor.fetchall())

            cursor.execute("SELECT designation, COUNT(*) FROM faculty_profiles GROUP BY designation ORDER BY COUNT(*) DESC")
            by_designation = dict(cursor.fetchall())

            cursor.execute("SELECT phd_country, COUNT(*) FROM faculty_profiles WHERE phd_country IS NOT NULL GROUP BY phd_country ORDER BY COUNT(*) DESC LIMIT 10")
            by_phd_country = dict(cursor.fetchall())

            return {
                "total_profiles": total_faculty,
                "institutions": by_institution,
                "departments": by_department,
                "designations": by_designation,
                "top_phd_countries": by_phd_country
            }
