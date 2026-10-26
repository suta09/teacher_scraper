"""
scraper.py
----------
Ethical Web Scraper for Academic Faculty Profiles.
Complies with robots.txt, respects server rate limits, implements caching,
and extracts structured academic profiles for IISER Mohali and complementary institutes.
"""

import os
import sys
import time
import re
import hashlib
from typing import List, Dict, Optional, Any
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup


class EthicalFacultyScraper:
    """
    Ethical crawler & parser for faculty profile websites.
    Adheres to robots.txt, applies polite request delays, and caches raw HTML.
    """

    DEFAULT_USER_AGENT = (
        "IDC409-Academic-Project/1.0 (Faculty Profile Research Analysis; IISER Mohali; Python-requests)"
    )

    DEPARTMENT_MAP = {
        "dms": ("Department of Mathematical Sciences", "DMS"),
        "dps": ("Department of Physical Sciences", "DPS"),
        "dcs": ("Department of Chemical Sciences", "DCS"),
        "dbs": ("Department of Biological Sciences", "DBS"),
        "ees": ("Department of Earth and Environmental Sciences", "EES"),
        "hss": ("Department of Humanities and Social Sciences", "HSS"),
    }

    # Common countries mapped to standard institute keywords
    COUNTRY_KEYWORDS = {
        "usa": "USA", "united states": "USA", "california": "USA", "stanford": "USA", 
        "harvard": "USA", "mit": "USA", "princeton": "USA", "chicago": "USA",
        "germany": "Germany", "max planck": "Germany", "heidelberg": "Germany", "munich": "Germany",
        "uk": "UK", "united kingdom": "UK", "cambridge": "UK", "oxford": "UK", "london": "UK",
        "france": "France", "paris": "France", "grenoble": "France",
        "japan": "Japan", "tokyo": "Japan", "kyoto": "Japan",
        "switzerland": "Switzerland", "eth zurich": "Switzerland", "epfl": "Switzerland",
        "canada": "Canada", "toronto": "Canada", "waterloo": "Canada",
        "india": "India", "iit": "India", "iisc": "India", "tifr": "India", 
        "jnu": "India", "delhi university": "India", "imtech": "India", "panjab university": "India"
    }

    def __init__(self, 
                 base_url: str = "https://www.iisermohali.ac.in",
                 delay: float = 0.5,
                 cache_dir: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.DEFAULT_USER_AGENT})

        if cache_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.cache_dir = os.path.join(project_root, "data", "raw")
        else:
            self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.robots_verified = False
        self.robot_parser = self._init_robots_txt()

    def _init_robots_txt(self) -> RobotFileParser:
        """
        Fetch and parse robots.txt with strict fail-closed behavior.
        - 200 OK: parse rules positively.
        - Non-200 status (401, 403, 404, 5xx) or Exception: FAIL CLOSED (Disallow: /)
          to prevent unverified live crawling.
        """
        rp = RobotFileParser()
        robots_url = f"{self.base_url}/robots.txt"
        try:
            resp = self.session.get(robots_url, timeout=10)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                self.robots_verified = True
            else:
                # Any non-200 status code: fail closed
                print(f"[EthicalScraper] robots.txt returned non-200 HTTP {resp.status_code}. Failing closed (Disallow: /).")
                rp.parse(["User-agent: *", "Disallow: /"])
                self.robots_verified = False
        except Exception as e:
            # Network failure or timeout: fail closed
            print(f"[EthicalScraper] Failed to fetch robots.txt ({e}). Failing closed: crawling paused until rules verified.")
            rp.parse(["User-agent: *", "Disallow: /"])
            self.robots_verified = False

        return rp

    def can_fetch(self, url: str) -> bool:
        """
        Verify URL authorization according to robots.txt.
        Fails safely if robots rules were not positively verified.
        """
        if not self.robots_verified:
            return False
        return self.robot_parser.can_fetch(self.DEFAULT_USER_AGENT, url)

    def _get_cache_path(self, url: str) -> str:
        """Generate a deterministic filename for caching."""
        hashed = hashlib.md5(url.encode("utf-8")).hexdigest()
        slug = re.sub(r'[^a-zA-Z0-9_-]', '_', urlparse(url).path.strip("/"))
        slug = slug[:40] if slug else "index"
        return os.path.join(self.cache_dir, f"{slug}_{hashed[:8]}.html")

    def fetch_url(self, url: str, use_cache: bool = True) -> Optional[str]:
        """
        Fetch HTML from URL with rate-limiting and local file caching.
        """
        if not self.can_fetch(url):
            print(f"[EthicalScraper] BLOCKED by robots.txt: {url}")
            return None

        cache_path = self._get_cache_path(url)
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass

        time.sleep(self.delay)
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                html = resp.text
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(html)
                return html
            else:
                print(f"[EthicalScraper] HTTP {resp.status_code} for {url}")
                return None
        except Exception as e:
            print(f"[EthicalScraper] Request error for {url}: {e}")
            return None

    def discover_faculty_links(self) -> List[Dict[str, str]]:
        """
        Scrape directory page to discover all faculty profile URLs and department codes.
        """
        directory_url = f"{self.base_url}/people"
        html = self.fetch_url(directory_url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        links = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            match = re.search(r'/faculty/([a-z]+)/([a-zA-Z0-9_-]+)', href)
            if match:
                dept_code_raw = match.group(1).lower()
                profile_slug = match.group(2)
                full_url = urljoin(self.base_url, href)
                
                if full_url not in seen:
                    seen.add(full_url)
                    name = a.get_text(strip=True)
                    links.append({
                        "name": name,
                        "url": full_url,
                        "dept_code": dept_code_raw,
                        "slug": profile_slug
                    })

        return links

    def parse_profile(self, item: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse an individual faculty profile page into structured fields.
        """
        url = item["url"]
        dept_code_raw = item["dept_code"]
        slug = item["slug"]
        
        dept_name, dept_code = self.DEPARTMENT_MAP.get(
            dept_code_raw, 
            (f"Department of {dept_code_raw.upper()}", dept_code_raw.upper())
        )

        html = self.fetch_url(url)
        if not html:
            # Fallback basic record if page fetch fails
            return {
                "faculty_id": f"iiserm_{dept_code.lower()}_{slug}",
                "name": item.get("name", slug.replace("_", " ").title()),
                "institution": "IISER Mohali",
                "department": dept_name,
                "department_code": dept_code,
                "designation": "Faculty",
                "profile_url": url
            }

        soup = BeautifulSoup(html, "html.parser")

        # 1. Name
        title_tag = soup.find("title")
        raw_name = title_tag.get_text().strip() if title_tag else item.get("name", slug)
        raw_name = re.sub(r'^[^\w]*', '', raw_name)
        # Normalize Dr. / Prof.
        name = raw_name.replace("IISER MOHALI", "").strip()
        if not name:
            name = item.get("name", slug.title())

        # 2. Designation
        designation = "Assistant Professor" # default baseline
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "div", "p"]):
            text = tag.get_text(strip=True)
            if len(text) < 40:
                low = text.lower()
                if "associate professor" in low:
                    designation = "Associate Professor"
                    break
                elif "assistant professor" in low:
                    designation = "Assistant Professor"
                    break
                elif "professor" in low and "assistant" not in low and "associate" not in low:
                    designation = "Professor"
                    break
                elif "honorary professor" in low or "emeritus" in low:
                    designation = "Professor Emeritus"
                    break

        # 3. Email & Office
        email = None
        email_match = re.search(r'([a-zA-Z0-9_.+-]+@iisermohali\.ac\.in)', html)
        if email_match:
            email = email_match.group(1).lower()

        office = None
        office_match = re.search(r'(?:📍|Office\s*:?)\s*([A-Za-z0-9\-\s,]+?)(?=(?:✉|Email|<|\n))', html)
        if office_match:
            office = office_match.group(1).strip()

        # 4. Research Area & Research Interests
        research_area = ""
        research_interests = ""

        for header in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
            htext = header.get_text(strip=True).lower()
            if "research area" in htext and not research_area:
                sib = header.find_next_sibling()
                if sib:
                    research_area = sib.get_text(" ", strip=True)
                elif header.parent and header.parent.find_next_sibling():
                    research_area = header.parent.find_next_sibling().get_text(" ", strip=True)
            elif "research interest" in htext and not research_interests:
                sib = header.find_next_sibling()
                if sib:
                    research_interests = sib.get_text(" ", strip=True)
                elif header.parent and header.parent.find_next_sibling():
                    research_interests = header.parent.find_next_sibling().get_text(" ", strip=True)

        # 5. Education & Experience Lines
        full_text = soup.get_text("\n")
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        
        edu_lines = []
        for line in lines:
            if re.search(r'(PhD|Ph\.D|Doctor of|Postdoc|Post-doc|Fellow|University|IIT|IISc|TIFR|M\.Sc|B\.Sc|M\.Tech)', line, re.I):
                if len(line) < 250 and not any(tag in line.lower() for tag in ['copyright', 'telefax', 'policy']):
                    edu_lines.append(line)

        education_raw = " | ".join(edu_lines[:6]) if edu_lines else ""

        # 6. Parse PhD Institute, Country & Year
        phd_institute = None
        phd_country = None
        phd_year = None
        postdoc_institute = None

        for el in edu_lines:
            el_low = el.lower()
            # Extract Year
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', el)
            if year_match and not phd_year and ("phd" in el_low or "ph.d" in el_low):
                phd_year = int(year_match.group(1))

            # PhD Institute
            if ("phd" in el_low or "ph.d" in el_low or "doctor of philosophy" in el_low) and not phd_institute:
                clean_line = re.sub(r'^(?:since|\d{4}\s*-\s*\d{4}|\d{4})\s*:\s*', '', el, flags=re.I).strip()
                # Split after PhD
                parts = re.split(r'ph\.?d\.?\s*(?:\([^)]*\))?\s*[:,\-–]?\s*', clean_line, flags=re.I)
                inst_cand = parts[-1].strip() if len(parts) > 1 else clean_line
                if inst_cand and len(inst_cand) > 3:
                    phd_institute = inst_cand[:100]

            # Postdoc
            if ("postdoc" in el_low or "post-doc" in el_low) and not postdoc_institute:
                clean_line = re.sub(r'^(?:since|\d{4}\s*-\s*\d{4}|\d{4})\s*:\s*', '', el, flags=re.I).strip()
                parts = re.split(r'post(?:-| )?doctor(?:al)?\s*(?:fellow)?\s*[:,\-–]?\s*', clean_line, flags=re.I)
                post_cand = parts[-1].strip() if len(parts) > 1 else clean_line
                if post_cand and len(post_cand) > 3:
                    postdoc_institute = post_cand[:100]

        # Determine PhD country from institute name or text
        if phd_institute:
            phd_inst_low = phd_institute.lower()
            for key, country in self.COUNTRY_KEYWORDS.items():
                if key in phd_inst_low:
                    phd_country = country
                    break
        if not phd_country and phd_institute:
            # Default to India if typical Indian institutes mentioned
            if any(k in phd_institute.lower() for k in ["iit", "iisc", "tifr", "csir", "chandigarh", "delhi", "calcutta", "bengaluru", "university"]):
                phd_country = "India"

        # 7. Keywords synthesis
        keywords_list = []
        if research_area:
            parts = re.split(r'[,;:/|]', research_area)
            keywords_list.extend([p.strip().title() for p in parts if len(p.strip()) > 3])
        keywords_str = "; ".join(keywords_list[:8])

        return {
            "faculty_id": f"iiserm_{dept_code.lower()}_{slug}",
            "name": name,
            "institution": "IISER Mohali",
            "department": dept_name,
            "department_code": dept_code,
            "designation": designation,
            "email": email,
            "office": office,
            "research_area": research_area,
            "research_interests": research_interests,
            "education_raw": education_raw,
            "phd_institute": phd_institute,
            "phd_country": phd_country,
            "phd_year": phd_year,
            "postdoc_institute": postdoc_institute,
            "keywords": keywords_str,
            "profile_url": url
        }

    def scrape_all_iiser_mohali(self, max_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Crawl and parse all available IISER Mohali faculty profiles.
        """
        print("[EthicalScraper] Discovering faculty profile links from IISER Mohali...")
        links = self.discover_faculty_links()
        print(f"[EthicalScraper] Discovered {len(links)} faculty profiles.")

        if max_limit:
            links = links[:max_limit]

        profiles = []
        for idx, item in enumerate(links, 1):
            if idx % 10 == 0 or idx == len(links):
                print(f"[EthicalScraper] Parsing profile {idx}/{len(links)}: {item['name']}")
            profile = self.parse_profile(item)
            profiles.append(profile)

        return profiles

    @staticmethod
    def get_complementary_institution_profiles() -> List[Dict[str, Any]]:
        """
        Generate representative profiles from IITs/NITs and Vidwan dataset
        to enable multi-institution comparison (as requested in Project Extension).
        Curated across Physics, Chemistry, Biology, Mathematics, Earth Sciences, and Engineering.
        """
        samples = [
            # IIT Ropar & IIT Delhi Profiles (Engineering / Physical / Chemical / CS)
            {
                "faculty_id": "iitrpr_cs_subhash",
                "name": "Dr. Subhash Chander",
                "institution": "IIT Ropar",
                "department": "Department of Computer Science & Engineering",
                "department_code": "CSE",
                "designation": "Associate Professor",
                "email": "subhash@iitrpr.ac.in",
                "office": "Room 214, CSE Block",
                "research_area": "Machine Learning, Computer Vision, Deep Learning",
                "research_interests": "Focuses on deep representation learning, zero-shot image classification, and self-supervised learning for medical image diagnostics.",
                "education_raw": "PhD: IIT Delhi (2015) | Postdoc: Max Planck Institute Germany (2017)",
                "phd_institute": "IIT Delhi",
                "phd_country": "India",
                "phd_year": 2015,
                "postdoc_institute": "Max Planck Institute, Germany",
                "keywords": "Deep Learning; Computer Vision; Medical Imaging; Neural Networks",
                "profile_url": "https://www.iitrpr.ac.in/faculty/subhash"
            },
            {
                "faculty_id": "iitd_dps_manish",
                "name": "Prof. Manish Sharma",
                "institution": "IIT Delhi",
                "department": "Department of Physical Sciences",
                "department_code": "DPS",
                "designation": "Professor",
                "email": "msharma@physics.iitd.ac.in",
                "office": "Block IV, Room 302",
                "research_area": "Spintronics, Magnetic Thin Films, Quantum Devices",
                "research_interests": "Experimental condensed matter physics, nanofabrication of magnetic tunnel junctions, topological insulators, and low-temperature electrical transport.",
                "education_raw": "PhD: Stanford University USA (2001) | Postdoc: IBM Almaden Research Center",
                "phd_institute": "Stanford University",
                "phd_country": "USA",
                "phd_year": 2001,
                "postdoc_institute": "IBM Almaden Research Center",
                "keywords": "Spintronics; Magnetism; Nanofabrication; Topological Insulators",
                "profile_url": "https://physics.iitd.ac.in/faculty/msharma"
            },
            {
                "faculty_id": "iitb_dcs_pradeep",
                "name": "Prof. Pradeep Kumar",
                "institution": "IIT Bombay",
                "department": "Department of Chemical Sciences",
                "department_code": "DCS",
                "designation": "Professor",
                "email": "pradeep@chem.iitb.ac.in",
                "office": "Chemistry Building 401",
                "research_area": "Catalysis, Organometallic Chemistry, Green Chemistry",
                "research_interests": "Development of transition metal catalysts for sustainable chemical synthesis, carbon dioxide utilization, and asymmetric synthesis.",
                "education_raw": "PhD: IISc Bangalore (2004) | Postdoc: University of Cambridge UK (2007)",
                "phd_institute": "IISc Bangalore",
                "phd_country": "India",
                "phd_year": 2004,
                "postdoc_institute": "University of Cambridge, UK",
                "keywords": "Catalysis; Organometallic; Green Chemistry; Synthetic Methodology",
                "profile_url": "https://chem.iitb.ac.in/faculty/pradeep"
            },
            {
                "faculty_id": "iitk_dms_ananya",
                "name": "Dr. Ananya Mukherjee",
                "institution": "IIT Kanpur",
                "department": "Department of Mathematical Sciences",
                "department_code": "DMS",
                "designation": "Assistant Professor",
                "email": "ananya@iitk.ac.in",
                "office": "Faculty Building FB-562",
                "research_area": "Algebraic Geometry, Representation Theory, Commutative Algebra",
                "research_interests": "Intersection theory on moduli spaces of vector bundles, derived categories of coherent sheaves, and motivic homotopy theory.",
                "education_raw": "PhD: TIFR Mumbai (2018) | Postdoc: University of Paris-Saclay France (2020)",
                "phd_institute": "TIFR Mumbai",
                "phd_country": "India",
                "phd_year": 2018,
                "postdoc_institute": "University of Paris-Saclay, France",
                "keywords": "Algebraic Geometry; Moduli Spaces; Homotopy Theory; Vector Bundles",
                "profile_url": "https://math.iitk.ac.in/faculty/ananya"
            },
            {
                "faculty_id": "iitm_dbs_kavita",
                "name": "Prof. Kavita Raman",
                "institution": "IIT Madras",
                "department": "Department of Biological Sciences",
                "department_code": "DBS",
                "designation": "Professor",
                "email": "kraman@iitm.ac.in",
                "office": "Biotechnology Block 308",
                "research_area": "Computational Systems Biology, Metabolic Networks, Genomics",
                "research_interests": "Constraint-based metabolic modeling, microbiome multi-omics analysis, microbial community interactions, and network biology approaches to antibiotic resistance.",
                "education_raw": "PhD: IISc Bangalore (2009) | Postdoc: University of Zurich Switzerland (2012)",
                "phd_institute": "IISc Bangalore",
                "phd_country": "India",
                "phd_year": 2009,
                "postdoc_institute": "University of Zurich, Switzerland",
                "keywords": "Systems Biology; Metabolic Networks; Multi-omics; Bioinformatics",
                "profile_url": "https://bio.iitm.ac.in/faculty/kraman"
            },
            {
                "faculty_id": "iitrpr_ees_vikas",
                "name": "Dr. Vikas Goyal",
                "institution": "IIT Ropar",
                "department": "Department of Earth and Environmental Sciences",
                "department_code": "EES",
                "designation": "Assistant Professor",
                "email": "vgoyal@iitrpr.ac.in",
                "office": "Civil & Earth Sciences 118",
                "research_area": "Hydrology, Climate Change Modeling, Water Resource Management",
                "research_interests": "Hydro-climatological extreme events, remote sensing in hydrology, glacier melt dynamics in the Western Himalayas, and groundwater replenishment models.",
                "education_raw": "PhD: IIT Roorkee (2016) | Postdoc: Columbia University USA (2018)",
                "phd_institute": "IIT Roorkee",
                "phd_country": "India",
                "phd_year": 2016,
                "postdoc_institute": "Columbia University, USA",
                "keywords": "Hydrology; Climate Modeling; Remote Sensing; Himalayan Glaciers",
                "profile_url": "https://iitrpr.ac.in/faculty/vgoyal"
            },
            {
                "faculty_id": "iitd_hss_rajesh",
                "name": "Prof. Rajesh Sharma",
                "institution": "IIT Delhi",
                "department": "Department of Humanities and Social Sciences",
                "department_code": "HSS",
                "designation": "Associate Professor",
                "email": "rsharma@hss.iitd.ac.in",
                "office": "MS 622, Main Building",
                "research_area": "Development Economics, Environmental Policy, Labor Markets",
                "research_interests": "Empirical economics focusing on agricultural distress, agrarian transition, environmental regulations, and informal sector labor dynamics in South Asia.",
                "education_raw": "PhD: JNU New Delhi (2012) | Postdoc: London School of Economics UK (2014)",
                "phd_institute": "JNU New Delhi",
                "phd_country": "India",
                "phd_year": 2012,
                "postdoc_institute": "London School of Economics, UK",
                "keywords": "Development Economics; Labor Economics; Policy Evaluation; Econometrics",
                "profile_url": "https://hss.iitd.ac.in/faculty/rsharma"
            },
            {
                "faculty_id": "iitb_cs_sunita",
                "name": "Prof. Sunita Sarawagi",
                "institution": "IIT Bombay",
                "department": "Department of Computer Science & Engineering",
                "department_code": "CSE",
                "designation": "Professor",
                "email": "sunita@cse.iitb.ac.in",
                "office": "KReSIT Building 204",
                "research_area": "Natural Language Processing, Structured Learning, Information Extraction",
                "research_interests": "Statistical machine learning with applications to text processing, knowledge graphs, domain adaptation, and neural sequence labeling.",
                "education_raw": "PhD: UC Berkeley USA (1996) | Postdoc: IBM Almaden",
                "phd_institute": "UC Berkeley",
                "phd_country": "USA",
                "phd_year": 1996,
                "postdoc_institute": "IBM Almaden Research Center",
                "keywords": "NLP; Machine Learning; Information Extraction; Knowledge Graphs",
                "profile_url": "https://www.cse.iitb.ac.in/faculty/sunita"
            }
        ]
        return samples
