"""
preprocessor.py
---------------
Text Cleaning, Tokenization, Normalization, and Keyword Processing
for Academic Faculty Profiles.
"""

import re
import string
from typing import List, Optional, Set
import pandas as pd


# Domain-specific academic stopwords that don't differentiate disciplines
ACADEMIC_STOPWORDS: Set[str] = {
    'research', 'group', 'aim', 'focus', 'focuses', 'interests', 'interested',
    'work', 'works', 'working', 'study', 'studies', 'studying', 'investigate',
    'investigating', 'investigates', 'understanding', 'understand', 'lab',
    'laboratory', 'department', 'current', 'present', 'project', 'projects',
    'field', 'fields', 'dr', 'prof', 'professor', 'using', 'used', 'use',
    'various', 'different', 'based', 'approach', 'approaches', 'development',
    'develop', 'developing', 'including', 'include', 'includes', 'also', 'well',
    'new', 'novel', 'many', 'like', 'one', 'two', 'via', 'e', 'g', 'i', 'etc'
}

STANDARD_ENGLISH_STOPWORDS: Set[str] = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren'
}

ALL_STOPWORDS = STANDARD_ENGLISH_STOPWORDS | ACADEMIC_STOPWORDS


def clean_text(text: Optional[str], remove_stopwords: bool = True) -> str:
    """
    Sanitize and normalize text:
    - Lowercase conversion
    - URL & email stripping
    - Punctuation removal
    - Stopword filtering (including academic noise terms)
    - Whitespace normalization
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove emails and URLs
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Lowercase
    text = text.lower()

    # Replace hyphens/slashes with space to preserve sub-words (e.g., physico-chemical -> physico chemical)
    text = re.sub(r'[-/\\_]', ' ', text)

    # Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 2]

    return " ".join(tokens)


def normalize_institution_name(inst_raw: Optional[str]) -> str:
    """
    Standardize common abbreviations and phrasing for PhD/Postdoc institutions.
    """
    if not inst_raw or not isinstance(inst_raw, str):
        return "Unknown"

    s = inst_raw.strip()
    low = s.lower()

    # Known Indian Premier Institutes
    if "iisc" in low:
        return "IISc Bangalore"
    if "tifr" in low:
        return "TIFR Mumbai"
    if "iit kanpur" in low or "iitk" in low:
        return "IIT Kanpur"
    if "iit bombay" in low or "iitb" in low:
        return "IIT Bombay"
    if "iit delhi" in low or "iitd" in low:
        return "IIT Delhi"
    if "iit madras" in low or "iitm" in low:
        return "IIT Madras"
    if "iit kharagpur" in low or "iitkgp" in low:
        return "IIT Kharagpur"
    if "iit roorkee" in low:
        return "IIT Roorkee"
    if "iit ropar" in low:
        return "IIT Ropar"
    if "panjab university" in low:
        return "Panjab University, Chandigarh"
    if "imtech" in low:
        return "CSIR-IMTech, Chandigarh"
    if "jnu" in low or "jawaharlal nehru university" in low:
        return "JNU New Delhi"
    if "delhi university" in low:
        return "University of Delhi"
    if "calcutta university" in low:
        return "University of Calcutta"
    if "iiser" in low:
        return "IISER"

    # Known International Institutes
    if "stanford" in low:
        return "Stanford University, USA"
    if "berkeley" in low:
        return "UC Berkeley, USA"
    if "san diego" in low or "ucsd" in low:
        return "UC San Diego, USA"
    if "cambridge" in low:
        return "University of Cambridge, UK"
    if "oxford" in low:
        return "University of Oxford, UK"
    if "max planck" in low:
        return "Max Planck Institute, Germany"
    if "eth" in low or "zurich" in low:
        return "ETH Zurich, Switzerland"

    # Clean short string
    return s[:60].strip()


def prepare_profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich raw faculty DataFrame with cleaned text, standardized institutes,
    and combined text features for machine learning.
    """
    df = df.copy()

    for col in ['research_area', 'research_interests', 'keywords']:
        if col not in df.columns:
            df[col] = ''

    # Clean research area and interests
    df['clean_research_area'] = df['research_area'].fillna('').apply(clean_text)
    df['clean_research_interests'] = df['research_interests'].fillna('').apply(clean_text)
    df['clean_keywords'] = df['keywords'].fillna('').apply(clean_text)

    # Combined text feature: weighting research area slightly higher by repetition
    df['feature_text'] = (
        df['clean_research_area'] + " " +
        df['clean_research_area'] + " " +
        df['clean_research_interests'] + " " +
        df['clean_keywords']
    ).str.strip()

    # Normalize PhD institute and country
    if 'phd_institute' in df.columns:
        df['norm_phd_institute'] = df['phd_institute'].apply(normalize_institution_name)
    else:
        df['norm_phd_institute'] = "Unknown"

    if 'phd_country' in df.columns:
        df['phd_country'] = df['phd_country'].fillna("Unknown")

    return df
