"""
eda.py
------
Exploratory Data Analysis (EDA) and Visualization Generator.
Fulfills IDC409 Requirement 3:
  - Department-wise counts
  - Designation distributions
  - PhD institute and country trends
  - Keyword frequency analysis and Word Clouds
  - Multi-institution comparisons
"""

import os
import re
from collections import Counter
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use('Agg') # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


class FacultyProfileEDA:
    """Generates charts, wordclouds, and statistical summaries for faculty profiles."""

    def __init__(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        self.df = df.copy()
        if output_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.output_dir = os.path.join(base_dir, "reports", "figures")
        else:
            self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Matplotlib clean styling
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['axes.edgecolor'] = '#cccccc'

    def plot_department_distribution(self) -> str:
        """Bar plot of faculty count per department."""
        dept_counts = self.df['department'].value_counts()
        
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            x=dept_counts.values, 
            y=dept_counts.index, 
            palette="crest",
            hue=dept_counts.index,
            legend=False
        )
        plt.title("Faculty Distribution by Academic Department", fontsize=14, pad=12, fontweight='bold')
        plt.xlabel("Number of Faculty Members", fontsize=11)
        plt.ylabel("")

        for i, v in enumerate(dept_counts.values):
            ax.text(v + 0.3, i, str(v), va='center', fontweight='bold', color='#333333')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "01_department_counts.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    def plot_designation_breakdown(self) -> str:
        """Donut chart of faculty designations."""
        desig_counts = self.df['designation'].value_counts()

        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("Set2", len(desig_counts))
        wedges, texts, autotexts = plt.pie(
            desig_counts.values,
            labels=desig_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.45, edgecolor='w')
        )
        for t in texts:
            t.set_fontsize(11)
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')

        plt.title("Faculty Seniority & Designation Breakdown", fontsize=14, pad=12, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "02_designation_counts.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    def plot_top_phd_institutes(self, top_n: int = 10) -> str:
        """Horizontal bar chart of top PhD granting institutes."""
        phd_col = 'norm_phd_institute' if 'norm_phd_institute' in self.df.columns else 'phd_institute'
        valid = self.df[self.df[phd_col].notna() & (self.df[phd_col] != 'Unknown')]
        top_insts = valid[phd_col].value_counts().head(top_n)

        plt.figure(figsize=(10, 5.5))
        ax = sns.barplot(
            x=top_insts.values,
            y=top_insts.index,
            palette="viridis",
            hue=top_insts.index,
            legend=False
        )
        plt.title(f"Top {top_n} PhD Granting Institutions", fontsize=14, pad=12, fontweight='bold')
        plt.xlabel("Number of Faculty Alumni", fontsize=11)
        plt.ylabel("")

        for i, v in enumerate(top_insts.values):
            ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "03_top_phd_institutes.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    def plot_phd_country_trends(self) -> str:
        """Distribution of PhD training countries."""
        valid = self.df[self.df['phd_country'].notna() & (self.df['phd_country'] != 'Unknown')]
        country_counts = valid['phd_country'].value_counts()

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            x=country_counts.index,
            y=country_counts.values,
            palette="mako",
            hue=country_counts.index,
            legend=False
        )
        plt.title("PhD Training Origin: Country-Wise Trends", fontsize=14, pad=12, fontweight='bold')
        plt.ylabel("Faculty Count", fontsize=11)
        plt.xlabel("Country", fontsize=11)

        for i, v in enumerate(country_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "04_phd_countries.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    def plot_top_keywords(self, top_n: int = 20) -> str:
        """Frequency analysis of the most prominent academic research terms."""
        all_words = []
        for text in self.df['feature_text'].dropna():
            all_words.extend(text.split())

        word_counts = Counter(all_words).most_common(top_n)
        words, counts = zip(*word_counts) if word_counts else ([], [])

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x=list(counts),
            y=list(words),
            palette="rocket",
            hue=list(words),
            legend=False
        )
        plt.title(f"Top {top_n} Most Frequent Research Terms in Academic Profiles", fontsize=14, pad=12, fontweight='bold')
        plt.xlabel("Occurrences across Faculty Profiles", fontsize=11)
        plt.ylabel("")

        for i, v in enumerate(counts):
            ax.text(v + 0.5, i, str(v), va='center', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "05_top_research_keywords.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    def generate_wordcloud(self) -> Optional[str]:
        """Generate high-resolution wordcloud of faculty research expertise."""
        if not HAS_WORDCLOUD:
            print("[EDA] WordCloud library not installed; skipping wordcloud generation.")
            return None

        combined_text = " ".join(self.df['feature_text'].dropna().tolist())
        if not combined_text.strip():
            return None

        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='magma',
            max_words=120,
            random_state=42
        ).generate(combined_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title("Faculty Research Themes Word Cloud", fontsize=16, pad=14, fontweight='bold')
        plt.tight_layout(pad=0)

        save_path = os.path.join(self.output_dir, "06_research_wordcloud.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    def plot_institution_comparison(self) -> str:
        """Compare faculty count across institutes (IISER Mohali vs IIT/Others)."""
        inst_counts = self.df['institution'].value_counts()

        plt.figure(figsize=(8, 4.5))
        ax = sns.barplot(
            x=inst_counts.index,
            y=inst_counts.values,
            palette="Set1",
            hue=inst_counts.index,
            legend=False
        )
        plt.title("Institution-Wise Faculty Profile Breakdown", fontsize=14, pad=12, fontweight='bold')
        plt.ylabel("Number of Profiles", fontsize=11)
        plt.xlabel("Academic Institution", fontsize=11)

        for i, v in enumerate(inst_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "07_institution_comparison.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return save_path

    def generate_all_plots(self) -> Dict[str, str]:
        """Run complete visualization suite and return dictionary of generated image paths."""
        plots = {
            "department_counts": self.plot_department_distribution(),
            "designation_breakdown": self.plot_designation_breakdown(),
            "top_phd_institutes": self.plot_top_phd_institutes(),
            "phd_countries": self.plot_phd_country_trends(),
            "top_keywords": self.plot_top_keywords(),
            "institution_comparison": self.plot_institution_comparison()
        }
        wc_path = self.generate_wordcloud()
        if wc_path:
            plots["wordcloud"] = wc_path

        return plots
