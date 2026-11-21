"""
StatVision OCR — Offline Statistical Distribution & Property Analyzer.
"""

from stat_vision.knowledge_base import DISTRIBUTIONS_DB, DistributionSpec, get_distribution_by_id, list_all_distributions
from stat_vision.analyzer import StatisticalImageAnalyzer, DistributionReport
from stat_vision.symbolic_engine import compute_numeric_summary, evaluate_curve_points
from stat_vision.sample_generator import generate_distribution_sample_image, generate_all_demo_samples

__all__ = [
    "DISTRIBUTIONS_DB",
    "DistributionSpec",
    "get_distribution_by_id",
    "list_all_distributions",
    "StatisticalImageAnalyzer",
    "DistributionReport",
    "compute_numeric_summary",
    "evaluate_curve_points",
    "generate_distribution_sample_image",
    "generate_all_demo_samples",
]
