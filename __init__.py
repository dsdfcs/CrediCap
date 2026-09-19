"""CrediCap: credibility-aware image-caption evaluation."""

from credicap.evidence_decomposition import IntegratedEvidenceScorer
from credicap.reference_credibility import ReferenceCredibilityModel
from credicap.score_correction import DirectionMagnitudeCorrector

__all__ = [
    "ReferenceCredibilityModel",
    "IntegratedEvidenceScorer",
    "DirectionMagnitudeCorrector",
]
