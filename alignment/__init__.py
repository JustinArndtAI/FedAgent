from .align_score import AlignmentScorer, alignment_score, bias_detect

# Import V2 if available
try:
    from .align_score_v2 import AlignmentScorerV2, alignment_score_v2
    # Use V2 as default
    AlignmentScorer = AlignmentScorerV2
    alignment_score = alignment_score_v2
except ImportError:
    pass

__all__ = ['AlignmentScorer', 'alignment_score', 'bias_detect']