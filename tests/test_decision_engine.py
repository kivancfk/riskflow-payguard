import pytest

from api.decision_engine import decide


def test_allow_below_review():
    assert decide(0.05, review_threshold=0.30, block_threshold=0.70) == "ALLOW"


def test_review_between_thresholds():
    assert decide(0.50, review_threshold=0.30, block_threshold=0.70) == "REVIEW"


def test_block_at_or_above_block_threshold():
    assert decide(0.70, review_threshold=0.30, block_threshold=0.70) == "BLOCK"
    assert decide(0.95, review_threshold=0.30, block_threshold=0.70) == "BLOCK"


def test_boundary_review_threshold():
    # Exactly at review threshold should escalate to REVIEW, not ALLOW.
    assert decide(0.30, review_threshold=0.30, block_threshold=0.70) == "REVIEW"


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        decide(0.5, review_threshold=0.8, block_threshold=0.2)
