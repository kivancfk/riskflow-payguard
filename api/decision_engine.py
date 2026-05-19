"""Maps a fraud probability to an ALLOW / REVIEW / BLOCK decision."""
from typing import Literal

Decision = Literal["ALLOW", "REVIEW", "BLOCK"]


def decide(
    fraud_probability: float,
    review_threshold: float,
    block_threshold: float,
) -> Decision:
    if review_threshold > block_threshold:
        raise ValueError("review_threshold must be <= block_threshold")
    if fraud_probability >= block_threshold:
        return "BLOCK"
    if fraud_probability >= review_threshold:
        return "REVIEW"
    return "ALLOW"
