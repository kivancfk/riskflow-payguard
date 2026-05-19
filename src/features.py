"""Feature engineering. This is where model quality is won or lost.

TODO:
    - velocity features: txns per card per hour / day
    - aggregations: mean/std/count of amount per card, per email domain
    - time deltas: seconds since previous txn on same card
    - hour-of-day, day-of-week
    - card-age / account-age signals
    - target encoding for high-cardinality categoricals (with proper CV)
"""
import pandas as pd


FEATURE_COLUMNS: list[str] = []  # filled once features are designed


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
