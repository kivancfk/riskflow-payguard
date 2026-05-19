"""Load IEEE-CIS fraud data and join transaction + identity tables.

TODO:
    - download_raw(): pull from Kaggle (or assume manual download into data/raw/)
    - load_raw(): read transaction + identity csvs
    - join_tables(): left-join identity onto transaction
    - basic_clean(): dtypes, obvious nulls
    - split(): time-aware train/val/test split (NOT random — fraud datasets are temporal)
"""
import pandas as pd


def load_raw(data_dir: str = "data/raw") -> pd.DataFrame:
    raise NotImplementedError
