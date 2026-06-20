"""
tests/test_clean.py
Pricewise — Sprint 1 Cleaning Pipeline Tests

Run with:
    pytest tests/test_clean.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load raw train.csv once for all tests."""
    path = Path("data/raw/train.csv")
    assert path.exists(), "data/raw/train.csv not found — download from Kaggle first"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def clean_df():
    """
    Load clean_data.csv once for all tests.
    If it doesn't exist, run the pipeline first.
    """
    path = Path("data/processed/clean_data.csv")

    if not path.exists():
        from src.clean import run_pipeline
        run_pipeline()

    string_cols = ["MSSubClass", "MoSold", "YrSold"]
    df = pd.read_csv(
        path,
        dtype={col: str for col in string_cols},
        keep_default_na=False,
        na_values=[""]
    )
    return df


# ─────────────────────────────────────────────
# FILE TESTS
# ─────────────────────────────────────────────

class TestOutputFile:

    def test_output_file_exists(self):
        """clean_data.csv must exist after pipeline runs."""
        assert Path("data/processed/clean_data.csv").exists(), \
            "clean_data.csv not found — run src/clean.py first"

    def test_output_file_not_empty(self):
        """clean_data.csv must not be empty."""
        size = Path("data/processed/clean_data.csv").stat().st_size
        assert size > 0, "clean_data.csv is empty"

    def test_output_file_readable(self, clean_df):
        """clean_data.csv must be readable as a dataframe."""
        assert isinstance(clean_df, pd.DataFrame)


# ─────────────────────────────────────────────
# SHAPE TESTS
# ─────────────────────────────────────────────

class TestShape:

    def test_row_count_reasonable(self, clean_df):
        """Must have at least 1458 rows — started with 1460, max 2 removed."""
        assert len(clean_df) >= 1458, \
            f"Too many rows removed: {len(clean_df)} remaining"

    def test_row_count_not_exceeded(self, clean_df, raw_df):
        """Cannot have more rows than raw data."""
        assert len(clean_df) <= len(raw_df), \
            f"Clean data has more rows than raw: {len(clean_df)} vs {len(raw_df)}"

    def test_column_count_reduced(self, clean_df, raw_df):
        """Clean data must have fewer columns than raw — dropped high-missing cols."""
        assert clean_df.shape[1] < raw_df.shape[1], \
            "No columns were dropped — check drop_high_missing()"

    def test_minimum_column_count(self, clean_df):
        """Must have at least 70 columns remaining."""
        assert clean_df.shape[1] >= 70, \
            f"Too few columns: {clean_df.shape[1]}"


# ─────────────────────────────────────────────
# MISSING VALUE TESTS
# ─────────────────────────────────────────────

class TestMissingValues:

    def test_zero_nulls(self, clean_df):
        """Zero null values across entire cleaned dataframe."""
        total_nulls = clean_df.isnull().sum().sum()
        assert total_nulls == 0, \
            f"Found {total_nulls} null values in clean_data.csv"

    def test_no_null_in_saleprice(self, clean_df):
        """SalePrice must have zero nulls."""
        assert clean_df["SalePrice"].isnull().sum() == 0

    def test_no_null_in_log_saleprice(self, clean_df):
        """LogSalePrice must have zero nulls."""
        assert clean_df["LogSalePrice"].isnull().sum() == 0

    def test_no_null_in_grlivarea(self, clean_df):
        """GrLivArea must have zero nulls."""
        assert clean_df["GrLivArea"].isnull().sum() == 0

    def test_no_null_in_lotfrontage(self, clean_df):
        """LotFrontage must be fully imputed."""
        assert clean_df["LotFrontage"].isnull().sum() == 0


# ─────────────────────────────────────────────
# TARGET VARIABLE TESTS
# ─────────────────────────────────────────────

class TestTargetVariable:

    def test_saleprice_exists(self, clean_df):
        """SalePrice column must be present."""
        assert "SalePrice" in clean_df.columns

    def test_saleprice_all_positive(self, clean_df):
        """All SalePrice values must be positive."""
        assert (clean_df["SalePrice"] > 0).all(), \
            "SalePrice contains non-positive values"

    def test_saleprice_reasonable_range(self, clean_df):
        """SalePrice must be within a reasonable range for Ames housing."""
        assert clean_df["SalePrice"].min() >= 10_000, \
            f"SalePrice min too low: {clean_df['SalePrice'].min()}"
        assert clean_df["SalePrice"].max() <= 1_000_000, \
            f"SalePrice max too high: {clean_df['SalePrice'].max()}"

    def test_log_saleprice_exists(self, clean_df):
        """LogSalePrice column must be present."""
        assert "LogSalePrice" in clean_df.columns

    def test_log_saleprice_is_finite(self, clean_df):
        """LogSalePrice must contain only finite values."""
        assert np.isfinite(clean_df["LogSalePrice"]).all(), \
            "LogSalePrice contains infinite or NaN values"

    def test_log_saleprice_matches_saleprice(self, clean_df):
        """LogSalePrice must equal log(SalePrice)."""
        expected = np.log(clean_df["SalePrice"])
        diff = (clean_df["LogSalePrice"] - expected).abs().max()
        assert diff < 1e-6, \
            f"LogSalePrice does not match log(SalePrice) — max diff: {diff}"

    def test_log_saleprice_skewness_reduced(self, clean_df):
        """LogSalePrice skewness must be below 0.5."""
        skew = clean_df["LogSalePrice"].skew()
        assert abs(skew) < 0.5, \
            f"LogSalePrice still skewed: {skew:.4f}"


# ─────────────────────────────────────────────
# DTYPE TESTS
# ─────────────────────────────────────────────

class TestDtypes:

    def test_numeric_cols_correct_dtype(self, clean_df):
        """All numeric columns must be int64 or float64."""
        numeric_cols = clean_df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            assert clean_df[col].dtype in [np.int64, np.float64], \
                f"{col} has unexpected dtype: {clean_df[col].dtype}"

    def test_mssubclass_is_string(self, clean_df):
        """MSSubClass must be stored as string — it's a category code."""
        assert pd.api.types.is_string_dtype(clean_df["MSSubClass"]), \
            "MSSubClass should be string, not numeric"

    def test_mosold_is_string(self, clean_df):
        """MoSold must be stored as string."""
        assert pd.api.types.is_string_dtype(clean_df["MoSold"]), \
            "MoSold should be string, not numeric"

    def test_yrsold_is_string(self, clean_df):
        """YrSold must be stored as string."""
        assert pd.api.types.is_string_dtype(clean_df["YrSold"]), \
            "YrSold should be string, not numeric"


# ─────────────────────────────────────────────
# OUTLIER TESTS
# ─────────────────────────────────────────────

class TestOutliers:

    def test_extreme_grlivarea_removed(self, clean_df):
        """The two extreme GrLivArea outliers must be removed."""
        extremes = clean_df[
            (clean_df["GrLivArea"] > 4000) &
            (clean_df["SalePrice"] < 200_000)
        ]
        assert len(extremes) == 0, \
            f"Extreme GrLivArea outliers still present: {len(extremes)} rows"

    def test_saleprice_not_over_winsorized(self, clean_df):
        """SalePrice must not have been Winsorized — it's excluded."""
        raw = pd.read_csv("data/raw/train.csv")
        assert clean_df["SalePrice"].max() <= raw["SalePrice"].max(), \
            "SalePrice max exceeds raw max — unexpected"


# ─────────────────────────────────────────────
# ORDINAL ENCODING TESTS
# ─────────────────────────────────────────────

class TestOrdinalEncoding:

    def test_overallqual_range(self, clean_df):
        """OverallQual must be 1-10."""
        assert clean_df["OverallQual"].between(1, 10).all(), \
            "OverallQual out of expected 1-10 range"

    def test_centralair_binary(self, clean_df):
        """CentralAir must be 0 or 1 after encoding."""
        assert set(clean_df["CentralAir"].unique()).issubset({0, 1}), \
            f"CentralAir has unexpected values: {clean_df['CentralAir'].unique()}"

    def test_pavedrive_range(self, clean_df):
        """PavedDrive must be 0, 1, or 2 after encoding."""
        assert set(clean_df["PavedDrive"].unique()).issubset({0, 1, 2}), \
            f"PavedDrive has unexpected values: {clean_df['PavedDrive'].unique()}"

    def test_exterqual_range(self, clean_df):
        """ExterQual must be 0-5 after encoding."""
        if "ExterQual" in clean_df.columns:
            assert clean_df["ExterQual"].between(0, 5).all(), \
                f"ExterQual out of 0-5 range"

    def test_kitchenqual_range(self, clean_df):
        """KitchenQual must be 0-5 after encoding."""
        if "KitchenQual" in clean_df.columns:
            assert clean_df["KitchenQual"].between(0, 5).all(), \
                f"KitchenQual out of 0-5 range"


# ─────────────────────────────────────────────
# PIPELINE FUNCTION TESTS
# ─────────────────────────────────────────────

class TestPipelineFunctions:

    def test_run_pipeline_returns_dataframe(self):
        """run_pipeline() must return a pandas DataFrame."""
        from src.clean import run_pipeline
        df = run_pipeline()
        assert isinstance(df, pd.DataFrame)

    def test_run_pipeline_zero_nulls(self):
        """run_pipeline() output must have zero nulls."""
        from src.clean import run_pipeline
        df = run_pipeline()
        assert df.isnull().sum().sum() == 0

    def test_load_raw_returns_dataframe(self):
        """load_raw() must return a DataFrame with expected shape."""
        from src.clean import load_raw
        df = load_raw()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1460, 81)

    def test_individual_functions_chainable(self):
        """All cleaning functions must be chainable without errors."""
        from src.clean import (
            load_raw, drop_high_missing, fill_absent_nulls,
            impute_true_nulls, remove_extremes, cap_outliers,
            fix_types, map_ordinals, add_log_target
        )
        df = load_raw()
        df = drop_high_missing(df)
        df = fill_absent_nulls(df)
        df = impute_true_nulls(df)
        df = remove_extremes(df)
        df = cap_outliers(df)
        df = fix_types(df)
        df = map_ordinals(df)
        df = add_log_target(df)
        assert df.isnull().sum().sum() == 0