"""
src/features.py
Pricewise — Production Feature Engineering Pipeline
Sprint 3

Reads:  data/processed/clean_data.csv
Writes: data/processed/featured_data.csv

Usage:
    python src/features.py

    or as a module:
    from src.features import run_feature_pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

CLEAN_PATH    = Path("data/processed/clean_data.csv")
FEATURED_PATH = Path("data/processed/featured_data.csv")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

STRING_COLS = ["MSSubClass", "MoSold", "YrSold"]

TARGET     = "LogSalePrice"
TARGET_RAW = "SalePrice"

NZV_THRESHOLD      = 0.95   # near-zero variance threshold
DOMINANT_THRESHOLD = 0.90   # dominant categorical threshold
SKEW_THRESHOLD     = 1.0    # log1p transform threshold

EXCLUDE_FROM_LOG = [
    TARGET, TARGET_RAW,
    "OverallQual", "OverallCond",
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual",
    "FireplaceQu", "GarageQual", "GarageCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "GarageFinish", "PavedDrive", "CentralAir",
    "LotShape", "LandContour", "LandSlope",
    "Functional",
    "FullBath", "HalfBath", "BsmtFullBath",
    "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageCars",
    "has_garage", "has_basement", "has_remodelled",
    "house_age", "remod_age",
    "qual_x_sf", "overall_score",
    "total_bathrooms",
    "YearBuilt", "YearRemodAdd",
]

PORCH_COLS = [
    "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch"
]

BATH_COMPONENTS = {
    "FullBath"    : 1.0,
    "HalfBath"    : 0.5,
    "BsmtFullBath": 1.0,
    "BsmtHalfBath": 0.5,
}

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Load clean_data.csv with correct dtypes."""
    df = pd.read_csv(
        path,
        dtype={col: str for col in STRING_COLS},
        keep_default_na=False,
        na_values=[""]
    )
    assert df.isnull().sum().sum() == 0, "Nulls found on load"
    print(f"[load_clean]          Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def drop_low_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-zero variance numeric, dominant categorical, and Id columns."""
    drop_cols = []

    # Near-zero variance numeric
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if col in [TARGET, TARGET_RAW, "Id"]:
            continue
        if df[col].value_counts(normalize=True).iloc[0] > NZV_THRESHOLD:
            drop_cols.append(col)

    # Dominant categorical
    for col in df.select_dtypes(include="object").columns:
        if df[col].value_counts(normalize=True).iloc[0] > DOMINANT_THRESHOLD:
            drop_cols.append(col)

    # Id
    if "Id" in df.columns:
        drop_cols.append("Id")

    df = df.drop(columns=drop_cols)
    print(f"[drop_low_signal]     Dropped {len(drop_cols)} columns")
    return df


def add_size_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer total_sf, total_bathrooms, total_porch_sf."""

    # total_sf
    df["total_sf"] = df["TotalBsmtSF"] + df["GrLivArea"]

    # total_bathrooms — only use columns that exist
    bath_active = {col: w for col, w in BATH_COMPONENTS.items()
                   if col in df.columns}
    df["total_bathrooms"] = sum(
        df[col] * w for col, w in bath_active.items()
    )

    # total_porch_sf
    porch_active = [c for c in PORCH_COLS if c in df.columns]
    df["total_porch_sf"] = df[porch_active].sum(axis=1)

    print(f"[add_size_features]   Added: total_sf, total_bathrooms, total_porch_sf")
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer house_age, remod_age, has_remodelled."""

    yr_sold = df["YrSold"].astype(int)

    df["house_age"]      = yr_sold - df["YearBuilt"]
    df["remod_age"]      = yr_sold - df["YearRemodAdd"]
    df["has_remodelled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    assert (df["house_age"] >= 0).all(), "Negative house_age values found"
    assert (df["remod_age"] >= 0).all(), "Negative remod_age values found"

    print(f"[add_age_features]    Added: house_age, remod_age, has_remodelled")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer qual_x_sf and overall_score."""

    df["qual_x_sf"]     = df["OverallQual"] * df["GrLivArea"]
    df["overall_score"] = df["OverallQual"] * df["OverallCond"]

    print(f"[add_interaction_features] Added: qual_x_sf, overall_score")
    return df


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer has_garage and has_basement binary flags."""

    df["has_garage"]   = (df["GarageArea"] > 0).astype(int)
    df["has_basement"] = (df["TotalBsmtSF"] > 0).astype(int)

    # has_pool skipped — PoolArea near-zero variance (13/1458 houses)

    print(f"[add_binary_flags]    Added: has_garage, has_basement")
    return df


def add_neighbourhood_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer neighbourhood_tier — Low / Mid / High based on median price."""

    neighbourhood_medians = (
        df.groupby("Neighborhood")[TARGET_RAW]
        .median()
    )

    tier_33 = neighbourhood_medians.quantile(0.33)
    tier_66 = neighbourhood_medians.quantile(0.66)

    def assign_tier(neighbourhood):
        median = neighbourhood_medians[neighbourhood]
        if median >= tier_66:
            return "High"
        elif median >= tier_33:
            return "Mid"
        else:
            return "Low"

    df["neighbourhood_tier"] = df["Neighborhood"].apply(assign_tier)

    print(f"[add_neighbourhood_tier] Added: neighbourhood_tier")
    print(f"  Tier thresholds — Low < ${tier_33:,.0f}, "
          f"Mid ${tier_33:,.0f}–${tier_66:,.0f}, "
          f"High ≥ ${tier_66:,.0f}")
    return df


def apply_log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to highly skewed numeric features."""

    numeric_eligible = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col not in EXCLUDE_FROM_LOG
    ]

    skew_vals       = df[numeric_eligible].skew().abs()
    transform_cols  = skew_vals[skew_vals > SKEW_THRESHOLD].index.tolist()

    for col in transform_cols:
        df[col] = np.log1p(df[col])

    print(f"[apply_log1p_transform] Transformed {len(transform_cols)} columns")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Neighborhood, one-hot encode all remaining object columns."""

    # Drop Neighborhood — replaced by neighbourhood_tier
    if "Neighborhood" in df.columns:
        df = df.drop(columns=["Neighborhood"])

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True,
        dtype=int
    )

    print(f"[encode_categoricals]  Encoded {len(cat_cols)} categorical columns")
    print(f"[encode_categoricals]  Final shape: {df.shape}")
    return df


def validate(df: pd.DataFrame) -> None:
    """Run final assertions before saving."""
    assert df.isnull().sum().sum() == 0, \
        "Nulls present in featured dataframe"
    assert len(df.select_dtypes(include="object").columns) == 0, \
        "Object columns remaining in featured dataframe"
    assert TARGET in df.columns, \
        "LogSalePrice missing"
    assert TARGET_RAW in df.columns, \
        "SalePrice missing"
    assert "qual_x_sf" in df.columns, \
        "qual_x_sf missing"
    assert "total_sf" in df.columns, \
        "total_sf missing"
    assert "house_age" in df.columns, \
        "house_age missing"
    print(f"[validate]            All assertions passed")


def save_featured(df: pd.DataFrame, path: Path = FEATURED_PATH) -> None:
    """Save featured dataframe and verify on reload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    saved = pd.read_csv(path)
    assert saved.shape == df.shape, \
        f"Shape mismatch on reload: {saved.shape} vs {df.shape}"
    assert saved.isnull().sum().sum() == 0, \
        "Nulls found in saved file"

    print(f"[save_featured]       Saved to {path}")
    print(f"[save_featured]       Shape : {saved.shape}")
    print(f"[save_featured]       Size  : {path.stat().st_size / 1024:.1f} KB")


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_feature_pipeline(
    clean_path: Path = CLEAN_PATH,
    output_path: Path = FEATURED_PATH
) -> pd.DataFrame:
    """Run full feature engineering pipeline end to end."""

    print("=" * 50)
    print("PRICEWISE — FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    df = load_clean(clean_path)
    df = drop_low_signal(df)
    df = add_size_features(df)
    df = add_age_features(df)
    df = add_interaction_features(df)
    df = add_binary_flags(df)
    df = add_neighbourhood_tier(df)
    df = apply_log1p_transform(df)
    df = encode_categoricals(df)

    validate(df)
    save_featured(df, output_path)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_feature_pipeline()