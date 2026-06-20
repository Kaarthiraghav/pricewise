"""
src/clean.py
Pricewise — Production Cleaning Pipeline
Sprint 1

Reads:  data/raw/train.csv
Writes: data/processed/clean_data.csv

Usage:
    python src/clean.py
    
    or as a module:
    from src.clean import run_pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

RAW_PATH       = Path("data/raw/train.csv")
PROCESSED_PATH = Path("data/processed/clean_data.csv")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

ABSENT_CATEGORICAL = [
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtExposure",
    "BsmtFinType2",
    "BsmtFinType1",
    "BsmtCond",
    "BsmtQual",
    "MasVnrType",
]

ABSENT_NUMERIC = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]

TRULY_MISSING_CATEGORICAL = [
    "Electrical",
    "MSZoning",
    "Utilities",
    "Functional",
    "Exterior1st",
    "Exterior2nd",
    "KitchenQual",
    "SaleType",
]

TRULY_MISSING_NUMERIC = [
    "LotFrontage",
]

DROP_THRESHOLD = 80  # drop columns with >80% missing

NUMERIC_AS_CATEGORY = [
    "MSSubClass",
    "MoSold",
    "YrSold",
]

EXCLUDE_FROM_CAPPING = [
    "Id",
    "SalePrice",
    "MoSold",
    "YrSold",
    "YearBuilt",
    "YearRemodAdd",
    "GarageYrBlt",
    "OverallQual",
    "OverallCond",
]

ORDINAL_MAPPINGS = {
    "ExterQual"   : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond"   : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual"    : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond"    : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC"   : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual" : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu" : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual"  : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond"  : {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "LotShape"    : {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandContour" : {"Low": 1, "HLS": 2, "Bnk": 3, "Lvl": 4},
    "Utilities"   : {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    "LandSlope"   : {"Sev": 1, "Mod": 2, "Gtl": 3},
    "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Functional"  : {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
                     "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "GarageFinish": {"No": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "PavedDrive"  : {"N": 0, "P": 1, "Y": 2},
    "CentralAir"  : {"N": 0, "Y": 1},
}

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw train.csv from disk."""
    df = pd.read_csv(path)
    print(f"[load_raw]         Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def drop_high_missing(df: pd.DataFrame, threshold: int = DROP_THRESHOLD) -> pd.DataFrame:
    """Drop columns where missing % exceeds threshold."""
    missing_pct = df.isnull().mean() * 100
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    print(f"[drop_high_missing] Dropped {len(drop_cols)} columns: {drop_cols}")
    return df


def fill_absent_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill absent categorical nulls with 'No', absent numeric nulls with 0."""
    cat_active = [c for c in ABSENT_CATEGORICAL if c in df.columns]
    num_active = [c for c in ABSENT_NUMERIC if c in df.columns]

    df[cat_active] = df[cat_active].fillna("No")
    df[num_active] = df[num_active].fillna(0)

    print(f"[fill_absent_nulls] Filled {len(cat_active)} cat cols with 'No'")
    print(f"[fill_absent_nulls] Filled {len(num_active)} num cols with 0")
    return df


def impute_true_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Impute truly missing categoricals with mode, LotFrontage with neighbourhood median."""
    cat_active = [c for c in TRULY_MISSING_CATEGORICAL if c in df.columns]

    for col in cat_active:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # LotFrontage — neighbourhood median
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        global_median = df["LotFrontage"].median()
        df["LotFrontage"] = df["LotFrontage"].fillna(global_median)

    print(f"[impute_true_nulls] Imputed {len(cat_active)} cat cols with mode")
    print(f"[impute_true_nulls] LotFrontage imputed by neighbourhood median")
    return df


def remove_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 2 extreme GrLivArea outliers identified by Dean De Cock."""
    before = len(df)
    df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 200000))]
    df = df.reset_index(drop=True)
    print(f"[remove_extremes]   Removed {before - len(df)} extreme GrLivArea rows")
    return df


def cap_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Winsorize numeric columns using IQR method."""
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col not in EXCLUDE_FROM_CAPPING
    ]

    total_capped = 0
    cols_capped  = 0

    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        n_before = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_before > 0:
            df[col]      = df[col].clip(lower=lower, upper=upper)
            total_capped += n_before
            cols_capped  += 1

    print(f"[cap_outliers]      Capped {total_capped} values across {cols_capped} columns")
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-coded category columns to string, strip whitespace."""
    cols_active = [c for c in NUMERIC_AS_CATEGORY if c in df.columns]
    for col in cols_active:
        df[col] = df[col].astype(str)

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df[col] = df[col].str.strip()

    print(f"[fix_types]         Converted {len(cols_active)} cols to string")
    print(f"[fix_types]         Stripped whitespace from {len(obj_cols)} object cols")
    return df


def map_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ordinal integer mappings to ordered categorical columns."""
    mapped  = 0
    skipped = []

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col not in df.columns:
            skipped.append(col)
            continue
        df[col] = df[col].map(mapping)
        mapped += 1

    print(f"[map_ordinals]      Mapped {mapped} ordinal columns")
    if skipped:
        print(f"[map_ordinals]      Skipped (not in df): {skipped}")
    return df


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add LogSalePrice column as log-transformed training target."""
    df["LogSalePrice"] = np.log(df["SalePrice"])
    print(f"[add_log_target]    LogSalePrice added")
    print(f"[add_log_target]    Skewness: raw={df['SalePrice'].skew():.4f} "
          f"log={df['LogSalePrice'].skew():.4f}")
    return df


def save_clean(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Save cleaned dataframe to CSV and verify on reload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    # Reload and verify
    string_cols = [c for c in NUMERIC_AS_CATEGORY if c in df.columns]
    saved = pd.read_csv(
        path,
        dtype={col: str for col in string_cols},
        keep_default_na=False,
        na_values=[""]
    )

    assert saved.shape == df.shape, \
        f"Shape mismatch on reload: {saved.shape} vs {df.shape}"
    assert saved.isnull().sum().sum() == 0, \
        "Nulls found in saved file"

    print(f"[save_clean]        Saved to {path}")
    print(f"[save_clean]        Shape: {saved.shape}")
    print(f"[save_clean]        Size : {path.stat().st_size / 1024:.1f} KB")


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    raw_path: Path = RAW_PATH,
    output_path: Path = PROCESSED_PATH
) -> pd.DataFrame:
    """Run full cleaning pipeline end to end."""

    print("=" * 50)
    print("PRICEWISE — CLEANING PIPELINE")
    print("=" * 50)

    df = load_raw(raw_path)
    df = drop_high_missing(df)
    df = fill_absent_nulls(df)
    df = impute_true_nulls(df)
    df = remove_extremes(df)
    df = cap_outliers(df)
    df = fix_types(df)
    df = map_ordinals(df)
    df = add_log_target(df)

    # Final assertion before saving
    assert df.isnull().sum().sum() == 0, "Nulls present before save"
    print(f"\n✅ Zero nulls confirmed")
    print(f"✅ Final shape: {df.shape}")

    save_clean(df, output_path)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()