## Sprint 1 — Data Cleaning

### Overview

Sprint 1 transforms the raw Ames Housing dataset into a clean, analysis-ready file that all downstream sprints read from. Every cleaning decision is documented with reasoning in `notebooks/01_cleaning.ipynb`. The production pipeline is available as a reusable script in `src/clean.py`.

---

### Dataset

**Source:** [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle)  
**Raw shape:** 1,460 rows × 81 columns (79 features + `Id` + `SalePrice`)  
**Clean shape:** 1,458 rows × 78 columns

---

### Pipeline

```
data/raw/train.csv
        ↓
drop_high_missing()     — removed 4 columns with >80% missing
fill_absent_nulls()     — 11 categorical cols → "No", 10 numeric cols → 0
impute_true_nulls()     — 8 categorical cols → mode, LotFrontage → neighbourhood median
remove_extremes()       — 2 GrLivArea outliers removed (Dean De Cock, 2011)
cap_outliers()          — IQR Winsorization on 27 numeric columns
fix_types()             — MSSubClass, MoSold, YrSold → string
map_ordinals()          — 20 quality/condition columns → ordered integers
add_log_target()        — LogSalePrice added as training target
        ↓
data/processed/clean_data.csv
```

---

### Missing Value Treatment

19 columns had missing values across the dataset. The critical distinction was classifying nulls as either **absent** (the feature doesn't exist for that house) or **truly missing** (the value was not recorded).

| Category                  | Columns | Treatment                        |
| ------------------------- | ------- | -------------------------------- |
| Drop >80% missing         | 4       | Removed entirely                 |
| Absent categorical        | 11      | Filled with `"No"`               |
| Absent numeric            | 10      | Filled with `0`                  |
| Truly missing categorical | 8       | Filled with mode                 |
| LotFrontage               | 1       | Filled with neighbourhood median |

> Columns dropped due to >80% missing: `PoolQC`, `MiscFeature`, `Alley`, `Fence`

> `LotFrontage` was imputed using the median of each house's neighbourhood rather than the global median — houses in the same neighbourhood tend to have similar lot frontage.

---

### Outlier Treatment

| Treatment            | Method                   | Impact                                |
| -------------------- | ------------------------ | ------------------------------------- |
| `GrLivArea` extremes | Manual removal           | 2 rows dropped                        |
| Numeric outliers     | IQR Winsorization (1.5×) | 1,448 values capped across 27 columns |

The two `GrLivArea` outliers (>4,000 sq ft, <$200,000) were identified in the original Ames Housing paper by Dean De Cock as partial interest sales — not standard market transactions. Winsorization was chosen over row-dropping for all other outliers to preserve sample size.

---

### Type Casting & Ordinal Encoding

| Step                | Action                                               | Columns |
| ------------------- | ---------------------------------------------------- | ------- |
| Numeric → category  | `MSSubClass`, `MoSold`, `YrSold` converted to string | 3       |
| Whitespace strip    | All categorical columns stripped                     | 42      |
| Ordinal encoding    | Quality/condition/ordered scales → integers          | 20      |
| Nominal categorical | Left as strings — one-hot encoding in Sprint 3       | 22      |

Quality and condition columns were mapped on a consistent scale: `No → 0, Po → 1, Fa → 2, TA → 3, Gd → 4, Ex → 5`.

---

### Target Variable

`SalePrice` is right-skewed (skewness: **1.8813**). Log transformation produces a near-normal distribution suitable for regression modeling.

| Metric   | SalePrice | LogSalePrice |
| -------- | --------- | ------------ |
| Min      | $34,900   | 10.4602      |
| Max      | $755,000  | 13.5346      |
| Mean     | $180,933  | 12.0240      |
| Skewness | 1.8813    | 0.1216       |
| Nulls    | 0         | 0            |

`LogSalePrice` is the training target in Sprint 4. `SalePrice` is retained for human-readable predictions in the Streamlit app.

---

### Output

| Check       | Result                          |
| ----------- | ------------------------------- |
| Null values | 0                               |
| Rows        | 1,458                           |
| Columns     | 78                              |
| File        | `data/processed/clean_data.csv` |
| File size   | 451.2 KB                        |

---

### How to Run

```bash
# Run full cleaning pipeline
python src/clean.py

# Run tests
pytest tests/test_clean.py -v
```

---

### Files

| File                            | Description                      |
| ------------------------------- | -------------------------------- |
| `notebooks/01_cleaning.ipynb`   | Full documented cleaning process |
| `src/clean.py`                  | Production cleaning pipeline     |
| `tests/test_clean.py`           | 34 pytest tests — all passing    |
| `data/processed/clean_data.csv` | Output artifact                  |

---

## Sprint 2 — EDA & Visualisation

### Overview

Sprint 2 explores the cleaned Ames Housing dataset to build a deep understanding of feature distributions, correlations, and relationships with `SalePrice`. Every finding is documented with a justified Sprint 3 implication — the goal was to finish EDA with a concrete feature engineering plan, not just plots.

---

### Dataset Explored

**Input:** `data/processed/clean_data.csv`
**Shape:** 1,458 rows × 78 columns
**Numeric features:** 53
**Categorical features:** 22

---

### Target Variable

`SalePrice` is right-skewed (skewness = 1.8813). Log transformation reduces skewness to 0.1216 — near-normal and confirmed by QQ plot.

| Metric   | SalePrice | LogSalePrice |
| -------- | --------- | ------------ |
| Min      | $34,900   | 10.4602      |
| Max      | $755,000  | 13.5346      |
| Mean     | $180,933  | 12.0240      |
| Median   | $163,000  | 12.0012      |
| Skewness | 1.8813    | 0.1216       |

`LogSalePrice` is the training target in Sprint 4. Predictions are converted back with `np.exp()` for display in the Streamlit app.

---

### Numeric Feature Distributions

| Category                               | Count |
| -------------------------------------- | ----- |
| Highly skewed (\|skewness\| > 1)       | 18    |
| Moderately skewed (0.5–1)              | 13    |
| Approximately normal                   | 22    |
| Near-zero variance (>95% single value) | 10    |

Area and SF features are zero-inflated — many houses have no garage, porch, or pool, creating a spike at 0. These features need binary flags in Sprint 3.

---

### Top 10 Correlated Features with LogSalePrice

| Feature      | Correlation | Direction |
| ------------ | ----------- | --------- |
| OverallQual  | 0.8192      | Positive  |
| GrLivArea    | 0.7330      | Positive  |
| GarageCars   | 0.6885      | Positive  |
| GarageArea   | 0.6286      | Positive  |
| TotalBsmtSF  | 0.6326      | Positive  |
| 1stFlrSF     | 0.6215      | Positive  |
| FullBath     | 0.5946      | Positive  |
| TotRmsAbvGrd | 0.5698      | Positive  |
| YearBuilt    | 0.5870      | Positive  |
| YearRemodAdd | 0.5653      | Positive  |

---

### Key Findings by Section

**Neighbourhood:** StoneBr is the most expensive neighbourhood (median $278,000), MeadowV the most affordable (median $88,000) — a 3.2× price ratio. Neighbourhoods are grouped into Low / Mid / High tiers for Sprint 3.

**OverallQual:** Near-monotonic relationship with price — the strongest single predictor. Each quality grade step adds approximately $26,000 to mean price.

**GrLivArea:** Strong linear relationship with LogSalePrice (r = 0.7330). Coloured scatter shows quality and size interact — high quality + large area produces the highest prices.

**GarageArea:** 5.6% of houses have no garage. Correlation is stronger when restricted to houses with garages (r = 0.6631 vs 0.6286 overall) — zero-inflation masks the true signal.

**YearBuilt:** Clear upward trend (r = 0.5870). Post-1990 prices rise sharply. Sprint 3 will convert to `house_age = YrSold - YearBuilt` for a cleaner signal.

---

### Sprint 3 Feature Engineering Plan

| Category      | Features                                        |
| ------------- | ----------------------------------------------- |
| Size          | `total_sf`, `total_bathrooms`, `total_porch_sf` |
| Age           | `house_age`, `remod_age`, `has_remodelled`      |
| Interaction   | `qual_x_sf`, `overall_score`                    |
| Binary flags  | `has_garage`, `has_basement`, `has_pool`        |
| Neighbourhood | `neighbourhood_tier` (Low / Mid / High)         |

**Features to drop in Sprint 3:** 16 columns flagged — 10 near-zero variance numeric features and 6 dominant categorical columns (>90% single value).

**Features for log1p transform:** 18 highly skewed area/SF features.

Full plan saved to `reports/eda_summary.md`.

---

### Figures Produced

15 plots saved to `reports/figures/` covering target distribution, feature skewness, correlation heatmap, scatter plots, categorical bar charts, neighbourhood ranking and boxplot, and key feature relationship plots.

---

### Files

| File                     | Description                       |
| ------------------------ | --------------------------------- |
| `notebooks/02_eda.ipynb` | Full EDA notebook — 8 sections    |
| `reports/eda_summary.md` | Sprint 3 feature engineering plan |
| `reports/figures/`       | 15 saved PNG plots                |

---

## Sprint 3 — Feature Engineering

### Overview

Sprint 3 transforms the cleaned dataset from Sprint 1 into a fully
engineered, model-ready feature set. Every feature is justified by
EDA findings from Sprint 2. The output is `data/processed/featured_data.csv`
— the direct input to Sprint 4 modeling.

---

### Input → Output

|                     | Value                              |
| ------------------- | ---------------------------------- |
| Input               | `data/processed/clean_data.csv`    |
| Input shape         | 1,458 × 78                         |
| Output              | `data/processed/featured_data.csv` |
| Output shape        | 1,458 × 172                        |
| Features engineered | 11                                 |
| Columns dropped     | 16                                 |

---

### Pipeline

```
data/processed/clean_data.csv
        ↓
drop_low_signal()          — 10 near-zero variance + 5 dominant categorical + Id
add_size_features()        — total_sf, total_bathrooms, total_porch_sf
add_age_features()         — house_age, remod_age, has_remodelled
add_interaction_features() — qual_x_sf, overall_score
add_binary_flags()         — has_garage, has_basement
add_neighbourhood_tier()   — Low / Mid / High
apply_log1p_transform()    — 5 highly skewed area/SF features
encode_categoricals()      — 17 nominal categoricals → 117 dummy columns
        ↓
data/processed/featured_data.csv
```

---

### Features Engineered

| Category    | Feature              | Formula                                | Correlation         |
| ----------- | -------------------- | -------------------------------------- | ------------------- |
| Size        | `total_sf`           | TotalBsmtSF + GrLivArea                | 0.8236              |
| Size        | `total_bathrooms`    | FullBath + 0.5×HalfBath + BsmtFullBath | 0.6722              |
| Size        | `total_porch_sf`     | Sum of all porch SF columns            | 0.3943              |
| Age         | `house_age`          | YrSold − YearBuilt                     | −0.5787             |
| Age         | `remod_age`          | YrSold − YearRemodAdd                  | −0.5685             |
| Age         | `has_remodelled`     | YearRemodAdd ≠ YearBuilt               | −0.0741             |
| Interaction | `qual_x_sf`          | OverallQual × GrLivArea                | 0.8490              |
| Interaction | `overall_score`      | OverallQual × OverallCond              | 0.6080              |
| Flag        | `has_garage`         | GarageArea > 0                         | 0.3230              |
| Flag        | `has_basement`       | TotalBsmtSF > 0                        | —                   |
| Location    | `neighbourhood_tier` | Low / Mid / High                       | 0.5867 (High dummy) |

> `has_pool` was not engineered — `PoolArea` was dropped as near-zero
> variance (only 13 of 1,458 houses have pools).

---

### Columns Dropped

| Category                                       | Count  |
| ---------------------------------------------- | ------ |
| Near-zero variance numeric (>95% single value) | 10     |
| Dominant categorical (>90% single category)    | 5      |
| `Id` + `Neighborhood`                          | 2      |
| **Total**                                      | **16** |

Shape went from 1,458 × 78 to 1,458 × 62 before feature engineering.

---

### Key Improvements vs Baseline

| Engineered           | Correlation | Raw                          | Correlation |
| -------------------- | ----------- | ---------------------------- | ----------- |
| `qual_x_sf`          | 0.8490      | `OverallQual`                | 0.8214      |
| `total_sf`           | 0.8236      | `GrLivArea`                  | 0.7255      |
| `total_bathrooms`    | 0.6722      | `FullBath`                   | 0.5946      |
| `neighbourhood_tier` | 0.5867      | Best raw neighbourhood dummy | 0.3518      |

`qual_x_sf` is the strongest feature in the final dataset (r = 0.8490),
outperforming both its parent features `OverallQual` and `GrLivArea`.

---

### Transformations Applied

**log1p transform** applied to 5 highly skewed area/SF features
(|skewness| > 1). `log1p` used over `log` to handle zero values
cleanly — `log1p(0) = 0`.

**One-hot encoding** applied to 17 nominal categorical columns with
`drop_first=True`, producing 117 dummy columns.

---

### Final Dataset

| Metric              | Value                    |
| ------------------- | ------------------------ |
| Rows                | 1,458                    |
| Final feature count | 170                      |
| Top feature         | `qual_x_sf` (r = 0.8490) |
| Null values         | 0                        |
| Object columns      | 0                        |
| File size           | 683.4 KB                 |

---

### How to Run

```bash
python src/features.py
```

---

### Files

| File                               | Description                                     |
| ---------------------------------- | ----------------------------------------------- |
| `notebooks/03_modelling.ipynb`     | Full feature engineering notebook — 10 sections |
| `src/features.py`                  | Production feature engineering pipeline         |
| `data/processed/featured_data.csv` | Output artifact                                 |
