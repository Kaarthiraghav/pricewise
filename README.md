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

| Category | Columns | Treatment |
|---|---|---|
| Drop >80% missing | 4 | Removed entirely |
| Absent categorical | 11 | Filled with `"No"` |
| Absent numeric | 10 | Filled with `0` |
| Truly missing categorical | 8 | Filled with mode |
| LotFrontage | 1 | Filled with neighbourhood median |

> Columns dropped due to >80% missing: `PoolQC`, `MiscFeature`, `Alley`, `Fence`

> `LotFrontage` was imputed using the median of each house's neighbourhood rather than the global median — houses in the same neighbourhood tend to have similar lot frontage.

---

### Outlier Treatment

| Treatment | Method | Impact |
|---|---|---|
| `GrLivArea` extremes | Manual removal | 2 rows dropped |
| Numeric outliers | IQR Winsorization (1.5×) | 1,448 values capped across 27 columns |

The two `GrLivArea` outliers (>4,000 sq ft, <$200,000) were identified in the original Ames Housing paper by Dean De Cock as partial interest sales — not standard market transactions. Winsorization was chosen over row-dropping for all other outliers to preserve sample size.

---

### Type Casting & Ordinal Encoding

| Step | Action | Columns |
|---|---|---|
| Numeric → category | `MSSubClass`, `MoSold`, `YrSold` converted to string | 3 |
| Whitespace strip | All categorical columns stripped | 42 |
| Ordinal encoding | Quality/condition/ordered scales → integers | 20 |
| Nominal categorical | Left as strings — one-hot encoding in Sprint 3 | 22 |

Quality and condition columns were mapped on a consistent scale: `No → 0, Po → 1, Fa → 2, TA → 3, Gd → 4, Ex → 5`.

---

### Target Variable

`SalePrice` is right-skewed (skewness: **1.8813**). Log transformation produces a near-normal distribution suitable for regression modeling.

| Metric | SalePrice | LogSalePrice |
|---|---|---|
| Min | $34,900 | 10.4602 |
| Max | $755,000 | 13.5346 |
| Mean | $180,933 | 12.0240 |
| Skewness | 1.8813 | 0.1216 |
| Nulls | 0 | 0 |

`LogSalePrice` is the training target in Sprint 4. `SalePrice` is retained for human-readable predictions in the Streamlit app.

---

### Output

| Check | Result |
|---|---|
| Null values | 0 |
| Rows | 1,458 |
| Columns | 78 |
| File | `data/processed/clean_data.csv` |
| File size | 451.2 KB |

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

| File | Description |
|---|---|
| `notebooks/01_cleaning.ipynb` | Full documented cleaning process |
| `src/clean.py` | Production cleaning pipeline |
| `tests/test_clean.py` | 34 pytest tests — all passing |
| `data/processed/clean_data.csv` | Output artifact |