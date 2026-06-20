# Pricewise — EDA Summary
## Sprint 2 Findings & Sprint 3 Implications

---

## Dataset
- Rows: 1458
- Features: 76 (excluding Id and target)
- Numeric features: 53
- Categorical features: 22

---

## Target Variable
- `SalePrice` is right-skewed (skewness = 1.8813)
- Log transformation reduces skewness to 0.1216
- Training target: `LogSalePrice`
- Predictions converted back with `np.exp()` for display

---

## Top 10 Correlated Features with LogSalePrice
- OverallQual: r = 0.8214
- GrLivArea: r = 0.7255
- GarageCars: r = 0.6846
- ExterQual: r = 0.6822
- KitchenQual: r = 0.6700
- GarageArea: r = 0.6631
- TotalBsmtSF: r = 0.6473
- 1stFlrSF: r = 0.6208
- BsmtQual: r = 0.6169
- GarageFinish: r = 0.6056

---

## Key Findings

### Numeric Features
- 18 features are highly skewed (|skewness| > 1)
- 10 features have near-zero variance (>95% single value)
- Area/SF features are zero-inflated — many houses lack these features

### Categorical Features
- 9 categorical columns have a dominant value (>80% of rows)
- 2 high-cardinality columns need special encoding

### Neighbourhood
- Most expensive: NridgHt (median $315,000)
- Most affordable: MeadowV (median $88,000)
- Price ratio top vs bottom: 3.6x
- Tier thresholds: Low < $139,340, Mid $139,340–$196,208, High > $196,208

### Feature Relationships
- OverallQual: strongest single predictor — near-monotonic with price
- GrLivArea: strong linear relationship (r = 0.7255)
- YearBuilt: clear upward trend — newer houses command premium
- 5.6% of houses have no garage — zero-inflation in GarageArea
- 0.0% of houses have no basement — zero-inflation in TotalBsmtSF

---

## Sprint 3 Plan

### Features to Engineer
- total_sf = TotalBsmtSF + GrLivArea
- total_bathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
- total_porch_sf = sum of all porch area columns
- house_age = YrSold - YearBuilt
- remod_age = YrSold - YearRemodAdd
- has_remodelled = 1 if YearRemodAdd != YearBuilt else 0
- qual_x_sf = OverallQual * GrLivArea
- overall_score = OverallQual * OverallCond
- has_garage = 1 if GarageArea > 0 else 0
- has_basement = 1 if TotalBsmtSF > 0 else 0
- has_pool = 1 if PoolArea > 0 else 0
- neighbourhood_tier = Low / Mid / High

### Features to Drop
- Utilities: Near-zero variance — one value >95% of rows
- BsmtFinSF2: Near-zero variance — one value >95% of rows
- LowQualFinSF: Near-zero variance — one value >95% of rows
- BsmtHalfBath: Near-zero variance — one value >95% of rows
- KitchenAbvGr: Near-zero variance — one value >95% of rows
- EnclosedPorch: Near-zero variance — one value >95% of rows
- 3SsnPorch: Near-zero variance — one value >95% of rows
- ScreenPorch: Near-zero variance — one value >95% of rows
- PoolArea: Near-zero variance — one value >95% of rows
- MiscVal: Near-zero variance — one value >95% of rows
- Street: Dominant category >90% — near-constant signal
- Condition2: Dominant category >90% — near-constant signal
- RoofMatl: Dominant category >90% — near-constant signal
- Heating: Dominant category >90% — near-constant signal
- Electrical: Dominant category >90% — near-constant signal
- Id: Row identifier — not a predictive feature

### Features for log1p Transform
- Utilities (skewness = -38.1838)
- Functional (skewness = -4.9085)
- LandSlope (skewness = -4.8100)
- GarageYrBlt (skewness = -3.8664)
- BsmtCond (skewness = -3.6981)
- CentralAir (skewness = -3.5274)
- LandContour (skewness = -3.4791)
- GarageCond (skewness = -3.3290)
- PavedDrive (skewness = -3.3061)
- BsmtFinType2 (skewness = 3.2942)
- GarageQual (skewness = -3.2258)
- ExterCond (skewness = 1.3955)
- BsmtQual (skewness = -1.2946)
- MasVnrArea (skewness = 1.2899)
- LotShape (skewness = -1.2865)
- BsmtExposure (skewness = 1.2205)
- OpenPorchSF (skewness = 1.1396)
- WoodDeckSF (skewness = 1.0880)
