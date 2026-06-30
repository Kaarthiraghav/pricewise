"""
src/train.py
Pricewise — Production Training Pipeline
Sprint 4

Reads:  data/processed/featured_data.csv
Writes: models/xgboost_model.pkl
        models/shap_explainer.pkl
        models/feature_columns.pkl

Usage:
    python src/train.py

    or as a module:
    from src.train import run_training_pipeline
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import optuna
import shap

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

FEATURED_PATH = Path("data/processed/featured_data.csv")
MODELS_PATH   = Path("models")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TARGET      = "LogSalePrice"
TARGET_RAW  = "SalePrice"
RANDOM_STATE = 42
N_TRIALS    = 50

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def load_featured(path: Path = FEATURED_PATH) -> pd.DataFrame:
    """Load featured_data.csv from Sprint 3."""
    df = pd.read_csv(path)
    assert df.isnull().sum().sum() == 0, "Nulls found in featured data"
    assert TARGET in df.columns, "LogSalePrice missing"
    assert TARGET_RAW in df.columns, "SalePrice missing"

    print(f"[load_featured]     Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Split into train (70%), validation (15%), test (15%)."""
    X = df.drop(columns=[TARGET, TARGET_RAW])
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )

    val_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # Leakage check
    train_idx = set(X_train.index)
    val_idx   = set(X_val.index)
    test_idx  = set(X_test.index)
    assert len(train_idx & val_idx) == 0, "Train/val overlap"
    assert len(train_idx & test_idx) == 0, "Train/test overlap"
    assert len(val_idx & test_idx) == 0, "Val/test overlap"

    print(f"[split_data]        Train: {len(X_train)}  "
          f"Val: {len(X_val)}  Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y, scaled: bool = False) -> dict:
    """Evaluate a model — returns RMSE/MAE in dollars, R² and RMSLE on log scale."""
    y_pred_log = model.predict(X)
    y_pred     = np.exp(y_pred_log)
    y_actual   = np.exp(y)

    rmse  = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae   = mean_absolute_error(y_actual, y_pred)
    r2    = r2_score(y, y_pred_log)
    rmsle = np.sqrt(mean_squared_error(y, y_pred_log))

    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsle": rmsle}


def train_baseline_models(X_train, y_train, X_val, y_val) -> dict:
    """Train Ridge, Lasso, Random Forest baselines."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_scaled, y_train)

    lasso = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    results = {
        "ridge": {"model": ridge, "metrics": evaluate(ridge, X_val_scaled, y_val)},
        "lasso": {"model": lasso, "metrics": evaluate(lasso, X_val_scaled, y_val)},
        "rf"   : {"model": rf,    "metrics": evaluate(rf, X_val, y_val)},
        "scaler": scaler,
    }

    print(f"[train_baselines]   Ridge RMSE: ${results['ridge']['metrics']['rmse']:,.0f}  "
          f"Lasso RMSE: ${results['lasso']['metrics']['rmse']:,.0f}  "
          f"RF RMSE: ${results['rf']['metrics']['rmse']:,.0f}")

    return results


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = N_TRIALS) -> dict:
    """Run Optuna hyperparameter tuning for XGBoost."""

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth"       : trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma"           : trial.suggest_float("gamma", 0.0, 5.0),
            "random_state"    : RANDOM_STATE,
            "n_jobs"          : -1,
            "verbosity"       : 0,
            "eval_metric"     : "rmse",
            "early_stopping_rounds": 20,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, y_pred))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[tune_xgboost]      Best validation RMSLE: {study.best_value:.6f}")
    print(f"[tune_xgboost]      Best params: {study.best_params}")

    return study.best_params


def train_final_model(X_trainval, y_trainval, params: dict) -> XGBRegressor:
    """Train final XGBoost on combined train + validation with tuned params."""
    final_params = dict(params)
    final_params.update({
        "random_state": RANDOM_STATE,
        "n_jobs"      : -1,
        "verbosity"   : 0,
    })

    model = XGBRegressor(**final_params)
    model.fit(X_trainval, y_trainval)

    print(f"[train_final_model] Trained on {len(X_trainval)} rows")
    return model


def fit_explainer(model, X_train) -> "shap.TreeExplainer":
    """Fit a SHAP TreeExplainer on the trained model."""
    explainer = shap.TreeExplainer(model)
    print(f"[fit_explainer]     SHAP TreeExplainer fitted")
    return explainer


def save_artifacts(model, explainer, feature_columns: list,
                   models_path: Path = MODELS_PATH) -> None:
    """Save model, explainer, and feature column order."""
    models_path.mkdir(parents=True, exist_ok=True)

    model_path     = models_path / "xgboost_model.pkl"
    explainer_path = models_path / "shap_explainer.pkl"
    columns_path   = models_path / "feature_columns.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)
    with open(columns_path, "wb") as f:
        pickle.dump(feature_columns, f)

    # Verify reload
    with open(model_path, "rb") as f:
        model_reloaded = pickle.load(f)
    test_pred_original = model.predict(np.zeros((1, len(feature_columns))))
    test_pred_reloaded = model_reloaded.predict(np.zeros((1, len(feature_columns))))
    assert np.allclose(test_pred_original, test_pred_reloaded), \
        "Reloaded model gives different predictions"

    print(f"[save_artifacts]    Model     : {model_path} "
          f"({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"[save_artifacts]    Explainer : {explainer_path} "
          f"({explainer_path.stat().st_size / 1024:.1f} KB)")
    print(f"[save_artifacts]    Columns   : {columns_path} "
          f"({columns_path.stat().st_size / 1024:.1f} KB)")


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_training_pipeline(
    featured_path: Path = FEATURED_PATH,
    models_path: Path = MODELS_PATH,
    n_trials: int = N_TRIALS
) -> dict:
    """Run full training pipeline end to end."""

    print("=" * 50)
    print("PRICEWISE — TRAINING PIPELINE")
    print("=" * 50)

    df = load_featured(featured_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    baseline_results = train_baseline_models(X_train, y_train, X_val, y_val)

    best_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials)

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_model = train_final_model(X_trainval, y_trainval, best_params)

    test_metrics = evaluate(final_model, X_test, y_test)
    print(f"\n[FINAL TEST METRICS]")
    print(f"  RMSE  : ${test_metrics['rmse']:,.0f}")
    print(f"  MAE   : ${test_metrics['mae']:,.0f}")
    print(f"  R²    : {test_metrics['r2']:.4f}")

    explainer = fit_explainer(final_model, X_train)

    save_artifacts(final_model, explainer, X_train.columns.tolist(), models_path)

    print("\n" + "=" * 50)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 50)

    return {
        "model"            : final_model,
        "explainer"        : explainer,
        "test_metrics"     : test_metrics,
        "baseline_results" : baseline_results,
        "best_params"      : best_params,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_training_pipeline()