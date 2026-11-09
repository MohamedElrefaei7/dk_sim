"""T-learner uplift model and top-fraction targeting.

This module only ever sees observable columns: it selects inputs through
FEATURES (or an explicit feature list passed by the caller). It must never
reference latent columns by name; a test greps this file to enforce that.
"""
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURES = ["deposit_freq", "session_minutes", "bet_escalation", "late_night_share"]


def select_features(df: pd.DataFrame, features: Sequence[str] = FEATURES) -> pd.DataFrame:
    return df.loc[:, list(features)]


def split(df: pd.DataFrame, seed: int, holdout_frac: float = 0.5):
    """50/50 train / held-out split, stratified by treatment."""
    return train_test_split(
        df, test_size=holdout_frac, stratify=df["treated"], random_state=seed
    )


def fit_t_learner(train: pd.DataFrame, seed: int, features: Sequence[str] = FEATURES):
    t = train["treated"].to_numpy() == 1
    X = select_features(train, features)
    y = train["y"].to_numpy()
    mu1 = HistGradientBoostingRegressor(random_state=seed).fit(X[t], y[t])
    mu0 = HistGradientBoostingRegressor(random_state=seed).fit(X[~t], y[~t])
    return mu1, mu0


def predict_uplift(models, df: pd.DataFrame, features: Sequence[str] = FEATURES) -> np.ndarray:
    mu1, mu0 = models
    X = select_features(df, features)
    return mu1.predict(X) - mu0.predict(X)


def top_fraction_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask selecting the top `frac` of rows by score (ties broken stably)."""
    k = int(round(frac * len(scores)))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def run_uplift_targeting(
    df: pd.DataFrame,
    seed: int,
    target_frac: float,
    holdout_frac: float = 0.5,
    features: Sequence[str] = FEATURES,
):
    """Fit on train, score held-out. Returns (held-out frame, uplift scores, targeted mask)."""
    train, held = split(df, seed, holdout_frac)
    models = fit_t_learner(train, seed, features)
    uplift = predict_uplift(models, held, features)
    return held, uplift, top_fraction_mask(uplift, target_frac)
