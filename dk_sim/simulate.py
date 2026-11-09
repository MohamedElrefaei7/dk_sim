"""Synthetic population and randomized promo experiment.

Hidden (latent) columns carry a `_true_` prefix so any leakage is greppable.
All randomness comes from np.random.default_rng(seed); draws happen in a
fixed order regardless of condition, so the same seed yields the same
population and noise across conditions (common random numbers).
"""
import numpy as np
import pandas as pd

from .config import CONDITIONS, SimConfig


def simulate(cfg: SimConfig, seed: int, condition: str) -> pd.DataFrame:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    rng = np.random.default_rng(seed)
    n = cfg.n_users

    # --- population ---
    at_risk = rng.random(n) < cfg.at_risk_prevalence
    engagement = rng.normal(0.0, 1.0, n) + cfg.engagement_shift * at_risk

    deposit_freq = rng.poisson(np.exp(0.5 + 0.4 * engagement))
    session_minutes = rng.lognormal(3.4 + 0.3 * engagement, 0.5)
    bet_escalation = np.clip(
        1.0 + 0.10 * engagement + cfg.escalation_risk_effect * at_risk + rng.normal(0.0, 0.2, n), 0.5, None
    )
    late_night_share = np.where(at_risk, rng.beta(3, 6, n), rng.beta(2, 8, n))

    # --- randomized experiment ---
    treated = rng.random(n) < cfg.treat_prob
    y0 = 20.0 * np.exp(0.5 * engagement) * rng.lognormal(0.0, 0.3, n)

    placebo_noise = rng.normal(0.0, cfg.placebo_effect_sd, n)  # always drawn
    if condition == "placebo":
        tau = cfg.base_effect + placebo_noise
    else:
        tau = cfg.base_effect * np.exp(cfg.effect_engagement_slope * engagement)
        if condition == "loss_chasing":
            tau = tau * np.where(at_risk, cfg.risk_multiplier, 1.0)

    y = y0 + treated * tau + rng.normal(0.0, cfg.outcome_noise_sd, n)

    return pd.DataFrame(
        {
            "deposit_freq": deposit_freq,
            "session_minutes": session_minutes,
            "bet_escalation": bet_escalation,
            "late_night_share": late_night_share,
            "treated": treated.astype(int),
            "y": y,
            "_true_at_risk": at_risk.astype(int),
            "_true_tau": tau,
            "_true_engagement": engagement,
        }
    )
