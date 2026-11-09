"""All simulation parameters in one place. Every value is illustrative,
not estimated from DraftKings (or any real operator's) data."""
from dataclasses import dataclass

CONDITIONS = ("placebo", "engagement_only", "loss_chasing")


@dataclass(frozen=True)
class SimConfig:
    # Population / experiment size
    n_users: int = 50_000
    at_risk_prevalence: float = 0.05
    n_seeds: int = 10
    target_frac: float = 0.10
    treat_prob: float = 0.5
    holdout_frac: float = 0.5

    # Latent engagement
    engagement_shift: float = 1.0  # extra mean engagement for at-risk users

    # Loss-chasing proxy: at-risk shift in bet_escalation. Spec value was 0.35;
    # lowered to 0.25 so at-risk users are learnable but not near-trivially
    # separable (logistic AUC ~0.94 -> ~0.905). See README "Parameter changes".
    escalation_risk_effect: float = 0.25

    # Promo effect (tau)
    base_effect: float = 5.0
    effect_engagement_slope: float = 0.4
    placebo_effect_sd: float = 1.0
    risk_multiplier: float = 3.0  # applied to at-risk users in loss_chasing only

    # Outcome noise
    outcome_noise_sd: float = 5.0
