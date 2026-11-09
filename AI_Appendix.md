# AI Appendix: Prompts Used to Create This Simulation

This simulation, its tests, figures, README and the published findings page were produced with an AI coding assistant: **Claude Code**, running **Claude Opus 5.5** (`claude-opus-5-5`), on 8–9 November 2026. The user wrote the study design. The assistant wrote the code, ran it, and reported results.

The prompts below are reproduced verbatim, in order. Text marked *Context* is not part of any prompt. It summarizes what the assistant had just reported, so each short follow-up prompt can be understood on its own.

---

## Prompt 1: Original specification

Pasted into Claude Code, run from the empty `dk_sim/` project directory.

````markdown
# Claude Code prompt: Promotional-targeting simulation (DraftKings ethics paper)

Paste everything below the line into Claude Code, run from an empty project directory.

---

## Context

I'm writing a short ethics case study on allegations that DraftKings used a machine-learning "elasticity" score to target promotions at customers whose spending rises most in response to bonuses. The paper argues the bias comes from the objective: a model trained to find promo-responsive customers will tend to find loss-chasers, without ever seeing a "problem gambler" label.

I need a small, reproducible simulation that tests this claim and produces two figures for the paper. Build it in one pass: code, tests, figures, README. Everything is synthetic. No real gambling data is involved, and the README must say so plainly.

## Files to create

```
dk_sim/
  __init__.py
  config.py        # all parameters in one frozen dataclass
  simulate.py      # synthetic population + randomized promo experiment
  model.py         # T-learner uplift model + top-decile targeting
  analysis.py      # runs all conditions x seeds, writes results + figures
tests/
  test_sim.py
outputs/           # created by analysis.py, not committed by hand
requirements.txt   # numpy, pandas, scikit-learn, matplotlib, pytest
README.md
```

Entry point: `python -m dk_sim.analysis` writes everything into `outputs/`.

## Simulation spec (`simulate.py`, `config.py`)

Defaults live in a frozen dataclass `SimConfig`: `n_users=50_000`, `at_risk_prevalence=0.05`, `n_seeds=10`, `target_frac=0.10`, `risk_multiplier` (per condition), and the noise and effect-size constants below. Every function takes a config and a seed, and all randomness flows from `np.random.default_rng(seed)`. No global RNG anywhere.

**Population.**

- `at_risk ~ Bernoulli(0.05)`. This is a **latent** label. It is used to generate data and to score results, and is never a model feature.
- `engagement ~ Normal(0, 1)`, plus a shift of `+1.0` for at-risk users. At-risk users are more engaged on average, but plenty of heavy recreational users exist in the upper tail. That overlap is deliberate: it's what makes the "we target engagement, not losses" defense testable.
- Observable features (the only model inputs):
  - `deposit_freq = Poisson(exp(0.5 + 0.4 * engagement))`
  - `session_minutes = LogNormal(3.4 + 0.3 * engagement, 0.5)`
  - `bet_escalation`: a loss-chasing index, defined as average stake after a loss divided by average stake after a win. Generate as `1.0 + 0.10 * engagement + 0.35 * at_risk + Normal(0, 0.2)`, clipped to ≥ 0.5. This is the at-risk-specific proxy.
  - `late_night_share = Beta(2, 8)` for non-risk users, `Beta(3, 6)` for at-risk users.

**Randomized promo experiment.** This mirrors the A/B-test data an operator would actually have.

- `treated ~ Bernoulli(0.5)`, independent of everything.
- Baseline weekly revenue (the user's net loss): `y0 = 20 * exp(0.5 * engagement) * LogNormal(0, 0.3)`.
- Individual promo effect `tau` depends on the condition:
  - `placebo`: `tau = 5 + Normal(0, 1)`, independent of all features. This is the pipeline sanity check.
  - `engagement_only`: `tau = 5 * exp(0.4 * engagement)`. At-risk users respond like any equally engaged user, with risk multiplier 1. This is DraftKings' stated position.
  - `loss_chasing`: as engagement_only, but multiplied by `risk_multiplier = 3.0` for at-risk users.
- Observed outcome: `y = y0 + treated * tau + Normal(0, outcome_noise_sd)`.

Return a DataFrame with the features, `treated`, `y`, and the hidden columns `at_risk`, `tau`, `engagement`. Hidden columns get a `_true_` prefix (`_true_at_risk`, `_true_tau`, `_true_engagement`) so leakage is greppable.

## Model (`model.py`)

- `FEATURES = ["deposit_freq", "session_minutes", "bet_escalation", "late_night_share"]` as a module-level constant. The model must select columns via this list only.
- T-learner: two `sklearn.ensemble.HistGradientBoostingRegressor` models, one fit on treated users and one on control. Predicted uplift is `mu1(x) - mu0(x)`. Use sklearn rather than LightGBM to keep dependencies light. Fix `random_state` from the seed.
- Split users 50/50 into train and held-out, stratified by `treated`. Fit on train only. Score and target on held-out only.
- Targeting: take the top `target_frac` of held-out users by predicted uplift.
- Also compute an **oracle** targeting by true `tau` on the same held-out set. This gives the upper bound.

## Analysis and figures (`analysis.py`)

For each condition and each of `n_seeds` seeds, compute on the held-out set:

- `base_rate`: at-risk share of the held-out set.
- `targeted_share`: at-risk share in the model's top decile.
- `representation_ratio = targeted_share / base_rate`.
- `oracle_ratio`: the same ratio, but targeting by true tau.
- `at_risk_uplift_share`: the fraction of total true incremental revenue (sum of `tau` over targeted users) that comes from at-risk users.
- `separability_auc`: the AUC of a logistic regression predicting at_risk from FEATURES (5-fold CV). This shows the label is learnable but not trivially so.

Write `outputs/results.csv` (one row per condition × seed) and `outputs/summary.json` (mean and 2.5/97.5 percentiles across seeds per condition and metric). Print a summary table to stdout.

**Figure 1 (`outputs/fig1_at_risk_share.png`), the paper's main figure.** A grouped bar chart. The x-axis has the three conditions (placebo, engagement-only, loss-chasing). Within each group, show bars for base rate, model-targeted top 10%, and oracle top 10%. Draw error bars from the seed percentile intervals and a dashed horizontal line at the 5% prevalence. Y-axis label: "Share of group that is at-risk".

**Figure 2 (`outputs/fig2_decile_curve.png`), supplementary.** At-risk share by predicted-uplift decile (1 = lowest, 10 = highest) for the engagement-only and loss-chasing conditions. Use mean lines with shaded seed intervals and the same dashed prevalence line.

Both figures: 300 dpi, colorblind-safe palette (e.g. matplotlib `tab10` with distinct hatching or markers so they survive grayscale printing), font size ≥ 11, a title, and no chart junk. Save PNGs only.

## Tests (`tests/test_sim.py`)

Each test should go red when the thing it guards is broken. Use `n_users=20_000` and 3 seeds in tests so the suite runs in well under a minute.

1. **`test_features_exclude_hidden_columns`**: assert `FEATURES` contains no column starting with `_true_`. Also call the model's feature-selection helper on a DataFrame and assert the resulting columns equal `FEATURES` exactly.
2. **`test_leak_would_be_caught`**: in the `loss_chasing` condition, append `_true_at_risk` to the feature list deliberately and confirm the representation ratio jumps above the non-leaked ratio. This proves test 1 guards something that matters. Keep the leaked feature list local to the test.
3. **`test_placebo_targets_at_base_rate`**: the mean representation ratio under `placebo` lies in `[0.8, 1.2]`. If the pipeline leaks risk or the split is wrong, this fails.
4. **`test_ordering_of_conditions`**: the mean representation ratio satisfies placebo < engagement_only < loss_chasing ≤ oracle(loss_chasing) + small tolerance.
5. **`test_monotonic_in_multiplier`**: with `risk_multiplier` in {1.0, 2.0, 3.0}, the mean representation ratio is strictly increasing.
6. **`test_randomization_balance`**: the overall treated share is within 0.01 of 0.5, and the treated share differs by < 0.02 between at-risk and non-risk users.
7. **`test_prevalence`**: the at-risk share is within 0.005 of 0.05 at n=20k.
8. **`test_not_trivially_separable`**: `separability_auc` is between 0.65 and 0.92. If at-risk users were perfectly separable, the result would be trivial; if they were inseparable, the proxies would be meaningless.
9. **`test_determinism`**: two runs with the same seed produce identical results DataFrames (`pd.testing.assert_frame_equal`).

## Non-negotiables

- **Never change a test threshold to make a test pass.** If a substantive test fails (3, 4, 5, or 8), you may adjust only `outcome_noise_sd` or the `engagement` shift. Record the exact change and the reason in the README under "Parameter changes". If the ordering still fails after reasonable tuning, stop and report the numbers instead of forcing them.
- Hidden `_true_` columns may be read only in `simulate.py`, the metric functions in `analysis.py`, and tests. Add a grep-based check in the test file that fails if `model.py` contains the string `_true_`.
- Every parameter is illustrative, not estimated from DraftKings data. The README must say this in its first paragraph.

## README.md

Write about 200 words:

- What this is: a synthetic simulation supporting an ethics case study, using no real data.
- How to reproduce: `pip install -r requirements.txt && pytest && python -m dk_sim.analysis`.
- The three conditions and what each represents.
- A results table pasted from `summary.json`.
- The two figures embedded.
- A "Limitations" paragraph covering three points: the effect sizes are assumptions; the result depends on at-risk users actually responding more strongly to promotions, which is the empirical question the paper flags; and the engagement-only condition shows how much concentration happens even under the company's stated policy.

## Live verification (do this last and report back)

1. Run `pytest -q` and paste the summary line.
2. Run `python -m dk_sim.analysis` and paste the printed summary table and the wall-clock time.
3. Open both PNGs and describe what each shows in one sentence. Check that every label is legible and that the dashed prevalence line appears.
4. Confirm these four facts from `summary.json`:
   - the placebo ratio is near 1;
   - engagement_only is above 1;
   - loss_chasing is above engagement_only;
   - the model ratio is below the oracle ratio.

   If any fails, say which, and do not paper over it.
5. List any parameter changes you made and why.
````

---

## Prompt 2

*Context:* At the spec defaults, two substantive tests failed: the placebo representation ratio was about 2.8 (target 0.8–1.2), and separability AUC was about 0.94 (cap 0.92). Sweeping the two permitted knobs did not fix the placebo result, and the assistant stopped as instructed. It proposed three options: (1) change the estimator, (2) keep the T-learner and treat the placebo result as a recorded finding, or (3) weaken the at-risk proxy.

> Let's go with option one.

---

## Prompt 3

*Context:* None of the estimators tried brought the placebo ratio into range reliably at paper scale: regularized T-learner, X-learner, cross-fitted DR-learner, bagged lower-confidence-bound ranking. One heavily regularized T-learner passed on the three test seeds but failed at 50,000 users (1.50), so the assistant did not adopt it. The assistant recommended option 2. It also noted that the AUC failure was unaffected by the estimator: it could be fixed either by setting the engagement shift to 0 or by weakening the `bet_escalation` proxy.

> Let's go with option 2. You choose the path for the AUC that keeps the simulation in the nature as described when we started.

*Outcome:* The engagement shift was kept at 1.0 to preserve the deliberate engagement overlap. The `bet_escalation` at-risk coefficient was lowered from 0.35 to 0.25 (AUC ≈ 0.905). Test 3 was reframed to check the oracle placebo ratio and the U-shaped placebo decile curve. See README "Parameter changes".

---

## Prompt 4

> Summarize the findings.

---

## Prompt 5

*Context:* The assistant offered to turn the summary into a shareable page with the two figures.

> Yes, do that.

---

## Prompt 6

> Add the prompts used to create this simulation in their own AI_Appendix.md markdown file.
