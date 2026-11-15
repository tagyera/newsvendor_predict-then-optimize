"""
Step 2: synthetic daily demand where we KNOW the true distribution.

Model (everything on the log scale, so demand is positive and right-skewed):

    log_mean_t  = log(base) + trend*t + weekly[dow] + yearly seasonality
                  + promo_effect*promo_t + holiday_effect*holiday_t + level_t
    sigma_t     = sigma_base + sigma_weekend*weekend_t + sigma_promo*promo_t
    demand_t    = round( exp( log_mean_t + sigma_t * Normal(0,1) ) )

So, given the features, demand is lognormal with parameters (log_mean_t, sigma_t).
That gives closed-form TRUE conditional mean and quantiles, which we use later
as a benchmark ("how close do my models get to the best possible decision?").

level_t is a slowly drifting hidden state (AR(1)). Models cannot see it directly,
but yesterday's/last week's demand partly reveals it. That is why lag features
(Step 4) have something real to learn.

Run from the repo root:
    python -m project1_newsvendor.src.data_synthetic
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .utils import SEED, get_rng

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


@dataclass
class DemandConfig:
    start: str = "2020-01-01"
    n_days: int = 1460  # 4 years of daily data
    base_level: float = 50.0  # typical demand before effects
    trend_per_day: float = 0.0002  # log-scale drift, roughly +7.5% per year
    # Monday..Sunday effects on the log scale (weekend is busiest)
    weekly_effect: tuple = (-0.10, -0.05, 0.00, 0.00, 0.10, 0.30, 0.20)
    yearly_amplitude: float = 0.15  # seasonal swing on the log scale
    promo_start_prob: float = 0.04  # chance a promotion starts on a given day
    promo_length: int = 3  # promotions last 3 days
    promo_effect: float = 0.35  # promo lifts demand by about 40%
    holiday_effect: float = 0.25
    holidays: tuple = ((1, 1), (5, 1), (12, 25), (12, 26))  # (month, day), synthetic
    # noise: its SPREAD depends on features (heteroscedasticity)
    sigma_base: float = 0.20
    sigma_weekend: float = 0.10
    sigma_promo: float = 0.15
    # hidden slowly-moving level (AR(1))
    level_phi: float = 0.95
    level_sd: float = 0.03


def generate_demand(cfg: DemandConfig | None = None, seed: int = SEED):
    """Return (observed, truth).

    observed: what a forecaster is allowed to see (date, demand, calendar, promo, holiday).
    truth:    the hidden generating parameters. NEVER use these as model features.
    """
    cfg = cfg or DemandConfig()
    rng = get_rng(seed)
    n = cfg.n_days

    dates = pd.date_range(cfg.start, periods=n, freq="D")
    t = np.arange(n)
    dow = dates.dayofweek.to_numpy()  # 0 = Monday ... 6 = Sunday
    month = dates.month.to_numpy()
    is_weekend = (dow >= 5).astype(int)

    # promotions: random start days, each lasting promo_length days
    promo = np.zeros(n, dtype=int)
    for i in np.flatnonzero(rng.random(n) < cfg.promo_start_prob):
        promo[i : i + cfg.promo_length] = 1

    # holidays: fixed calendar dates
    hol = set(cfg.holidays)
    holiday = np.array(
        [(m, d) in hol for m, d in zip(dates.month, dates.day)], dtype=int
    )

    # hidden AR(1) level, started from its stationary distribution
    level = np.zeros(n)
    shocks = rng.normal(0.0, cfg.level_sd, n)
    level[0] = rng.normal(0.0, cfg.level_sd / np.sqrt(1 - cfg.level_phi**2))
    for i in range(1, n):
        level[i] = cfg.level_phi * level[i - 1] + shocks[i]

    yearly = cfg.yearly_amplitude * np.sin(
        2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    )
    log_mean = (
        np.log(cfg.base_level)
        + cfg.trend_per_day * t
        + np.asarray(cfg.weekly_effect)[dow]
        + yearly
        + cfg.promo_effect * promo
        + cfg.holiday_effect * holiday
        + level
    )
    sigma = cfg.sigma_base + cfg.sigma_weekend * is_weekend + cfg.sigma_promo * promo

    demand = np.rint(np.exp(log_mean + sigma * rng.standard_normal(n))).astype(int)

    observed = pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
            "dow": dow,
            "month": month,
            "is_weekend": is_weekend,
            "promo": promo,
            "holiday": holiday,
        }
    )
    truth = pd.DataFrame(
        {
            "date": dates,
            "true_log_mean": log_mean,
            "true_sigma": sigma,
            "true_level": level,
        }
    )
    return observed, truth


# ---------- closed-form truths for a lognormal ----------


def true_quantile(truth: pd.DataFrame, tau: float) -> np.ndarray:
    """True conditional tau-quantile of demand: exp(log_mean + sigma * z_tau)."""
    return np.exp(
        truth["true_log_mean"] + truth["true_sigma"] * norm.ppf(tau)
    ).to_numpy()


def true_mean(truth: pd.DataFrame) -> np.ndarray:
    """True conditional mean of demand: exp(log_mean + sigma^2 / 2)."""
    return np.exp(truth["true_log_mean"] + truth["true_sigma"] ** 2 / 2).to_numpy()


def save_synthetic(observed, truth, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    observed.to_csv(out_dir / "synthetic_demand.csv", index=False)
    truth.to_csv(out_dir / "synthetic_truth.csv", index=False)


def load_synthetic(in_dir: Path = PROCESSED_DIR):
    observed = pd.read_csv(in_dir / "synthetic_demand.csv", parse_dates=["date"])
    truth = pd.read_csv(in_dir / "synthetic_truth.csv", parse_dates=["date"])
    return observed, truth


if __name__ == "__main__":
    observed, truth = generate_demand()
    save_synthetic(observed, truth)
    print(f"Saved {len(observed)} rows to {PROCESSED_DIR}")
    print(observed.head(10).to_string(index=False))
