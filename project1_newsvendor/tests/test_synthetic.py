"""Run from the repo root:  python -m pytest project1_newsvendor/tests -q"""

import numpy as np

from project1_newsvendor.src.data_synthetic import (
    DemandConfig,
    generate_demand,
    true_mean,
    true_quantile,
)


def test_same_seed_gives_identical_data():
    a, _ = generate_demand(seed=1)
    b, _ = generate_demand(seed=1)
    assert a.equals(b)


def test_different_seed_gives_different_data():
    a, _ = generate_demand(seed=1)
    b, _ = generate_demand(seed=2)
    assert not a["demand"].equals(b["demand"])


def test_shape_dates_and_positive_demand():
    cfg = DemandConfig(n_days=500)
    obs, truth = generate_demand(cfg)
    assert len(obs) == len(truth) == 500
    assert obs["date"].is_monotonic_increasing and obs["date"].is_unique
    assert (obs["demand"] > 0).all()


def test_observed_table_has_no_hidden_truth_columns():
    obs, _ = generate_demand()
    assert not any(c.startswith("true_") for c in obs.columns)


def test_true_quantile_has_correct_coverage():
    """Demand should fall at or below the true tau-quantile about tau of the time."""
    cfg = DemandConfig(n_days=20000)
    obs, truth = generate_demand(cfg, seed=7)
    for tau in (0.5, 0.75, 0.9):
        coverage = (obs["demand"].to_numpy() <= true_quantile(truth, tau)).mean()
        assert abs(coverage - tau) < 0.02


def test_mean_above_median_for_skewed_demand():
    _, truth = generate_demand()
    assert (true_mean(truth) > true_quantile(truth, 0.5)).all()
