"""Run from `code/`: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocess import _half_sample_mode  # noqa: E402


def test_half_sample_mode_three_points_keep_closer_pair():
    assert _half_sample_mode(np.array([0.0, 1.0, 1.1])) == 1.05
    assert _half_sample_mode(np.array([0.0, 0.1, 1.0])) == 0.05
    # narrows 5 -> 3 samples [0, 1, 1.1] before stopping: same answer as the 3-sample input
    assert _half_sample_mode(np.array([9.0, 5.0, 1.1, 0.0, 1.0])) == 1.05


def test_half_sample_mode_peak_and_nonfinite():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(2.0, 0.05, 700), rng.uniform(2.5, 6.0, 300), [np.nan]])
    assert abs(_half_sample_mode(x) - 2.0) < 0.05
