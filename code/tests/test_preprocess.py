"""Run from `code/`: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocess import _half_sample_mode, motion_correct  # noqa: E402


def test_half_sample_mode_three_points_keep_closer_pair():
    assert _half_sample_mode(np.array([0.0, 1.0, 1.1])) == 1.05
    assert _half_sample_mode(np.array([0.0, 0.1, 1.0])) == 0.05
    # narrows 5 -> 3 samples [0, 1, 1.1] before stopping: same answer as the 3-sample input
    assert _half_sample_mode(np.array([9.0, 5.0, 1.1, 0.0, 1.0])) == 1.05


def test_half_sample_mode_peak_and_nonfinite():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(2.0, 0.05, 700), rng.uniform(2.5, 6.0, 300), [np.nan]])
    assert abs(_half_sample_mode(x) - 2.0) < 0.05


def _dff_frame(n=4000, seed=0):
    """dF/F of G and R that follow Iso with factors 0.7 and 0.2 (a slow random walk)."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "G": 0.7 * walk + rng.normal(0, 0.01, n),
            "Iso": walk + rng.normal(0, 0.01, n),
            "R": 0.2 * walk + rng.normal(0, 0.01, n),
        }
    )


def test_motion_correct_ignores_nonfinite_frames():
    dff = _dff_frame()
    _, _, coef_ref, _, _ = motion_correct(dff)
    assert abs(coef_ref["G"] - 0.7) < 0.05 and abs(coef_ref["R"] - 0.2) < 0.05
    dff.loc[1000:1010, "G"] = np.nan  # a gap in one channel
    dff.loc[2500, "Iso"] = np.inf  # a frame without Iso
    dff.loc[3500:, "Iso"] = np.nan  # an Iso tail
    mc, _, coef, _, weights = motion_correct(dff)
    no_iso = ~np.isfinite(dff["Iso"])
    for ch in ("G", "R"):
        gap = no_iso | ~np.isfinite(dff[ch])
        assert (mc[ch].isna() == gap).all()
        assert (np.isnan(weights[ch]) == gap).all()
    assert (mc["Iso"].isna() == no_iso).all()
    assert abs(coef["G"] - coef_ref["G"]) < 0.05 and abs(coef["R"] - coef_ref["R"]) < 0.05
