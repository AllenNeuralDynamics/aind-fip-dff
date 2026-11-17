"""Run from `code/`: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocess import (  # noqa: E402
    _evaluate_baseline,
    _half_sample_mode,
    chunk_processing,
    motion_correct,
)


def test_half_sample_mode_three_points_keep_closer_pair():
    assert _half_sample_mode(np.array([0.0, 1.0, 1.1])) == 1.05
    assert _half_sample_mode(np.array([0.0, 0.1, 1.0])) == 0.05
    # narrows 5 -> 3 samples [0, 1, 1.1] before stopping: same answer as the 3-sample input
    assert _half_sample_mode(np.array([9.0, 5.0, 1.1, 0.0, 1.0])) == 1.05


def test_half_sample_mode_peak_and_nonfinite():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(2.0, 0.05, 700), rng.uniform(2.5, 6.0, 300), [np.nan]])
    assert abs(_half_sample_mode(x) - 2.0) < 0.05


def _bleaching_trace(n=3000, seed=0):
    t = np.arange(n) * 0.05
    rng = np.random.default_rng(seed)
    return 800 * np.exp(-t / 150) + 500 + rng.normal(0, 3, n), t


@pytest.mark.parametrize("method", ["poly", "exp", "bright-v2"])
def test_chunk_processing_drops_nonfinite_samples(method, caplog):
    """Non-finite samples are dropped before fitting, not interpolated."""
    tc, t = _bleaching_trace()
    tc[1000:1010] = np.nan  # NaN signal in the middle
    tc[1500] = np.inf
    t[2500] = np.nan  # NaN timestamp in the middle
    t[2900:] = np.nan  # frames without timestamp at the end
    valid = np.isfinite(tc) & np.isfinite(t)
    dff, params, fit = chunk_processing(tc, t, method=method, n_frame_to_cut=0)
    dff_ref, params_ref, fit_ref = chunk_processing(
        tc[valid], t[valid], method=method, n_frame_to_cut=0
    )
    # dF/F: NaN at the dropped samples, as if they were absent elsewhere
    assert np.isnan(dff[~valid]).all()
    np.testing.assert_array_equal(dff[valid], dff_ref)
    assert np.isfinite(dff_ref).all() and params.keys() == params_ref.keys()
    # baseline: the fit's, NaN only without a timestamp, also where the signal is not finite
    np.testing.assert_array_equal(fit[valid], fit_ref)
    np.testing.assert_array_equal(np.isfinite(fit), np.isfinite(t))
    assert fit[999] > fit[1005] > fit[1010]  # continues the bleaching decay
    assert f"dropping {(~valid).sum()} non-finite" in caplog.text


def test_chunk_processing_drops_nonfinite_signal_that_the_median_filter_hides():
    tc, t = _bleaching_trace()
    tc[1500] = np.inf  # a median of 3 would turn it into a finite value
    dff, _, fit = chunk_processing(tc, t, method="poly", n_frame_to_cut=0, kernel_size=3)
    assert np.isnan(dff[1500]) and np.isfinite(np.delete(dff, 1500)).all()
    assert np.isfinite(fit).all()


@pytest.mark.parametrize("method", ["poly", "exp", "bright-v2"])
def test_evaluate_baseline_reproduces_the_fit(method):
    tc, t = _bleaching_trace()
    _, params, fit = chunk_processing(tc, t, method=method, n_frame_to_cut=0)
    coefs = np.array(list(params.values()), dtype=float)
    np.testing.assert_allclose(_evaluate_baseline(method, coefs, t), fit, rtol=1e-12)


def test_chunk_processing_without_valid_frames_gives_nans(caplog):
    dff, _, fit = chunk_processing(np.full(500, np.nan), np.arange(500) * 0.05, method="poly")
    assert np.isnan(dff).all() and np.isnan(fit).all()
    assert "nothing to fit" in caplog.text


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
    dff.loc[2900:3400, "G"] = np.nan  # a longer one, where Iso is finite
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
    ok = mc["G"].notna()  # the Iso term is centred over these frames, so the mean is kept
    assert abs(mc["G"][ok].mean() - dff["G"][ok].mean()) < 1e-3


def test_motion_correct_leaves_cut_frames_out_of_the_fit():
    dff = _dff_frame()
    _, _, _, _, weights = motion_correct(dff, n_frame_to_cut=100)
    assert np.isnan(weights["G"][:100]).all() and np.isfinite(weights["G"][100:]).all()
    mc, _, _, _, _ = motion_correct(dff, n_frame_to_cut=100)
    assert np.isfinite(mc.to_numpy()).all()  # the cut frames are still corrected
