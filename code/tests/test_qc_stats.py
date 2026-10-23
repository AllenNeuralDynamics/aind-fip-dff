"""Run from `code/`: python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.qc_stats import baseline_levels, noise_inflation  # noqa: E402

FS = 20


def _trace(offsets=(0.0, 0.0, 0.0), minutes=30, seed=0):
    """Noise plus sparse positive transients; `offsets` added to the first/middle/last 5 min."""
    rng = np.random.default_rng(seed)
    n = minutes * 60 * FS
    x = 0.01 * rng.standard_normal(n)
    for i in rng.choice(n - 100, size=40, replace=False):
        x[i : i + 100] += 0.05 * np.exp(-np.arange(100) / 20)
    w = 5 * 60 * FS
    x[:w] += offsets[0]
    x[n // 2 - w // 2 : n // 2 + w // 2] += offsets[1]
    x[n - w :] += offsets[2]
    return x


def test_baseline_levels_recover_offsets():
    lv = baseline_levels(_trace((0.02, 0.0, -0.01)), FS)
    assert abs(lv["start"] - 0.02) < 0.002
    assert abs(lv["mid"]) < 0.002
    assert abs(lv["end"] + 0.01) < 0.002
    assert lv["frac_baseline"] > 0.9


def test_baseline_levels_short_or_nonfinite_is_none():
    assert baseline_levels(_trace(minutes=10), FS) is None
    x = _trace()
    x[5] = np.nan
    assert baseline_levels(x, FS) is None


def test_noise_inflation():
    rng = np.random.default_rng(1)
    n = 30 * 60 * FS
    g, iso = rng.standard_normal(n), rng.standard_normal(n)
    sos = butter(2, 3, fs=FS, output="sos")
    # no subtraction: only the final low-pass, so no inflation
    assert abs(noise_inflation(g, sosfiltfilt(sos, g), FS, 3) - 1) < 1e-6
    # subtracting 2 x an independent Iso of equal noise: sqrt(1 + 2^2)
    assert abs(noise_inflation(g, sosfiltfilt(sos, g - 2 * iso), FS, 3) - np.sqrt(5)) < 0.1
    assert np.isnan(noise_inflation(g, np.full(n, np.nan), FS, 3))
