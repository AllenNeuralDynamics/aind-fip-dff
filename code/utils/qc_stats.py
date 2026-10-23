"""Statistics behind the baseline-level and motion-correction-noise QC metrics.

Only numpy/scipy and `aind_ophys_utils`, so they can be tested without the
capsule's NWB stack.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt
from aind_ophys_utils.signal_utils import noise_std

#: Length of the start/middle/end windows of `baseline_levels` [s], capped at
#: a sixth of the recording.
BASELINE_WINDOW_S = 300
#: Recordings shorter than this [s] get no baseline levels.
BASELINE_MIN_DURATION_S = 1000


def _mode(x: np.ndarray, scale: float) -> float:
    """Peak of the histogram of `x` (bin width `scale`/4), smoothed with a 2-bin Gaussian."""
    lo, hi = np.percentile(x, [1, 99])
    n_bins = max(int((hi - lo) / (scale / 4)), 20)
    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    i = int(np.argmax(gaussian_filter1d(counts.astype(float), 2)))
    return float(0.5 * (edges[i] + edges[i + 1]))


def baseline_levels(dff: np.ndarray, fs: float) -> dict | None:
    """Baseline level of dF/F in the first, middle and last window of the recording.

    The level is the mode of the 1 Hz low-passed trace, which sits at the
    baseline when transients are sparse; it is 0 for a baseline F0 that tracks
    the signal and negative (positive) when F0 is too high (low).

    Parameters
    ----------
    dff : np.ndarray
        dF/F trace (fraction, not percent).
    fs : float
        Sampling rate [Hz].

    Returns
    -------
    dict or None
        `start`, `mid`, `end` (levels, same units as `dff`) and
        `frac_baseline`, the fraction of samples within 2 robust sigma above
        a 60 s running median (low for traces dominated by transients, where
        the mode is not the baseline). None if the recording is shorter than
        `BASELINE_MIN_DURATION_S` or not finite.
    """
    n = len(dff)
    if n < BASELINE_MIN_DURATION_S * fs or not np.isfinite(dff).all():
        return None
    xf = sosfiltfilt(butter(2, 1.0, fs=fs, output="sos"), dff)
    step = max(int(round(fs / 2)), 1)
    running = np.repeat(median_filter(xf[::step], size=int(60 * fs / step), mode="nearest"), step)[:n]
    resid = xf - running
    below = resid[resid < 0]
    scale = 1.4826 * np.median(np.abs(below)) if below.size else 0.0
    if not scale > 0:
        return None
    w = min(int(BASELINE_WINDOW_S * fs), n // 6)
    mid = n // 2 - w // 2
    return {
        "start": _mode(xf[:w], scale),
        "mid": _mode(xf[mid : mid + w], scale),
        "end": _mode(xf[n - w :], scale),
        "frac_baseline": float((resid < 2 * scale).mean()),
    }


def noise_inflation(dff: np.ndarray, motion_corrected: np.ndarray, fs: float, cutoff_freq_noise: float | None) -> float:
    """Factor by which the Iso subtraction raises the high-frequency noise of a channel.

    `motion_correct` fits its coefficient on a 0.05 Hz low-passed signal but
    subtracts `coef * Iso` at all frequencies, which adds Iso noise to the
    channel where the two are not coupled. The ratio of the MAD noise
    estimates (`noise_std(method="mad")`, as for the calibration ratio) compares
    the corrected trace with `dff` low-passed at `cutoff_freq_noise` only, so
    the final low-pass of `motion_correct` cancels. 1 = no noise added.
    """
    ref = dff
    if cutoff_freq_noise is not None and cutoff_freq_noise < fs / 2:
        ref = sosfiltfilt(butter(2, cutoff_freq_noise, fs=fs, output="sos"), dff)
    noise_ref = float(noise_std(ref, method="mad"))
    if not noise_ref > 0 or not np.isfinite(motion_corrected).all():
        return float("nan")
    return float(noise_std(motion_corrected, method="mad")) / noise_ref
