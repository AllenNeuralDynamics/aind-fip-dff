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


def test_brightfit_v2_signed_third_follows_early_dip_then_rise():
    """A cancelling pair of fast terms (early dip-then-rise) is out of reach of non-negative bleach amplitudes plus a
    single brightening term; the experimental `signed_third` option follows it. On a plain trace it changes nothing."""
    from utils.preprocess import tc_brightfit_v2

    t = np.arange(40000) * 0.05
    rng = np.random.default_rng(1)
    f0 = (8200 + 126 * np.exp(-t / 3939) + 562 * np.exp(-t / 1246) - 1500 * np.exp(-t / 2773)
          + 3799 * np.exp(-t / 39) - 3951 * np.exp(-t / 48.6))
    y = f0 + rng.normal(0, 40, t.size)
    early = t < 150
    err = lambda f: np.sqrt(np.mean((f[early] - f0[early]) ** 2))
    f_shipped, _ = tc_brightfit_v2(y, t)
    diag = {}
    f_signed, p_signed = tc_brightfit_v2(y, t, signed_third=True, diagnostics=diag)
    assert diag["accepted"] == 2 and diag["best_neg_ratio"] < 0.7 and diag["best_neg_tau"] < 180
    assert err(f_signed) < 0.1 * err(f_shipped)
    assert (p_signed[[1, 3, 5]] < 0).any()  # a negative bleach amplitude, i.e. the pair

    y_plain = 500 + 800 * np.exp(-t / 150) + rng.normal(0, 3, t.size)
    np.testing.assert_array_equal(tc_brightfit_v2(y_plain, t)[0], tc_brightfit_v2(y_plain, t, signed_third=True)[0])


def test_chunk_processing_signed_returns_selection_diagnostics():
    y, t = _bleaching_trace()
    _, params, _ = chunk_processing(y, t, method="bright-v2-signed", sampling_rate=20)
    assert {k for k in params if isinstance(k, str)} == {
        "diag_best_pos_ratio", "diag_best_neg_ratio", "diag_best_neg_tau", "diag_accepted"}
    assert len([k for k in params if isinstance(k, int)]) == 9
    _, params_v2, _ = chunk_processing(y, t, method="bright-v2", sampling_rate=20)
    assert all(isinstance(k, int) for k in params_v2)


def test_signed_third_tie_break_prefers_positive_candidate_in_near_tie(monkeypatch):
    """The four start candidates are stubbed to early RSS ratios (pos 0.50, pairs 0.40-0.49): with a tie-break margin a
    negative candidate wins only if its ratio is below margin x the best positive one."""
    import utils.preprocess as pp

    n = 100
    tc = np.ones(n)  # residual of the stub fits is (1 - f) so the early RSS ratio is (1 - f)^2 with f0_best = 0
    ts = np.arange(n, dtype=float)
    p_start = np.array([1.0, 0.1, 500.0, 0.1, 100.0])

    def run(ratios, margin):
        fits = iter(ratios)

        def stub(tc_ds, ts_ds, x0, bounds, **kw):
            ratio, neg = next(fits)
            x = np.array([1.0, 0.1, 500.0, 0.1, 100.0, -0.1 if neg else 0.1, 20.0])
            return np.full(n, 1 - np.sqrt(ratio)), type("R", (), {"x": x})()

        monkeypatch.setattr(pp, "nonlinear_fit", stub)
        params, n_exp = pp._try_signed_third_exp(tc, ts, p_start, np.zeros(n), False, n, 0.97, {}, None, margin)
        return n_exp, bool((params[1:7:2] < 0).any())

    near_tie = [(0.50, False), (0.46, True), (0.90, True), (0.95, True)]  # negative only 8% better
    clear_win = [(0.50, False), (0.30, True), (0.90, True), (0.95, True)]  # negative 40% better
    assert run(near_tie, None) == (3, True)  # no tie-break: lowest RSS wins
    assert run(near_tie, 0.9) == (3, False)  # near-tie goes to the positive candidate
    assert run(clear_win, 0.9) == (3, True)  # a clear improvement keeps the negative one
    assert run([(0.80, False), (0.40, True), (0.95, True), (0.95, True)], 0.75) == (3, True)  # both pass, the negative one is far enough ahead (0.40 < 0.75 x 0.80)
    assert run([(0.99, False), (0.40, True), (0.95, True), (0.95, True)], 0.75) == (3, True)  # positive fails the gate


def test_chunk_processing_tie_methods_match_signed_when_negative_wins_clearly():
    from utils.preprocess import SIGNED_METHODS

    assert SIGNED_METHODS["bright-v2-signed"] is None and SIGNED_METHODS["bright-v2-signed-tie75"] < SIGNED_METHODS["bright-v2-signed-tie90"] < 1
    y, t = _bleaching_trace()
    for m in SIGNED_METHODS:
        _, params, _ = chunk_processing(y, t, method=m, sampling_rate=20)
        assert len([k for k in params if isinstance(k, int)]) == 9 and "diag_accepted" in params
