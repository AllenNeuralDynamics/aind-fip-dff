"""Experimental bisection variant of `tc_brightfit_v2`'s M-estimator choice.

Background: `tc_brightfit_v2` (see `preprocess.py`) fits every trace with a
single fixed M-estimator (`M_DFF`, `AsymmetricTukeyBiweight(c_pos=3.5,
c_neg=4.0)`). This module instead bisects `c_pos` per trace so the fit's own
calibration ratio (see `_calibration_ratio`) lands at a target value -- the
same idea `brightfit-comparison.ipynb`'s `nonlinear_fit_bisect` tested (and
`aind-ophys-dff` already uses in production), adapted here to
`tc_brightfit_v2`'s actual multi-stage architecture instead of a simplified
single-stage fit.

This is NOT wired into `run_capsule.py`/`reprocess.py`'s `--dff_methods` and
is not part of aind-fip-dff PR #91 -- it exists purely to benchmark against
the 383-asset sweep and inform the bright_legacy-vs-bright M-estimator
discussion with real numbers (motivated by: bright_legacy's own calibration
ratio is close to 1.0, and colleagues want a candidate that's close to
bright_legacy's behavior without reintroducing bright_legacy's full cost).

Cost model -- deliberately NOT assumed to be cheap
---------------------------------------------------
`tc_brightfit_v2`'s decimated model-selection steps (cold-start 2-exp, try
brightening, try 3rd exponential) and `_snap_degenerate_slow_terms` always
fit with `M=None` (OLS) -- `c_pos`/`c_neg` never affect which model gets
selected. Only the final full-resolution IRLS refit ("Step 4") depends on M.
So here, model selection runs ONCE per trace (same cost as the default
`tc_brightfit_v2`), and only Step 4 -- full resolution, `maxiter` IRLS
iterations, almost certainly the single most expensive part of the pipeline
-- is repeated once per bisection candidate (up to `n_iter + 1` times
total, vs. `tc_brightfit_v2`'s single call). Use `benchmark_bisect_vs_default`
below to measure the actual multiplier on real traces before assuming any
particular speedup number.
"""

import logging
import time
import warnings

import numpy as np
import aind_ophys_utils.baseline_fitting as _baseline_fitting_module
from aind_ophys_utils.baseline_fitting import (
    AsymmetricTukeyBiweight as NonlinearFitAsymmetricTukeyBiweight,
)
from aind_ophys_utils.baseline_fitting import nonlinear_fit, sum_of_exps
from aind_ophys_utils.array_utils import downsample_array
from aind_ophys_utils.signal_utils import noise_std

from .preprocess import (
    _init_sum_of_exps,
    _pad_sum_of_exps_params,
    _snap_degenerate_slow_terms,
    _sort_bleach_params,
)

#: DIAGNOSTIC (temporary): nonlinear_fit's own outer IRLS loop (`maxiter`,
#: default 5) calls scipy.optimize.minimize once per outer iteration, but
#: only returns the LAST call's OptimizeResult as `res` -- so logging
#: res.nit (as this module already does after every nonlinear_fit call)
#: only sees the final, often-already-converged inner solve. A real
#: pathological asset showed e.g. a logged nit=1 with 64s elapsed since
#: the previous log line -- the expensive part is in an earlier, unlogged
#: outer IRLS iteration's inner call. Patching scipy.optimize.minimize
#: itself (as imported into aind_ophys_utils.baseline_fitting's own
#: namespace) logs every inner call, closing that blind spot. Remove once
#: this question (is the maxiter cap, at whatever value, ever actually
#: load-bearing?) is answered -- not meant to ship.
_orig_scipy_minimize = _baseline_fitting_module.minimize


def _logging_minimize(*args, **kwargs):
    t0 = time.perf_counter()
    res = _orig_scipy_minimize(*args, **kwargs)
    logging.info(
        f"tc_brightfit_v2_bisect: inner minimize() call took "
        f"{time.perf_counter() - t0:.2f}s, nit={res.nit}, success={res.success}, "
        f"message={getattr(res, 'message', None)!r}"
    )
    return res


_baseline_fitting_module.minimize = _logging_minimize


def _calibration_ratio(signal: np.ndarray, f0: np.ndarray) -> tuple[float, float]:
    """Duplicated from `run_capsule._calibration_ratio` (same definition,
    kept in sync by hand) rather than imported -- `run_capsule.py` pulls in
    the full NWB I/O stack (pynwb, hdmf_zarr, aind_data_schema, ...) at
    import time, which this lightweight, standalone-runnable experiment
    module has no other reason to depend on."""
    resid = signal - f0
    valid = np.isfinite(resid)
    if not valid.any():
        return np.nan, np.nan
    sigma = float(noise_std(signal[valid], method="welch"))
    deep = resid[valid & (resid < 0)]
    if len(deep) <= 10 or sigma == 0:
        return np.nan, sigma
    return float(np.median(-deep) / (0.6745 * sigma)), sigma


def _decimated_model_selection(
    trace_valid: np.ndarray,
    ts_valid: np.ndarray,
    ds: int,
    rss_thresh: tuple[float, float],
    t_eval_exp3: float,
) -> tuple[np.ndarray, int, bool, str]:
    """`tc_brightfit_v2`'s Steps 1-3 + initial degeneracy check, copied
    (not imported) so this experiment never touches the already-reviewed
    `tc_brightfit_v2` -- this is the one part of the pipeline that's
    M-independent (every `nonlinear_fit` call here uses `M=None`), so it
    only needs to run ONCE regardless of how many `c_pos` candidates
    bisection tries on Step 4 below.

    Returns
    -------
    params_ds : np.ndarray
        Fitted parameter vector (decimated-selection winner).
    n_exp_won, include_bright : the selected model.
    model_str : str
        Human-readable description, as in `tc_brightfit_v2`.
    """
    tc_ds = downsample_array(trace_valid, factors=ds, strategy="first")
    ts_ds = downsample_array(ts_valid, factors=ds, strategy="first")
    dt_ds = float(ts_ds[1] - ts_ds[0]) if len(ts_ds) > 1 else float(ds)
    n_eval_ds = min(len(tc_ds), round(t_eval_exp3 / dt_ds))
    # Pinned explicitly (see SCIPY_LBFGSB_DEFAULTS) -- plain OLS (M=None),
    # so scipy's own defaults are fine here; just not left implicit.
    kw_ds = dict(model=sum_of_exps, M=None, optimizer_options=SCIPY_LBFGSB_DEFAULTS)

    def _rss_ds_full(f0):
        return float(np.sum((tc_ds - f0) ** 2))

    def _rss_ds_early(f0):
        return float(np.sum((tc_ds[:n_eval_ds] - f0[:n_eval_ds]) ** 2))

    # Step 1: 2-exp OLS cold start on the decimated trace.
    x0, bnd = _init_sum_of_exps(tc_ds, n_exp=2, include_brightening=False)
    f0_ds, res_ds = nonlinear_fit(tc_ds, ts_ds, x0=x0, bounds=bnd, **kw_ds)
    logging.info(
        f"tc_brightfit_v2_bisect: model selection step 1 (2-exp OLS) "
        f"nit={res_ds.nit}, success={res_ds.success}, message={res_ds.message!r}"
    )
    rss_ds = _rss_ds_full(f0_ds)
    f0_ds_best = f0_ds
    params_ds = _sort_bleach_params(res_ds.x, 2)
    n_exp_won = 2
    include_bright = False

    # Step 2: try adding brightening (full decimated-trace RSS).
    x0_b = np.concatenate([params_ds, [min(-0.05 * params_ds[0], 0.0), 2000.0]])
    _, bnd_b = _init_sum_of_exps(tc_ds, n_exp=n_exp_won, include_brightening=True)
    f0_b_ds, res_b_ds = nonlinear_fit(tc_ds, ts_ds, x0=x0_b, bounds=bnd_b, **kw_ds)
    logging.info(
        f"tc_brightfit_v2_bisect: model selection step 2 (+bright OLS) "
        f"nit={res_b_ds.nit}, success={res_b_ds.success}, message={res_b_ds.message!r}"
    )
    rss_b_ds = _rss_ds_full(f0_b_ds)
    if rss_b_ds < rss_thresh[0] * rss_ds:
        rss_ds, f0_ds_best, params_ds, include_bright = (
            rss_b_ds,
            f0_b_ds,
            _sort_bleach_params(res_b_ds.x, n_exp_won, True),
            True,
        )

    # Step 3: try adding a 3rd bleach exponential (early-window RSS).
    n_exp_next = n_exp_won + 1
    new_exp = np.array([max(0.05 * params_ds[0], 0.0), 50.0])
    x0_3 = (
        np.concatenate([params_ds[:-2], new_exp, params_ds[-2:]])
        if include_bright
        else np.concatenate([params_ds, new_exp])
    )
    _, bnd_3 = _init_sum_of_exps(tc_ds, n_exp=n_exp_next, include_brightening=include_bright)
    f0_3_ds, res_3_ds = nonlinear_fit(tc_ds, ts_ds, x0=x0_3, bounds=bnd_3, **kw_ds)
    logging.info(
        f"tc_brightfit_v2_bisect: model selection step 3 (3rd-exp OLS) "
        f"nit={res_3_ds.nit}, success={res_3_ds.success}, message={res_3_ds.message!r}"
    )
    p3_sorted = _sort_bleach_params(res_3_ds.x, n_exp_next, include_bright)
    tau_new = p3_sorted[2 + 2 * n_exp_won]
    if (
        _rss_ds_early(f0_3_ds) < rss_thresh[1] * _rss_ds_early(f0_ds_best)
        and tau_new < 180.0
    ):
        params_ds = p3_sorted
        n_exp_won = n_exp_next
    model_str = f"{n_exp_won}-exp{'+bright' if include_bright else ''}"

    params_ds, n_exp_won, include_bright, snapped = _snap_degenerate_slow_terms(
        params_ds, n_exp_won, include_bright, tc_ds, ts_ds
    )
    if snapped:
        model_str += f" [snapped: {','.join(snapped)}]"

    return params_ds, n_exp_won, include_bright, model_str


#: nonlinear_fit's own default is {"maxiter": 20000, "ftol": 1e-12,
#: "gtol": 1e-10} -- generous enough that an aggressive, redescending
#: M (e.g. c_pos near the bottom of the bisection bracket) can make the
#: robust loss landscape locally flat over large regions on a real,
#: messy trace, and scipy's L-BFGS-B can burn through most of that
#: 20000-iteration budget trying to satisfy a gtol it may never actually
#: reach, rather than failing outright. Confirmed in practice: one real
#: candidate at c_pos=2.0 took 172.82s on its own (vs well under 1s on
#: clean synthetic traces). Used for every bisection candidate below --
#: no separate high-precision final refit: a 9-parameter L-BFGS-B problem
#: that's well-posed converges in far fewer than 500 iterations anyway,
#: and the pathological (locally-flat-loss) case this guards against
#: wouldn't be meaningfully improved by more iterations either, since
#: "flat" means many parameter values already give nearly the same loss.
#: Also well within precedent: tc_brightfit (bright_legacy, already in
#: production) uses maxiter=5 and a much looser relative-deviance
#: tolerance (tol=1e-3) for its own IRLS convergence check -- these
#: settings aren't a shortcut invented for this experiment.
#:
#: DIAGNOSTIC (temporary): maxiter bumped 500 -> 15000 to isolate whether
#: the loosened ftol/gtol above are *already* what stops a pathological
#: candidate early (via its own CONVERGENCE message, well under 500
#: iterations), or whether 500 is actually the thing doing the truncating
#: (message would show the iteration-limit was hit instead). The new
#: nit/success/message logging in _full_resolution_refit will show which,
#: on the next run against the known artifact-laden pathological asset.
#: Revert to 500 once that's answered -- this value is not meant to ship.
SEARCH_OPTIMIZER_OPTIONS = {"maxiter": 15000, "ftol": 1e-9, "gtol": 1e-7}

#: scipy.optimize.minimize's own L-BFGS-B defaults, pinned explicitly here
#: rather than left to nonlinear_fit's current default ({"maxiter": 20000,
#: "ftol": 1e-12, "gtol": 1e-10} -- far tighter than scipy's own, and the
#: root cause investigated above) so this module's behavior doesn't
#: silently change if/when that library default is updated. Used for the
#: plain-OLS (M=None) model-selection calls below, where there's no
#: redescending-M pathology to guard against -- scipy's own defaults are
#: a reasonable, unremarkable choice there.
SCIPY_LBFGSB_DEFAULTS = {"maxiter": 15000, "ftol": 2.220446049250313e-09, "gtol": 1e-5}


def _full_resolution_refit(
    trace_valid: np.ndarray,
    ts_valid: np.ndarray,
    params_ds: np.ndarray,
    n_exp_won: int,
    include_bright: bool,
    model_str: str,
    M,
    maxiter: int,
    sigma_anneal_steps: int,
    fixed_sigma,
    optimizer_options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, int, bool, str]:
    """`tc_brightfit_v2`'s Step 4 (+ its own snap-refit loop), copied so the
    bisection search below can call it once per candidate `c_pos` without
    touching the already-reviewed `tc_brightfit_v2`.

    Returns an extra `n_refit_calls` (vs. `tc_brightfit_v2`'s equivalent)
    -- this loop is itself M-dependent and runs once per bisection
    candidate, so a real trace needing several snap-refit rounds multiplies
    bisection's total cost well beyond `n_iter + 1`; see
    `tc_brightfit_v2_bisect`'s per-candidate logging.

    `optimizer_options` is passed straight through to `nonlinear_fit` --
    see `SEARCH_OPTIMIZER_OPTIONS` above for why this bounds worst-case
    cost without meaningfully compromising fit quality."""
    kw_full = dict(model=sum_of_exps, M=M, maxiter=maxiter, sigma_anneal_steps=sigma_anneal_steps)
    if fixed_sigma is not None:
        kw_full["fixed_sigma"] = fixed_sigma
    if optimizer_options is not None:
        kw_full["optimizer_options"] = optimizer_options

    if n_exp_won == 0 and not include_bright:
        f0 = np.full_like(trace_valid, params_ds[0])
        return f0, params_ds, n_exp_won, include_bright, model_str, 0

    _, bnd_full = _init_sum_of_exps(trace_valid, n_exp=n_exp_won, include_brightening=include_bright)
    f0, res_full = nonlinear_fit(trace_valid, ts_valid, x0=params_ds, bounds=bnd_full, **kw_full)
    logging.info(
        f"tc_brightfit_v2_bisect: full-resolution refit (initial) "
        f"nit={res_full.nit}, success={res_full.success}, message={res_full.message!r}"
    )
    params_full = _sort_bleach_params(res_full.x, n_exp_won, include_bright)
    n_refit_calls = 1

    while True:
        params_full, n_exp_final, include_bright_final, snapped_final = _snap_degenerate_slow_terms(
            params_full, n_exp_won, include_bright, trace_valid, ts_valid
        )
        if not snapped_final:
            break
        n_exp_won, include_bright = n_exp_final, include_bright_final
        model_str += f" [snapped: {','.join(snapped_final)}]"
        if n_exp_won == 0 and not include_bright:
            f0 = np.full_like(trace_valid, params_full[0])
            return f0, params_full, n_exp_won, include_bright, model_str, n_refit_calls
        _, bnd_full = _init_sum_of_exps(trace_valid, n_exp=n_exp_won, include_brightening=include_bright)
        f0, res_full = nonlinear_fit(trace_valid, ts_valid, x0=params_full, bounds=bnd_full, **kw_full)
        logging.info(
            f"tc_brightfit_v2_bisect: full-resolution refit (snap-loop) "
            f"nit={res_full.nit}, success={res_full.success}, message={res_full.message!r}"
        )
        params_full = _sort_bleach_params(res_full.x, n_exp_won, include_bright)
        n_refit_calls += 1

    return f0, params_full, n_exp_won, include_bright, model_str, n_refit_calls


def tc_brightfit_v2_bisect(
    trace: np.ndarray,
    timestamps: np.ndarray,
    c_neg: float = 4.0,
    c_lo: float = 2.0,
    c_hi: float = 4.685,
    n_iter: int = 5,
    target_ratio: float = 1.0,
    rss_thresh: tuple[float, float] = (0.98, 0.97),
    maxiter: int = 5,
    ds: int = 10,
    fixed_sigma: float | str | None = "auto",
    sigma_anneal_steps: int = 4,
    t_eval_exp3: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Experimental bisect variant of `tc_brightfit_v2`: instead of a single
    fixed M-estimator for every trace, bisects `c_pos` (holding `c_neg`
    fixed) so the final fit's own calibration ratio lands at `target_ratio`.

    See this module's docstring for the cost model -- model selection runs
    once; only the full-resolution refit repeats per bisection candidate.

    Parameters
    ----------
    trace, timestamps : np.ndarray
        Fiber photometry signal and its timestamps.
    c_neg : float, optional
        Fixed negative-tail Tukey threshold. Default 4.0, matching
        `tc_brightfit_v2`'s own `M_DFF` default -- only `c_pos` is bisected.
    c_lo, c_hi : float, optional
        Bisection bracket for `c_pos`. Default (2.0, 4.685): 4.685 is the
        canonical (fully symmetric) Tukey constant; 2.0 is aggressive enough
        to usually overshoot `target_ratio` from below (matches
        `brightfit-comparison.ipynb`'s `nonlinear_fit_bisect`).
    n_iter : int, optional
        Bisection steps; precision in `c_pos` is `(c_hi - c_lo) / 2**n_iter`.
        Default 5 -> +/-0.08. Each step costs one more full-resolution
        refit (see cost model above) -- this is the main speed/precision
        knob.
    target_ratio : float, optional
        Calibration-ratio target to bisect toward. Default 1.0 (closest to
        `bright_legacy`'s own pooled calibration ratio, ~1.004 in the
        383-asset sweep). Pass a different value (e.g. a symmetric-M,
        Iso-channel-derived target) to test an alternative target directly
        without touching the bisection logic itself.
    rss_thresh, maxiter, ds, fixed_sigma, sigma_anneal_steps, t_eval_exp3 :
        Same meaning as `tc_brightfit_v2`.

    Returns
    -------
    f0 : np.ndarray
        Baseline at the winning `c_pos`.
    params : np.ndarray
        Fitted parameters, padded the same way as `tc_brightfit_v2`.
    c_pos_chosen : float
        The winning `c_pos` value.
    model_str : str
        Human-readable model description, plus the winning `c_pos` and the
        number of full-resolution refits actually performed (for timing
        analysis -- see `benchmark_bisect_vs_default`).
    """
    valid = np.isfinite(trace) & np.isfinite(timestamps)
    if not valid.any():
        raise ValueError(
            "tc_brightfit_v2_bisect: every sample is non-finite (trace or "
            "timestamps) -- nothing to fit."
        )
    n_dropped = int((~valid).sum())
    if n_dropped:
        warnings.warn(
            f"tc_brightfit_v2_bisect: dropping {n_dropped} non-finite "
            "sample(s) (trace or timestamps) before fitting.",
            stacklevel=2,
        )
        trace_valid = trace[valid]
        ts_valid = timestamps[valid]
    else:
        trace_valid = trace
        ts_valid = timestamps

    if fixed_sigma == "auto":
        sigma_est = float(noise_std(trace_valid, method="welch"))
        fixed_sigma = sigma_est if np.isfinite(sigma_est) and sigma_est > 0 else None

    t_start = time.perf_counter()

    # Model selection: ONCE, M-independent (see module docstring).
    t0 = time.perf_counter()
    params_ds, n_exp_won, include_bright, model_str = _decimated_model_selection(
        trace_valid, ts_valid, ds, rss_thresh, t_eval_exp3
    )
    logging.info(
        f"tc_brightfit_v2_bisect: model selection ({model_str}) took "
        f"{time.perf_counter() - t0:.2f}s"
    )

    n_full_refits = 0
    n_candidates = 0

    def _fit_and_ratio(c_pos):
        nonlocal n_full_refits, n_candidates
        n_candidates += 1
        t0 = time.perf_counter()
        M = NonlinearFitAsymmetricTukeyBiweight(c_pos=c_pos, c_neg=c_neg)
        f0, params_full, n_exp_f, bright_f, mstr, n_refit_calls = _full_resolution_refit(
            trace_valid,
            ts_valid,
            params_ds,
            n_exp_won,
            include_bright,
            model_str,
            M,
            maxiter,
            sigma_anneal_steps,
            fixed_sigma,
            optimizer_options=SEARCH_OPTIMIZER_OPTIONS,
        )
        n_full_refits += n_refit_calls
        ratio, _ = _calibration_ratio(trace_valid, f0)
        logging.info(
            f"tc_brightfit_v2_bisect: candidate {n_candidates} (c_pos={c_pos:.3f}) "
            f"took {time.perf_counter() - t0:.2f}s, {n_refit_calls} refit call(s) "
            f"(snap-loop), ratio={ratio:.3f}"
        )
        return f0, params_full, n_exp_f, bright_f, mstr, ratio

    # Evaluate the lower bound -- early exit if already at/above target
    # (mirrors brightfit-comparison.ipynb's nonlinear_fit_bisect).
    best_f0, best_params, best_n_exp, best_bright, best_mstr, ratio_lo = _fit_and_ratio(c_lo)
    best_c = c_lo

    if not (np.isnan(ratio_lo) or ratio_lo >= target_ratio):
        best_dist = abs(ratio_lo - target_ratio)
        lo, hi = c_lo, c_hi
        for _ in range(n_iter):
            c_mid = (lo + hi) / 2
            f0_mid, params_mid, n_exp_mid, bright_mid, mstr_mid, ratio_mid = _fit_and_ratio(c_mid)
            if np.isnan(ratio_mid):
                hi = c_mid
                continue
            dist = abs(ratio_mid - target_ratio)
            if dist < best_dist:
                best_dist = dist
                best_f0, best_params, best_n_exp, best_bright, best_mstr = (
                    f0_mid,
                    params_mid,
                    n_exp_mid,
                    bright_mid,
                    mstr_mid,
                )
                best_c = c_mid
            if ratio_mid < target_ratio:
                lo = c_mid  # still too aggressive -> raise lower bound
            else:
                hi = c_mid  # permissive enough -> tighten upper bound

    final_model_str = f"{best_mstr} [bisect c_pos={best_c:.3f}, {n_full_refits} full refit(s)]"
    logging.info(
        f"tc_brightfit_v2_bisect: done in {time.perf_counter() - t_start:.2f}s total, "
        f"{n_candidates} candidate(s), {n_full_refits} full-resolution refit call(s), "
        f"winning c_pos={best_c:.3f}"
    )

    f0_out = best_f0
    if n_dropped:
        f0_out = np.full(len(trace), np.nan)
        f0_out[valid] = best_f0

    return f0_out, _pad_sum_of_exps_params(best_params, best_n_exp, best_bright), best_c, final_model_str


def benchmark_bisect_vs_default(
    trace: np.ndarray, timestamps: np.ndarray, **bisect_kwargs
) -> dict:
    """Time `tc_brightfit_v2` (default, fixed M) against
    `tc_brightfit_v2_bisect` on one real trace -- run this on a handful of
    real traces BEFORE assuming any particular speedup/slowdown number (see
    this module's docstring). Reports wall-clock time for both, the
    resulting speedup-vs-bright_legacy erosion factor (using this PR's own
    validated bright_legacy median fit time), and the winning `c_pos` and
    number of full-resolution refits bisection actually needed.
    """
    import time

    from .preprocess import tc_brightfit_v2

    #: Median per-trace fit time from aind-fip-dff PR #91's own 383-asset
    #: sweep (demean run) -- see the PR description's Validation section.
    BRIGHT_MEDIAN_S = 1.94
    BRIGHT_LEGACY_MEDIAN_S = 57.0

    t0 = time.perf_counter()
    tc_brightfit_v2(trace, timestamps)
    t_default = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _, c_pos_chosen, model_str = tc_brightfit_v2_bisect(trace, timestamps, **bisect_kwargs)
    t_bisect = time.perf_counter() - t0

    return dict(
        t_default_s=t_default,
        t_bisect_s=t_bisect,
        bisect_overhead_factor=t_bisect / t_default if t_default > 0 else float("nan"),
        c_pos_chosen=c_pos_chosen,
        model_str=model_str,
        # Extrapolated from this PR's own real-sweep medians, scaled by the
        # *measured* per-trace overhead here -- a rough estimate, not a
        # substitute for actually running the 383-asset sweep with bisect.
        est_bright_bisect_median_s=BRIGHT_MEDIAN_S * (t_bisect / t_default if t_default > 0 else float("nan")),
        est_speedup_vs_bright_legacy=(
            BRIGHT_LEGACY_MEDIAN_S / (BRIGHT_MEDIAN_S * (t_bisect / t_default))
            if t_default > 0
            else float("nan")
        ),
    )
