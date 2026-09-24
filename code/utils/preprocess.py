import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aind_ophys_utils.array_utils import downsample_array
from aind_ophys_utils.baseline_fitting import (
    AsymmetricTukeyBiweight as NonlinearFitAsymmetricTukeyBiweight,
)
from aind_ophys_utils.baseline_fitting import nonlinear_fit, sum_of_exps
from aind_ophys_utils.signal_utils import noise_std
from scipy.optimize import curve_fit, minimize
from scipy.signal import butter, medfilt, sosfiltfilt
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from statsmodels.api import RLM, add_constant
from statsmodels.robust import scale
from statsmodels.robust.norms import RobustNorm, TukeyBiweight


def tc_crop(tc: np.ndarray, n_frame_to_cut: int) -> np.ndarray:
    """Remove the first few seconds of the time course."""
    return tc[n_frame_to_cut:]


def tc_slidingbase(tc: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Set up sliding baseline to calculate dF/F."""
    sos = butter(2, 0.0001, btype="low", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, tc)


def tc_dFF(tc: np.ndarray, tc_base: np.ndarray, b_percentile: float) -> np.ndarray:
    """Obtain dF/F using median of values within sliding baseline.

    Parameters
    ----------
    tc : np.ndarray
        Time course signal.
    tc_base : np.ndarray
        Baseline signal.
    b_percentile : float
        Percentile for baseline calculation.

    Returns
    -------
    np.ndarray
        dF/F signal.
    """
    tc_dFoF = tc / tc_base
    sorted_dFoF = np.sort(tc_dFoF)
    b_median = np.median(sorted_dFoF[: round(len(sorted_dFoF) * b_percentile)])
    return tc_dFoF - b_median


def tc_filling(tc: np.ndarray, n_frame_to_cut: int) -> np.ndarray:
    """Fill in the gap left by cropping out the first few timesteps."""
    return np.append(np.ones([n_frame_to_cut, 1]) * tc[0], tc)


def triple_exp(x, params):
    """
    Triple exponential function: a * exp(-b * x) + c * exp(-d * x) + e * exp(-f * x) + g
    """
    return (
        params[0] * np.exp(-params[1] * x)
        + params[2] * np.exp(-params[3] * x)
        + params[4] * np.exp(-params[5] * x)
        + params[6]
    )


def tc_triexpfit(
    tc: np.ndarray, timestamps: np.ndarray, sampling_rate: float, xtol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a triple exponential fit to the given data.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    timestamps : np.ndarray
        Fiber photometry timestamps.

    Returns
    -------
    tuple
        - tc_triexp : np.ndarray
            Fitted baseline.
        - popt : np.ndarray
            Optimal values for the parameters of the preprocessing.
    """
    # Low-pass filter
    sos = butter(2, 0.01, btype="low", fs=sampling_rate, output="sos")
    tc = sosfiltfilt(sos, tc)

    # Calculate initial parameter estimates
    fs = int(sampling_rate)  # shorthand
    # Basic statistics for initial values
    start_mean = np.mean(tc[:fs])
    end_mean = np.mean(tc[-60 * fs :])
    late_10min = np.mean(tc[-10 * 60 * fs : -10 * 60 * fs + 10 * fs])
    late_5min = np.mean(tc[-5 * 60 * fs : -5 * 60 * fs + 10 * fs])
    # intercept
    p0 = np.zeros(7)
    p0[6] = end_mean
    # Fastest decay parameters
    p0[0] = start_mean - np.mean(tc[2 * 60 * fs : 2 * 60 * fs + fs])
    tmp = 1 - (start_mean - np.mean(tc[60 * fs : 61 * fs])) / p0[0]
    p0[1] = 0.05 if tmp <= 0 else -np.log(tmp) / 60
    # Slowest decay parameters
    tmp = (late_10min - end_mean) / (late_5min - end_mean)
    p0[5] = 1 / 3600 if tmp <= 1 else np.log(tmp) / (5 * 60)
    p0[4] = (late_10min - end_mean) / np.exp(p0[5] * (-10 * 60))
    # Middle decay parameters
    p0[2] = start_mean - end_mean - p0[4]
    p0[3] = (p0[1] + p0[5]) / 2
    # Clean up invalid values
    p0 = np.maximum(0, np.nan_to_num(p0))
    params_str = ", ".join(f"{v:.5g}" for v in p0)
    logging.info(f"Initial parameters for method 'tri-exp':  {params_str}")

    # Fit curve
    popt, _ = curve_fit(
        lambda x, a, b, c, d, e, f, g: triple_exp(x, [a, b, c, d, e, f, g]),
        timestamps,
        tc,
        p0=p0,
        maxfev=10000,
        bounds=(0, np.inf),
        xtol=xtol,
        x_scale=[1, 0.0001, 1, 0.0001, 1, 0.0001, 1],
    )
    tc_triexp = triple_exp(timestamps, popt)

    # Calculate goodness-of-fit metrics
    ss_res = np.sum((tc - tc_triexp) ** 2)
    ss_tot = np.sum((tc - np.mean(tc)) ** 2)
    logging.info(
        f"R-squared: {1 - (ss_res / ss_tot):.5f}  "
        f"SS_res: {ss_res:.5g}  "
        f"SS_tot: {ss_tot:.5g}"
    )

    return tc_triexp, popt


def tc_polyfit(
    tc: np.ndarray, timestamps: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit with polynomial to remove bleaching artifact.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    timestamps : np.ndarray
        Fiber photometry timestamps.
    degree : int
        Degree of the polynomial to fit.

    Returns
    -------
    tuple
        - tc_poly : np.ndarray
            Fitted baseline.
        - coefs : np.ndarray
            Optimal values for the parameters of the preprocessing.
    """
    coefs = np.polyfit(timestamps, tc, deg=degree)
    tc_poly = np.polyval(coefs, timestamps)
    return tc_poly, coefs


def tc_expfit(tc: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit with Biphasic exponential decay.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    timestamps : np.ndarray
        Fiber photometry timestamps.

    Returns
    -------
    tuple
        - tc_exp : np.ndarray
            Fitted baseline.
        - popt : np.ndarray
            Optimal values for the parameters of the preprocessing.
    """

    def func(x, a, b, c, d):
        return a * np.exp(-b * x) + c * np.exp(-d * x)

    try:  # try first providing initial estimates
        tc0 = tc[:20].mean()
        popt, pcov = curve_fit(
            func,
            timestamps,
            tc,
            (0.9 * tc0, 1 / 3600, 0.1 * tc0, 1 / 200),
            maxfev=20000,
        )
    except RuntimeError:
        popt, pcov = curve_fit(func, timestamps, tc, maxfev=20000)
    tc_exp = func(timestamps, *popt)
    return tc_exp, popt


def baseline(
    timestamps: np.ndarray,
    b_inf: float,
    b_slow: float = 0,
    b_fast: float = 0,
    b_rapid: float = 0,
    b_bright: float = 0,
    t_slow: float = np.inf,
    t_fast: float = np.inf,
    t_rapid: float = np.inf,
    t_bright: float = np.inf,
) -> np.ndarray:
    """Baseline with Triphasic exponential decay (bleaching) x increasing saturating exponential (brightening).

    Parameters
    ----------
    timestamps : np.ndarray
        Fiber photometry timestamps.
    b_inf : float
        Asymptotic baseline value.
    b_slow : float, optional
        Amplitude of the slow decay component. Default is 0.
    b_fast : float, optional
        Amplitude of the fast decay component. Default is 0.
    b_rapid : float, optional
        Amplitude of the rapid decay component. Default is 0.
    b_bright : float, optional
        Amplitude of the brightening component. Default is 0.
    t_slow : float, optional
        Time constant of the slow decay component in seconds. Default is np.inf.
    t_fast : float, optional
        Time constant of the fast decay component in seconds. Default is np.inf.
    t_rapid : float, optional
        Time constant of the rapid decay component in seconds. Default is np.inf.
    t_bright : float, optional
        Time constant of the brightening component in seconds. Default is np.inf.

    Returns
    -------
    np.ndarray
        Baseline signal.
    """
    return (
        b_inf
        * (
            1
            + b_slow * np.exp(-timestamps / t_slow)
            + b_fast * np.exp(-timestamps / t_fast)
            + b_rapid * np.exp(-timestamps / t_rapid)
        )
        * (1 - b_bright * np.exp(-timestamps / t_bright))
    )


def plot_fit(x, trace, fs=20, title=None, color="C0"):
    """Plot the fitted baseline and residuals.

    Parameters
    ----------
    x : array-like
        Parameters for the baseline function.
    trace : np.ndarray
        Original trace data.
    fs : float, optional
        Sampling rate in Hz. Default is 20.
    title : str, optional
        Title for the plot. Default is None.
    color : str, optional
        Color for the trace. Default is "C0".

    Returns
    -------
    None
        The function displays a matplotlib figure.
    """
    T = len(trace)
    F0 = baseline(*x, T=T)
    logging.info(
        "b_inf={:9.4f}, b_slow={:6.4f}, b_fast={:6.4f}, b_rapid={:6.4f}, b_bright={:6.4f}, ".format(
            *x[:5]
        )
    )
    logging.info(
        "                 t_slow={:6.0f}, t_fast={:6.0f}, t_rapid={:6.0f}, t_bright={:6.0f}".format(
            *x[5:]
        )
    )
    fig, ax = plt.subplots(2, 1, figsize=(15, 3), sharex=True)
    ax[0].plot(np.arange(T) / fs, trace, label="data", c=color)
    ax[0].plot(np.arange(T) / fs, F0, label="fit", c="C1")
    ax[0].set_ylabel("Trace")
    ax[0].legend()
    ax[1].plot(np.arange(T) / fs, trace - F0, c=color)
    ax[1].axhline(0, c="k", ls="--")
    ax[1].set_xlabel("Time [seconds]")
    ax[1].set_ylabel("Residual")
    ax[1].set_xlim(-T / fs * 0.01, T / fs * 1.01)
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout(pad=0.4)
    plt.show()


def tc_brightfit(
    trace: np.ndarray,
    timestamps: np.ndarray,
    rss_thresh: float | tuple[float, float] | str = (0.98, 0.997),
    M: RobustNorm | None = TukeyBiweight(3),
    maxiter: int = 5,
    tol: float = 1e-3,
    update_scale: bool = True,
    skewness_factor: float = 1.0,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit trace with baseline (bleaching x brightening) using OLS or IRLS.

    More complex models that include brightening and/or a third exponential
    are only selected if they notably improve the fit by reducing the RSS.

    Parameters
    ----------
    trace : np.ndarray
        Fiber photometry signal.
    timestamps : np.ndarray
        Fiber photometry timestamps.
    rss_thresh : float or tuple of float or str, optional
        Factor(s) used for model selection. Default is (0.98, 0.997).
        If a tuple, then the order is (brightening, 3rd exponential).
        A more complex model (with 2 additional parameters)
        is accepted if the RSS decreases by at least this factor.
        Automatically calculated if "AIC" or "BIC".
    M : statsmodels.robust.norms.RobustNorm or None, optional
        The robust criterion function for downweighting outliers.
        Default is TukeyBiweight(3).
    maxiter : int, optional
        The maximum number of IRLS iterations to try. Default is 5.
        Has to be >0 for robust regression, 0 uses only OLS.
    tol : float, optional
        The convergence tolerance of the estimate. Default is 1e-3.
    update_scale : bool, optional
        If False, scale estimate for weights is held constant over iteration.
        If True, it is updated for each fit. Default is True.
    skewness_factor : float, optional
        Scaling factor to correct for bias by performing asymmetric
        robust regression based on skewness of the residuals. Default is 1.0.
    plot : bool, optional
        Whether to plot the fits. Default is False.

    Returns
    -------
    tuple
        - baseline : np.ndarray
            The fitted baseline.
        - params : np.ndarray
            Optimal values for the parameters of the preprocessing.
    """

    # constants for fancy logging
    CEND = "\33[0m"
    CBOLD = "\33[1m"
    CRED = "\33[31m"
    CGREEN = "\33[32m"

    T = len(trace)
    Tds = T // 10
    if rss_thresh == "BIC":
        rss_thresh = [Tds ** (-2 / Tds)] * 2
    elif rss_thresh == "AIC":
        rss_thresh = [np.exp(-4 / Tds)] * 2

    def optimize(trace, x0, ds=1, maxiter=20000, weights=1, plot=plot):
        """if item in x0 is set to np.nan it is not optimized but
        set to its default, i.e. this exponential term is excluded
        """
        trace_ds = trace[: T // ds * ds].reshape(-1, ds).mean(1)
        optimize_param = ~np.isnan(x0)
        params = np.array([0] * 5 + [np.inf] * 4)  # default params if not optimized

        def objective(params_to_optimize):
            params[optimize_param] = params_to_optimize
            return np.sum(
                weights
                * (
                    trace_ds
                    - baseline(timestamps[ds // 2 :: ds][: len(trace_ds)], *params)
                )
                ** 2
            )

        bounds = np.array(
            [(0, np.inf)] * 5 + [(300, np.inf), (1, 1200), (1, 180), (60, np.inf)]
        )
        bounds[2, 0] = -np.inf  # allow amplitude of fast component to be negative
        bounds[0, 0] = trace[-1000:].mean() / 10
        res = minimize(
            objective,
            np.array(x0)[optimize_param],
            bounds=bounds[optimize_param],
            method="Nelder-Mead",
            options={"maxiter": maxiter},
        )
        params[optimize_param] = res.x
        logging.info(
            f"Cost: {res.fun:.3f}  "
            f"Success: {CGREEN if res.success else CRED} {res.success} {CEND}  "
            f"{res.message}"
        )
        if plot:
            plot_fit(params, trace, timestamps)
        return params, res.fun, res.success, res.message

    x0 = np.array(
        [trace[-1000:].mean(), 0.35, 0.2, np.nan, np.nan, 3600, 240, np.nan, np.nan]
    )
    logging.info(f"{CBOLD}Fit of 10x decimated trace with double-exp{CEND}")
    x2, cost2, success2, _ = optimize(trace, x0, 10)
    if x2[6] > x2[5]:  # swap t_slow and t_fast if optimization returns t_fast > t_slow
        x2[[1, 2, 5, 6]] = x2[[2, 1, 6, 5]]

    x0[~np.isnan(x0)] = x2[~np.isnan(x0)]
    x0[[4, 8]] = 0.1, 2000
    logging.info(f"{CBOLD}Fit of 10x decimated trace with brightening{CEND}")
    xB, costB, successB, _ = optimize(trace, x0, 10, 3000)

    cost_ratio = costB / cost2
    include_bright = cost_ratio < rss_thresh[0]
    logging.info(
        f"Cost reduction by including brightening is {(cost_ratio-1)*100:.3f}%, "
        f"thus {CBOLD}{'including' if include_bright else 'skipping'}{CEND} brightening term."
        + ("\n" if plot else "")
    )
    if include_bright:
        x0[~np.isnan(x0)] = xB[~np.isnan(x0)]
    else:
        x0[[4, 8]] = np.nan
    x0[[3, 7]] = 0.1, 50
    logging.info(f"{CBOLD}Fit of 10x decimated trace with triple-exp{CEND}")
    x3, cost3, success3, _ = optimize(trace, x0, 10, 3000)
    # swap as needed to ensure t_slow > t_fast > t_rapid
    order = np.argsort(x3[5:8])[::-1]
    x3[5:8] = x3[5 + order]
    x3[1:4] = x3[1 + order]

    cost_ratio = cost3 / (costB if include_bright else cost2)
    include_3rd = cost_ratio < rss_thresh[1]
    logging.info(
        f"Cost reduction by including 3rd exponential is {(cost_ratio-1)*100:.3f}%, "
        f"thus {CBOLD}{'including' if include_3rd else 'skipping'}{CEND} 3rd exponential term."
        + ("\n" if plot else "")
    )
    if include_3rd:
        x0[~np.isnan(x0)] = x3[~np.isnan(x0)]
    else:
        x0[[3, 7]] = np.nan
    params = np.array([0] * 5 + [np.inf] * 4)
    params[~np.isnan(x0)] = x0[~np.isnan(x0)]
    logging.info(
        f"Cost on original trace with params obtained on decimated trace is "
        f"{np.sum((trace - baseline(timestamps, *params)) ** 2):.3f}"
    )
    logging.info(
        f"{CBOLD}Fit of original trace with {'triple-exp' if include_3rd else 'double-exp'} "
        f"and {'' if include_bright else 'no '}brightening{CEND}"
    )
    x, cost, success, msg = optimize(trace, x0)

    # robust fit down-weighting outliers using IRLS
    # see https://github.com/statsmodels/statsmodels/blob/main/statsmodels/robust/robust_linear_model.py#L196
    if maxiter > 0 and M is not None and cost > 0:
        f0 = baseline(timestamps, *x)
        resid = trace - f0
        scl = scale.mad(resid if skewness_factor == 0 else resid[resid < 0], center=0)
        deviance = M(resid / scl).sum()
        iteration = 0
        converged = False
        while not converged:
            iteration += 1
            if scl == 0.0:
                warnings.warn(
                    "Estimated scale is 0.0 indicating that the most"
                    " last iteration produced a perfect fit of the "
                    "weighted data."
                )
                break
            if skewness_factor != 0:
                avg_skew = np.mean(
                    [skew((resid)[t0 : t0 + 1200]) for t0 in range(0, len(resid), 1200)]
                )
                resid[resid > 0] *= np.exp(skewness_factor * avg_skew)
            weights = M.weights(resid / scl)
            x[np.isnan(x0)] = np.nan  # set params of excluded terms to nan
            x, cost, success, msg = optimize(trace, x, weights=weights, plot=False)
            f0 = baseline(timestamps, *x)
            resid = trace - f0
            if update_scale is True:
                scl = scale.mad(
                    resid if skewness_factor == 0 else resid[resid < 0], center=0
                )
            dev_pre = deviance
            deviance = M(resid / scl).sum()
            converged = iteration >= maxiter or np.abs(deviance / dev_pre - 1) < tol
        logging.info(
            f"{CBOLD}IRLS fit {iteration}/{maxiter} of original trace with "
            f"{'triple-exp' if include_3rd else 'double-exp'} "
            f"and {'' if include_bright else 'no '}brightening{CEND}"
        )
        logging.info(
            f"Cost: {cost:.3f}  Success: {CGREEN if success else CRED} {success} {CEND} {msg}"
        )
        if plot:
            plot_fit(x, trace, timestamps)

    return baseline(timestamps, *x), x


def _init_sum_of_exps(
    trace: np.ndarray, n_exp: int = 2, include_brightening: bool = False
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    """Build an initial guess and bounds for fitting `trace` with `sum_of_exps`.

    b_inf is initialized from the 10th percentile of the last 1000 frames and
    bounded below by a tenth of their mean, so the asymptote can't collapse
    to zero. Bleach amplitudes are constrained >= 0; brightening (when
    included) is a dedicated negative-amplitude term rather than a negative
    bleach amplitude. tau1's and the brightening tau's upper bounds are
    capped (not left unbounded) to avoid a b_inf/b1 identifiability
    degeneracy as tau -> infinity; see `_snap_degenerate_slow_terms` for the
    companion post-fit check.

    Parameters
    ----------
    trace : np.ndarray
        Fiber photometry signal (after cropping the initial transient).
    n_exp : int
        Number of bleaching exponential terms. 1, 2, or 3.
    include_brightening : bool
        Add a negative-amplitude exponential term for brightening.

    Returns
    -------
    x0 : np.ndarray
        Initial parameter vector for `sum_of_exps`.
    bounds : tuple of (float, float)
        Bounds for each parameter in `x0`, for `nonlinear_fit`.
    """
    TAU1_CAP = 30000.0
    TAU_BRIGHT_CAP = 20000.0
    AMP_CAP_FACTOR = 10.0

    b_inf = float(np.percentile(trace[-1000:], 10))
    b_inf_lo = float(trace[-1000:].mean() / 10)
    amp = float(trace[:500].mean() - b_inf)
    amp_cap = AMP_CAP_FACTOR * max(abs(amp), 1.0)

    amp_fracs = {1: [1.0], 2: [0.7, 0.3], 3: [0.65, 0.30, 0.05]}[n_exp]
    tau_inits = {1: [600.0], 2: [3600.0, 600.0], 3: [3600.0, 600.0, 30.0]}[n_exp]
    tau_bounds = {
        1: [(60, TAU1_CAP)],
        2: [(300, TAU1_CAP), (1, 5000)],
        3: [(300, TAU1_CAP), (1, 5000), (1, 180)],
    }[n_exp]

    x0 = [b_inf]
    bounds = [(b_inf_lo, np.inf)]
    for frac, tau, (tlo, thi) in zip(amp_fracs, tau_inits, tau_bounds):
        x0 += [amp * frac, tau]
        bounds += [(0, amp_cap), (tlo, thi)]

    if include_brightening:
        x0 += [-0.05 * b_inf, 2000.0]
        bounds += [(-amp_cap, 0), (60, TAU_BRIGHT_CAP)]

    return np.array(x0), tuple(bounds)


def _sort_bleach_params(
    params: np.ndarray, n_exp: int, include_bright: bool = False
) -> np.ndarray:
    """Sort a `sum_of_exps` parameter vector's bleach (amplitude, tau) pairs
    by tau descending, leaving b_inf and the brightening pair (if any)
    untouched."""
    p = params.copy()
    taus = np.array([params[2 + 2 * i] for i in range(n_exp)])
    for new_i, old_i in enumerate(np.argsort(taus)[::-1]):
        p[1 + 2 * new_i] = params[1 + 2 * old_i]
        p[2 + 2 * new_i] = params[2 + 2 * old_i]
    return p


def _snap_degenerate_slow_terms(
    params: np.ndarray,
    n_exp: int,
    include_bright: bool,
    tc: np.ndarray,
    ts: np.ndarray,
    near_bound_frac: float = 0.95,
) -> tuple[np.ndarray, int, bool, list[str]]:
    """Drop the slowest bleach term and/or the brightening term if either
    landed at (or near) its tau upper bound, and refit without it.

    A term whose tau sits at its cap is not a real slow process -- it is
    `_init_sum_of_exps`'s b_inf/tau1 (or b_inf/tau_bright) degeneracy, where
    an arbitrary amount of b_inf gets misattributed to a physically
    meaningless "slow exponential" instead of the bound preventing it
    outright. Only the slowest bleach term is checked, since the other
    bleach terms' tighter bounds reflect genuine fast timescales rather than
    this degeneracy.

    Returns
    -------
    params, n_exp, include_bright : the resulting (possibly reduced) model,
        refit if anything was dropped; unchanged otherwise.
    snapped : list of str
        Which terms were dropped ("tau1", "bright"); empty if none.
    """
    TAU1_CAP, TAU_BRIGHT_CAP = 30000.0, 20000.0  # must match _init_sum_of_exps
    snapped = []
    if params[2] >= near_bound_frac * TAU1_CAP:
        snapped.append("tau1")
    if include_bright and params[-1] >= near_bound_frac * TAU_BRIGHT_CAP:
        snapped.append("bright")
    if not snapped:
        return params, n_exp, include_bright, snapped

    new_n_exp = n_exp - (1 if "tau1" in snapped else 0)
    new_bright = include_bright and "bright" not in snapped
    if new_n_exp < 1:
        return np.array([float(np.mean(tc))]), 0, False, snapped

    x0, bnd = _init_sum_of_exps(tc, n_exp=new_n_exp, include_brightening=new_bright)
    _, res = nonlinear_fit(tc, ts, model=sum_of_exps, x0=x0, bounds=bnd, M=None)
    return (
        _sort_bleach_params(res.x, new_n_exp, new_bright),
        new_n_exp,
        new_bright,
        snapped,
    )


def _pad_sum_of_exps_params(
    params: np.ndarray, n_exp: int, include_bright: bool
) -> np.ndarray:
    """Pad a variable-length `sum_of_exps` parameter vector to the fixed
    9-slot layout [b_inf, b1, tau1, b2, tau2, b3, tau3, b_bright, tau_bright],
    filling unused bleach/brightening terms with (amplitude=0, tau=inf) so
    every trace's fitted-parameter table has the same shape regardless of
    which model was selected for that trace.
    """
    out = np.array([0.0, 0.0, np.inf, 0.0, np.inf, 0.0, np.inf, 0.0, np.inf])
    out[0] = params[0]
    out[1 : 1 + 2 * n_exp] = params[1 : 1 + 2 * n_exp]
    if include_bright:
        out[7:9] = params[-2:]
    return out


#: Default M-estimator for `tc_brightfit_v2`'s dF/F fit. Kept as a distinct
#: name (not just "M") from `motion_correct`'s M_MOTION_CORRECTION default,
#: since the two tune unrelated fits and are easy to conflate.
M_DFF = NonlinearFitAsymmetricTukeyBiweight(c_pos=3.5, c_neg=4.0)


def tc_brightfit_v2(
    trace: np.ndarray,
    timestamps: np.ndarray,
    M: RobustNorm | None = M_DFF,
    rss_thresh: tuple[float, float] = (0.98, 0.97),
    maxiter: int = 5,
    ds: int = 10,
    fixed_sigma: float | str | None = "auto",
    sigma_anneal_steps: int = 4,
    t_eval_exp3: float = 120.0,
    correction: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit trace with a sum-of-exponentials baseline (bleaching, optionally
    with a negative-amplitude brightening term) via `aind_ophys_utils`'s
    `nonlinear_fit`.

    Model selection cold-starts at 2 bleach exponentials on a `ds`-times
    decimated trace, then tries adding a brightening term and a 3rd bleach
    exponential (each kept only if it improves the RSS by `rss_thresh`), and
    finally refits the winning model on the full-resolution trace with
    robust IRLS (M-estimator `M`), warm-started from the decimated winner.
    See `_snap_degenerate_slow_terms` for a post-fit check that drops any
    term whose time constant landed at its upper bound.

    Any sample where `trace` or `timestamps` is non-finite is dropped (not
    interpolated) before fitting -- a single corrupted sample can otherwise
    poison the IRLS loss to NaN everywhere. The returned baseline is NaN at
    those positions and finite elsewhere; a `UserWarning` reports how many
    samples were dropped, and a `ValueError` is raised if none are usable.

    Parameters
    ----------
    trace : np.ndarray
        Fiber photometry signal.
    timestamps : np.ndarray
        Fiber photometry timestamps.
    M : statsmodels.robust.norms.RobustNorm or None, optional
        The robust criterion function for the final IRLS fit. Default is
        `M_DFF` (AsymmetricTukeyBiweight(c_pos=3.5, c_neg=4.0) from
        `aind_ophys_utils.baseline_fitting`).
    rss_thresh : tuple of float, optional
        RSS-ratio thresholds (brightening, 3rd exponential) for accepting
        each more complex candidate model. Default is (0.98, 0.97).
    maxiter : int, optional
        Maximum number of IRLS iterations for the final fit. Default is 5.
    ds : int, optional
        Decimation factor used for model-selection candidates; the final
        fit always runs on the full-resolution trace. Default is 10.
    fixed_sigma : float or "auto" or None, optional
        IRLS noise scale for the final fit. "auto" estimates it via
        `noise_std(trace, method="welch")` -- the PSD-based estimate is
        symmetric and unbiased since it's computed from a high-frequency
        band with no calcium-transient energy, unlike `method="mad"`, which
        strips positive residuals first and so underestimates sigma from
        the negative/central tail alone. A float uses `fixed_sigma`
        directly; None falls back to `nonlinear_fit`'s own adaptive-scale
        IRLS. Default is "auto".
    sigma_anneal_steps : int, optional
        Geometric IRLS-sigma annealing steps passed to `nonlinear_fit`.
        Default is 4.
    t_eval_exp3 : float, optional
        Length (seconds) of the early window used to compare the 3rd
        exponential candidate against the current winner; brightening is
        compared on the full trace. Default is 120.0.
    correction : {"median", "pct70"} or None, optional
        Optional per-trace, post-hoc centering correction applied to the
        fitted baseline (`trace - baseline` is the residual in both cases):
          - None (default): no correction.
          - "median": shift the baseline by the plain median of the
            residuals. Exactly zero-centered on clean data, robust up to
            50% one-sided contamination (calcium activity only ever pushes
            residuals positive) by construction.
          - "pct70": shift the baseline by the median of the lowest 70% of
            residuals -- the same recipe `poly`/`exp`/`tri-exp` already use
            in production (`tc_dFF`, via `b_percentile`), applied here to
            this method's residual instead of `tc_dFF`'s ratio. Robust up
            to 30% contamination by construction (guaranteed, since that
            fraction is dropped before taking the median), at the cost of a
            small deliberate offset even on clean data (the 35th percentile
            of a symmetric residual isn't its center) -- a different
            tradeoff from "median", not a strictly better or worse one.
        Any other value raises `ValueError`.

    Returns
    -------
    tuple
        - baseline : np.ndarray
            The fitted baseline.
        - params : np.ndarray
            Fitted parameters, padded to
            [b_inf, b1, tau1, b2, tau2, b3, tau3, b_bright, tau_bright]
            (unused terms set to amplitude=0, tau=inf).
    """
    # Guard against non-finite samples: drop (not interpolate) any frame
    # where the trace or its timestamp is non-finite, before anything else
    # touches them. A single corrupted sample can otherwise poison the
    # IRLS loss to NaN everywhere, collapsing the fit to its parameter
    # bounds with no usable gradient.
    valid = np.isfinite(trace) & np.isfinite(timestamps)
    n_dropped = int((~valid).sum())
    if n_dropped:
        if not valid.any():
            raise ValueError(
                "tc_brightfit_v2: every sample is non-finite (trace or "
                "timestamps) -- nothing to fit."
            )
        warnings.warn(
            f"tc_brightfit_v2: dropping {n_dropped} non-finite sample(s) "
            "(trace or timestamps) before fitting.",
            stacklevel=2,
        )
        trace_valid = trace[valid]
        ts_valid = timestamps[valid]
    else:
        trace_valid = trace
        ts_valid = timestamps

    if fixed_sigma == "auto":
        fixed_sigma = float(noise_std(trace_valid, method="welch"))

    tc_ds = downsample_array(trace_valid, factors=ds, strategy="first")
    ts_ds = downsample_array(ts_valid, factors=ds, strategy="first")

    dt_ds = float(ts_ds[1] - ts_ds[0]) if len(ts_ds) > 1 else float(ds)
    n_eval_ds = min(len(tc_ds), round(t_eval_exp3 / dt_ds))

    kw_ds = dict(model=sum_of_exps, M=None)
    kw_full = dict(
        model=sum_of_exps, M=M, maxiter=maxiter, sigma_anneal_steps=sigma_anneal_steps
    )
    if fixed_sigma is not None:
        kw_full["fixed_sigma"] = fixed_sigma

    def _rss_ds_full(f0):
        return float(np.sum((tc_ds - f0) ** 2))

    def _rss_ds_early(f0):
        return float(np.sum((tc_ds[:n_eval_ds] - f0[:n_eval_ds]) ** 2))

    # Step 1: 2-exp OLS cold start on the decimated trace.
    x0, bnd = _init_sum_of_exps(tc_ds, n_exp=2, include_brightening=False)
    f0_ds, res_ds = nonlinear_fit(tc_ds, ts_ds, x0=x0, bounds=bnd, **kw_ds)
    rss_ds = _rss_ds_full(f0_ds)
    f0_ds_best = f0_ds
    params_ds = _sort_bleach_params(res_ds.x, 2)
    n_exp_won = 2
    include_bright = False

    # Step 2: try adding brightening (full decimated-trace RSS).
    x0_b = np.concatenate([params_ds, [-0.05 * params_ds[0], 2000.0]])
    _, bnd_b = _init_sum_of_exps(tc_ds, n_exp=n_exp_won, include_brightening=True)
    f0_b_ds, res_b_ds = nonlinear_fit(tc_ds, ts_ds, x0=x0_b, bounds=bnd_b, **kw_ds)
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
    new_exp = np.array([0.05 * params_ds[0], 50.0])
    x0_3 = (
        np.concatenate([params_ds[:-2], new_exp, params_ds[-2:]])
        if include_bright
        else np.concatenate([params_ds, new_exp])
    )
    _, bnd_3 = _init_sum_of_exps(
        tc_ds, n_exp=n_exp_next, include_brightening=include_bright
    )
    f0_3_ds, res_3_ds = nonlinear_fit(tc_ds, ts_ds, x0=x0_3, bounds=bnd_3, **kw_ds)
    p3_sorted = _sort_bleach_params(res_3_ds.x, n_exp_next, include_bright)
    tau_new = p3_sorted[2 + 2 * n_exp_won]
    if (
        _rss_ds_early(f0_3_ds) < rss_thresh[1] * _rss_ds_early(f0_ds_best)
        and tau_new < 180.0
    ):
        params_ds = p3_sorted
        n_exp_won = n_exp_next
    model_str = f"{n_exp_won}-exp{'+bright' if include_bright else ''}"

    # Drop any term whose time constant landed at its upper bound and refit.
    params_ds, n_exp_won, include_bright, snapped = _snap_degenerate_slow_terms(
        params_ds, n_exp_won, include_bright, tc_ds, ts_ds
    )
    if snapped:
        model_str += f" [snapped: {','.join(snapped)}]"

    # Step 4: final full-resolution IRLS fit, warm-started from the decimated winner.
    if n_exp_won == 0:
        f0 = np.full_like(trace_valid, params_ds[0])
    else:
        _, bnd_full = _init_sum_of_exps(
            trace_valid, n_exp=n_exp_won, include_brightening=include_bright
        )
        f0, _ = nonlinear_fit(
            trace_valid, ts_valid, x0=params_ds, bounds=bnd_full, **kw_full
        )

    if correction == "median":
        f0 = f0 + np.median(trace_valid - f0)
    elif correction == "pct70":
        r_sorted = np.sort(trace_valid - f0)
        f0 = f0 + np.median(r_sorted[: round(len(r_sorted) * 0.7)])
    elif correction is not None:
        raise ValueError(
            f"tc_brightfit_v2: unknown correction {correction!r}; "
            'expected None, "median", or "pct70".'
        )

    logging.info(f"Fit of original trace with model selection: {model_str}")

    if n_dropped:
        f0_full = np.full(len(trace), np.nan)
        f0_full[valid] = f0
        f0 = f0_full

    return f0, _pad_sum_of_exps_params(params_ds, n_exp_won, include_bright)


# dF/F total function
def chunk_processing(
    tc: np.ndarray,
    timestamps: np.ndarray,
    method: str = "poly",
    n_frame_to_cut: int = 100,
    kernel_size: int = 1,
    sampling_rate: float = 20,
    degree: int = 4,
    b_percentile: float = 0.7,
    robust: bool = True,
    correction: str | None = None,
    M: RobustNorm | None = None,
    trace_id: str = "",
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Calculate dF/F of the fiber photometry signal.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    timestamps : np.ndarray
        Fiber photometry timestamps.
    method : str, optional
        Method to preprocess the data. Options: poly, exp, tri-exp, bright,
        bright_legacy. Default is "poly".
    n_frame_to_cut : int, optional
        Number of frames to crop from the beginning of the signal.
        Default is 100.
    kernel_size : int, optional
        Size of the kernel for median filtering. Default is 1.
    sampling_rate : float, optional
        Sampling rate of the signal in Hz. Default is 20.
    degree : int, optional
        Degree of the polynomial to fit. Default is 4.
    b_percentile : float, optional
        Percentile to calculate the baseline. Default is 0.7.
    robust : bool, optional
        Whether to fit baseline using IRLS (robust regression, only
        'bright_legacy' method). Default is True.
    correction : {"median", "pct70"} or None, optional
        Optional per-trace centering correction (only 'bright' method, see
        `tc_brightfit_v2`). Default is None (no correction).
    M : RobustNorm or None, optional
        Optional M-estimator override for the 'bright' method's IRLS fit,
        passed straight through to `tc_brightfit_v2` (only 'bright' method;
        no effect on any other method). Default is None, which leaves
        `tc_brightfit_v2` on its own default (`M_DFF`,
        `AsymmetricTukeyBiweight(c_pos=3.5, c_neg=4.0)`).
    trace_id : str, optional
        Trace identifier for logging purposes, e.g. 'G_0', default is ''.

    Returns
    -------
    tuple
        - tc_dFoF : np.ndarray
            dF/F of fiber photometry signal.
        - tc_params : dict
            Dictionary with the parameters of the preprocessing.
        - tc_fit_filled : np.ndarray
            The fitted baseline, including the filled beginning portion.
    """
    tc_cropped = tc_crop(tc, n_frame_to_cut)
    ts = tc_crop(timestamps, n_frame_to_cut)
    tc_filtered = medfilt(tc_cropped, kernel_size=kernel_size)
    try:
        if method == "poly":
            tc_fit, tc_coefs = tc_polyfit(tc_filtered, ts, degree)
        elif method == "exp":
            tc_fit, tc_coefs = tc_expfit(tc_filtered, ts)
        elif method == "tri-exp":
            try:
                tc_fit, tc_coefs = tc_triexpfit(
                    tc_filtered, ts, sampling_rate, xtol=1e-5
                )
            except RuntimeError:
                tc_fit, tc_coefs = tc_triexpfit(
                    tc_filtered, ts, sampling_rate, xtol=1e-4
                )
        elif method == "bright_legacy":
            tc_fit, tc_coefs = tc_brightfit(tc_filtered, ts)
        elif method == "bright":
            # Only pass M through when explicitly overridden -- omitting the
            # kwarg (rather than passing M=None) lets tc_brightfit_v2's own
            # default (M_DFF) apply exactly as before when no override is
            # given, avoiding any ambiguity about what an explicit M=None
            # means at that level.
            brightfit_kwargs = {"correction": correction}
            if M is not None:
                brightfit_kwargs["M"] = M
            tc_fit, tc_coefs = tc_brightfit_v2(
                tc_filtered, ts, **brightfit_kwargs
            )

        if method in ("bright", "bright_legacy"):
            tc_dFoF = tc_filtered / tc_fit - 1
        else:
            tc_estim = tc_filtered - tc_fit
            tc_base = tc_slidingbase(tc_filtered, sampling_rate)
            tc_dFoF = tc_dFF(tc_estim, tc_base, b_percentile)
        tc_dFoF = tc_filling(tc_dFoF, n_frame_to_cut)
        tc_params = {i_coef: tc_coefs[i_coef] for i_coef in range(len(tc_coefs))}
    except Exception as e:
        logging.warning(
            f"Processing {trace_id} with method {method} failed with Error {e}. Setting dF/F to nans."
        )
        tc_dFoF = np.full(tc.shape, np.nan)
        tc_fit = np.full(tc_filtered.shape, np.nan)
        tc_params = {
            i_coef: np.nan
            for i_coef in range(
                {
                    "poly": 5,
                    "exp": 4,
                    "tri-exp": 7,
                    "bright": 9,
                    "bright_legacy": 9,
                }[method]
            )
        }

    return tc_dFoF, tc_params, tc_filling(tc_fit, n_frame_to_cut)


class OneSidedHuber(RobustNorm):
    """
    One-sided Huber norm for robust regression.

    This norm applies standard quadratic loss to residuals less than or equal to
    the threshold value (z ≤ c), and a linear loss to residuals greater than the
    threshold (z > c). This makes the estimator robust against positive outliers
    while treating negative residuals as in ordinary least squares.

    Parameters
    ----------
    c : float, optional
        Threshold parameter that controls the transition from quadratic to linear
        loss. Default is 1.345, which gives 95% efficiency under the normal
        distribution (same as statsmodels HuberT).
    """

    def __init__(self, c=1.345):  # default same as statsmodels HuberT
        self.c = c

    def rho(self, z):
        # Loss function
        return np.where(z <= self.c, 0.5 * z**2, self.c * (z - 0.5 * self.c))

    def psi(self, z):
        # Influence function
        return np.where(z <= self.c, z, self.c)

    def weights(self, z):
        # Weights for IRLS
        return np.where(z <= self.c, 1.0, self.c / z)

    def psi_deriv(self, z):
        # Derivative of influence function
        return np.where(z <= self.c, 1.0, 0.0)


class AsymmetricTukeyBiweight(RobustNorm):
    """
    Asymmetric Tukey Biweight norm for robust regression.

    Allows different tuning constants for positive and negative residuals,
    providing more flexibility in handling asymmetric outliers.

    Parameters
    ----------
    c_pos : float, optional
        Tuning constant for positive residuals, default is 4.685
    c_neg : float, optional
        Tuning constant for negative residuals, default is 4.685
    """

    def __init__(self, c_pos=4.685, c_neg=4.685):
        if c_pos <= 0 or c_neg <= 0:
            raise ValueError("Tuning constants must be positive")
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.factor_pos = c_pos**2 / 6
        self.factor_neg = c_neg**2 / 6

    def rho(self, z):
        z = np.asarray(z)
        res = np.empty_like(z)
        # Handle positive side
        pos_mask = z > 0
        if np.isinf(self.c_pos):
            res[pos_mask] = 0.5 * z[pos_mask] ** 2
        else:
            pos_inside = pos_mask & (z <= self.c_pos)
            pos_outside = z > self.c_pos
            res[pos_inside] = self.factor_pos * (
                1 - (1 - (z[pos_inside] / self.c_pos) ** 2) ** 3
            )
            res[pos_outside] = self.factor_pos
        # Handle negative side
        neg_mask = z <= 0
        if np.isinf(self.c_neg):
            res[neg_mask] = 0.5 * z[neg_mask] ** 2
        else:
            neg_inside = neg_mask & (z >= -self.c_neg)
            neg_outside = z < -self.c_neg
            res[neg_inside] = self.factor_neg * (
                1 - (1 - (z[neg_inside] / self.c_neg) ** 2) ** 3
            )
            res[neg_outside] = self.factor_neg

        return res

    def psi(self, z):
        z = np.asarray(z)
        res = np.zeros_like(z)
        pos_inside = (z > 0) & (z <= self.c_pos)
        neg_inside = (z <= 0) & (z >= -self.c_neg)
        res[pos_inside] = z[pos_inside] * (1 - (z[pos_inside] / self.c_pos) ** 2) ** 2
        res[neg_inside] = z[neg_inside] * (1 - (z[neg_inside] / self.c_neg) ** 2) ** 2
        return res

    def weights(self, z):
        z = np.asarray(z)
        res = np.zeros_like(z)
        pos_inside = (z > 0) & (z <= self.c_pos)
        neg_inside = (z <= 0) & (z >= -self.c_neg)
        res[pos_inside] = (1 - (z[pos_inside] / self.c_pos) ** 2) ** 2
        res[neg_inside] = (1 - (z[neg_inside] / self.c_neg) ** 2) ** 2
        return res

    def psi_deriv(self, z):
        z = np.asarray(z)
        res = np.zeros_like(z)
        pos_inside = (z > 0) & (z <= self.c_pos)
        neg_inside = (z <= 0) & (z >= -self.c_neg)
        t_pos = z[pos_inside] / self.c_pos
        t_pos_sq = t_pos**2
        res[pos_inside] = (1 - t_pos_sq) ** 2 - 4 * t_pos_sq * (
            1 - t_pos_sq
        ) / self.c_pos**2
        t_neg = z[neg_inside] / self.c_neg
        t_neg_sq = t_neg**2
        res[neg_inside] = (1 - t_neg_sq) ** 2 - 4 * t_neg_sq * (
            1 - t_neg_sq
        ) / self.c_neg**2
        return res


class OneSidedTukeyBiweight(AsymmetricTukeyBiweight):
    """
    A one-sided Tukey Biweight norm that applies quadratic loss to negative
    residuals and Tukey biweight loss to positive residuals.

    This is implemented as a special case of AsymmetricTukeyBiweight
    with c_neg=np.inf, which simplifies to quadratic loss for negative values.
    """

    def __init__(self, c=4.685):
        super().__init__(c_pos=c, c_neg=np.inf)


#: Default M-estimator for `motion_correct`'s regression. Kept as a distinct
#: name (not just "M") from `tc_brightfit_v2`'s M_DFF default, since the two
#: tune unrelated fits and are easy to conflate.
M_MOTION_CORRECTION = AsymmetricTukeyBiweight(c_pos=3, c_neg=4.0)


def motion_correct(
    dff: pd.DataFrame,
    fs: float = 20,
    cutoff_freq_motion: float = 0.05,
    cutoff_freq_noise: float = 3,
    M: RobustNorm = M_MOTION_CORRECTION,
    mode: str = "demean",
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Perform motion correction on fiber's dF/F traces by regressing out isosbestic traces.

    Parameters
    ----------
    dff : pd.DataFrame
        DataFrame containing the dF/F traces of the fiber photometry signals.
    fs : float, optional
        Sampling rate of the signal in Hz. Default is 20.
    cutoff_freq_motion : float, optional
        Cutoff frequency of the lowpass Butterworth filter that's only
        applied for estimating the regression coefficient, in Hz.
        Default is 0.05.
    cutoff_freq_noise : float, optional
        Cutoff frequency of the lowpass Butterworth filter
        that's applied to filter out noise, in Hz.
        Default is 3.
    M : RobustNorm, optional
        Robust criterion function used to downweight outliers.
        Default is M_MOTION_CORRECTION (AsymmetricTukeyBiweight(c_pos=3, c_neg=4.0)).
    mode : str, optional
        How the fitted regression is applied to each channel:
        "demean" subtracts `coef * raw_Iso`, re-centered to zero mean, and
        discards the fitted intercept -- a channel's own F0-fitting bias
        passes straight through unchanged. "intercept" additionally
        subtracts the fitted intercept, which also removes each channel's
        own constant F0-fitting bias but relies on that channel having no
        genuine tonic (constant, non-transient) signal of interest,
        since regression cannot distinguish the two. Default is "demean".

    Returns
    -------
    tuple
        - dff_mc : pd.DataFrame
            Preprocessed fiber photometry signal with motion correction applied.
        - dff_filt : pd.DataFrame
            Low-pass filtered dF/F fiber photometry signal.
        - coeffs : dict
            The regression coefficients.
        - intercepts : dict
            The regression intercepts.
        - weights : dict
            The final regression weights.
    """
    if mode not in ("demean", "intercept"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'demean' or 'intercept'.")
    if np.isnan(dff["Iso"]).any() or np.isinf(dff["Iso"]).any():
        c = {ch: np.nan for ch in dff.columns}
        return np.nan * dff, np.nan * dff, c, c, c
    sos = butter(N=2, Wn=cutoff_freq_motion, fs=fs, output="sos")
    dff_filt = sosfiltfilt(sos, dff, axis=0).T
    idx_iso = dff.columns.get_loc("Iso")
    motion = dff_filt[idx_iso]
    no_nans = ~np.isnan(dff_filt.sum(1))
    no_nans[idx_iso] = False  # skip regressing motion against motion, it's obviously 1
    if M is not None:
        coef = np.empty((no_nans.sum(), 2))
        w = np.empty((no_nans.sum(), len(motion)))
        for i, d in enumerate(dff_filt[no_nans]):
            rlm_result = RLM(d, add_constant(motion), M=M).fit()
            coef[i] = rlm_result.params
            w[i] = (
                rlm_result.weights
                if hasattr(rlm_result.model, "weights")
                else np.ones(len(d))
            )
        intercept = np.array(coef)[:, 0]
        coef = np.maximum(coef[:, 1:], 0)
    else:
        lr = LinearRegression(fit_intercept=True, positive=True).fit(
            motion[:, None], dff_filt[no_nans].T
        )
        coef = lr.coef_
        intercept = lr.intercept_
        w = np.ones((no_nans.sum(), len(motion)))
    weights = np.full_like(dff_filt, np.nan)
    weights[no_nans] = w
    weights[idx_iso] = 1
    weights = {ch: w for ch, w in zip(dff.columns, weights)}
    motions = np.full_like(dff_filt, np.nan)
    motions[no_nans] = coef * dff["Iso"].values
    if mode == "demean":
        motions -= motions.mean(axis=1, keepdims=True)
    else:  # mode == "intercept"
        motions[no_nans] += intercept[:, None]
    dff_mc = dff - motions.T
    dff_mc["Iso"] = 0
    dff_filt = pd.DataFrame(dff_filt.T)
    dff_filt.columns = dff_mc.columns
    c = np.full(len(motions), np.nan)
    c[no_nans] = coef.ravel()
    c[idx_iso] = 1
    coef = {ch: c_ for ch, c_ in zip(dff.columns, c)}
    c[no_nans] = intercept
    c[idx_iso] = 0
    intercept = {ch: c_ for ch, c_ in zip(dff.columns, c)}
    if cutoff_freq_noise is not None and cutoff_freq_noise < fs / 2:
        sos = butter(N=2, Wn=cutoff_freq_noise, fs=fs, output="sos")
        dff_mc = dff_mc.apply(lambda x: sosfiltfilt(sos, x))
    return dff_mc, dff_filt, coef, intercept, weights
