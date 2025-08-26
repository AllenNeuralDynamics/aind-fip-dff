import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
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


def tc_polyfit(
    tc: np.ndarray, sampling_rate: float, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit with polynomial to remove bleaching artifact.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    sampling_rate : float
        Sampling rate of the signal in Hz.
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
    time_seconds = np.arange(len(tc)) / sampling_rate
    coefs = np.polyfit(time_seconds, tc, deg=degree)
    tc_poly = np.polyval(coefs, time_seconds)
    return tc_poly, coefs


def tc_expfit(
    tc: np.ndarray, sampling_rate: float = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Fit with Biphasic exponential decay.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    sampling_rate : float, optional
        Sampling rate of the signal in Hz. Default is 20.

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

    time_seconds = np.arange(len(tc)) / sampling_rate
    try:  # try first providing initial estimates
        tc0 = tc[: int(sampling_rate)].mean()
        popt, pcov = curve_fit(
            func,
            time_seconds,
            tc,
            (0.9 * tc0, 1 / 3600, 0.1 * tc0, 1 / 200),
            maxfev=10000,
        )
    except RuntimeError:
        popt, pcov = curve_fit(func, time_seconds, tc, maxfev=10000)
    tc_exp = func(time_seconds, *popt)
    return tc_exp, popt


def baseline(
    t: np.ndarray,
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
    t : np.ndarray
        Time vector.
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
    T : int, optional
        Length of the trace in samples. Default is 70000.
    fs : float, optional
        Sampling rate in Hz. Default is 20.

    Returns
    -------
    np.ndarray
        Baseline signal.
    """
    return (
        b_inf
        * (
            1
            + b_slow * np.exp(-t / t_slow)
            + b_fast * np.exp(-t / t_fast)
            + b_rapid * np.exp(-t / t_rapid)
        )
        * (1 - b_bright * np.exp(-t / t_bright))
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
    t = np.arange(T) / fs
    F0 = baseline(t, *x)
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
    ax[0].plot(t, trace, label="data", c=color)
    ax[0].plot(t, F0, label="fit", c="C1")
    ax[0].set_ylabel("Trace")
    ax[0].legend()
    ax[1].plot(t, trace - F0, c=color)
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
    fs: float = 20,
    rss_thresh: float | tuple[float, float] | str = (0.98, 0.995),
    M: RobustNorm | None = TukeyBiweight(3),
    maxiter: int = 10,
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
    fs : float, optional
        Sampling rate of the signal in Hz. Default is 20.
    rss_thresh : float or tuple of float or str, optional
        Factor(s) used for model selection. Default is (0.98, 0.995).
        If a tuple, then the order is (brightening, 3rd exponential).
        A more complex model (with 2 additional parameters)
        is accepted if the RSS decreases by at least this factor.
        Automatically calculated if "AIC" or "BIC".
    M : RobustNorm or None, optional
        The robust criterion function for downweighting outliers.
        Default is TukeyBiweight(3).
    maxiter : int, optional
        The maximum number of IRLS iterations to try. Default is 10.
        Has to be >0 for robust regression, 0 uses only OLS.
    tol : float, optional
        IRLS convergence tolerance. Default is 1e-3.
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
    tuple[np.ndarray, np.ndarray]
        - Fitted baseline signal
        - Optimized parameters [b_inf, b_slow, b_fast, b_rapid, b_bright,
                                t_slow, t_fast, t_rapid, t_bright]
    """

    # constants for fancy logging
    CEND, CBOLD, CRED, CGREEN = "\33[0m", "\33[1m", "\33[31m", "\33[32m"

    # Calculate thresholds if using information criteria
    T = len(trace)
    Tds = T // 10
    if rss_thresh == "BIC":
        rss_thresh = [Tds ** (-2 / Tds)] * 2
    elif rss_thresh == "AIC":
        rss_thresh = [np.exp(-4 / Tds)] * 2

    def _optimize_baseline(
        trace: np.ndarray,
        x0: np.ndarray,
        ds: int = 1,
        max_nfev: int | None = None,
        weights: float | np.ndarray = 1,
        plot: bool = plot,
    ) -> tuple[np.ndarray, float, bool, str]:
        """Optimize baseline parameters for given trace.
        If item in x0 is set to np.nan it is not optimized but
        set to its default, i.e. this exponential term is excluded
        """
        trace_ds = trace[: T // ds * ds].reshape(-1, ds).mean(1)
        t = np.arange(ds // 2, T, ds) / fs
        optimize_param = ~np.isnan(x0)
        params = np.array([0] * 5 + [np.inf] * 4)  # default params if not optimized

        def residuals(params_to_optimize):
            params[optimize_param] = params_to_optimize
            return np.sqrt(weights) * (trace_ds - baseline(t, *params))

        res = least_squares(
            residuals,
            x0[optimize_param],
            bounds=(0, np.inf),
            xtol=1e-4,
            max_nfev=max_nfev,
        )
        params[optimize_param] = res.x
        logging.info(
            f"Cost: {res.cost:.3f}  "
            f"Success: {CGREEN if res.success else CRED} {res.success} {CEND}  "
            f"{res.message}"
        )
        if plot:
            plot_fit(params, trace, fs)
        return params, res.cost, res.success, res.message

    x0 = np.array(
        [trace[-1000:].mean(), 0.35, 0.2, np.nan, np.nan, 3600, 240, np.nan, np.nan]
    )
    logging.info(f"{CBOLD}Fit of 10x decimated trace with double-exp{CEND}")
    x2, cost2, success2, _ = _optimize_baseline(trace, x0, 10)
    if x2[6] > x2[5]:  # swap t_slow and t_fast if optimization returns t_fast > t_slow
        x2[[1, 2, 5, 6]] = x2[[2, 1, 6, 5]]

    x0[~np.isnan(x0)] = x2[~np.isnan(x0)]
    x0[[4, 8]] = 0.1, 2000
    logging.info(f"{CBOLD}Fit of 10x decimated trace with brightening{CEND}")
    xB, costB, successB, _ = _optimize_baseline(trace, x0, 10)

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
    x3, cost3, success3, _ = _optimize_baseline(trace, x0, 10)
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
    t = np.arange(T) / fs
    logging.info(
        f"Cost on original trace with params obtained on decimated trace is "
        f"{np.sum((trace - baseline(t, *params)) ** 2) / 2:.3f}"
    )
    logging.info(
        f"{CBOLD}Fit of original trace with {'triple-exp' if include_3rd else 'double-exp'} "
        f"and {'' if include_bright else 'no '}brightening{CEND}"
    )
    x, cost, success, msg = _optimize_baseline(trace, x0)

    # robust fit down-weighting outliers using IRLS
    # see https://github.com/statsmodels/statsmodels/blob/main/statsmodels/robust/robust_linear_model.py#L196
    if maxiter > 0 and M is not None and cost > 0:
        f0 = baseline(t, *x)
        resid = trace - f0
        scl = scale.mad(resid[None if skewness_factor == 0 else resid < 0], center=0)
        deviance = M(resid / scl).sum()
        iteration = 0
        converged = False
        while not converged:
            iteration += 1
            if scl == 0.0:
                logging.warning(
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
            x, cost, success, msg = _optimize_baseline(
                trace, x, weights=weights, plot=False
            )
            f0 = baseline(t, *x)
            resid = trace - f0
            if update_scale is True:
                scl = scale.mad(
                    resid[None if skewness_factor == 0 else resid < 0], center=0
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
            plot_fit(x, trace, fs)

    return baseline(t, *x), x


# dF/F total function
def chunk_processing(
    tc: np.ndarray,
    method: str = "poly",
    n_frame_to_cut: int = 100,
    kernel_size: int = 1,
    sampling_rate: float = 20,
    degree: int = 4,
    b_percentile: float = 0.7,
    robust: bool = True,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Calculate dF/F of the fiber photometry signal.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    method : str, optional
        Method to preprocess the data. Options: poly, exp, bright.
        Default is "poly".
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
        Whether to fit baseline using IRLS (robust regression, only 'bright' method).
        Default is True.

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
    tc_filtered = medfilt(tc_cropped, kernel_size=kernel_size)
    try:
        if method == "poly":
            tc_fit, tc_coefs = tc_polyfit(tc_filtered, sampling_rate, degree)
        if method == "exp":
            tc_fit, tc_coefs = tc_expfit(tc_filtered, sampling_rate)
        if method == "bright":
            tc_fit, tc_coefs = tc_brightfit(tc_filtered, sampling_rate)
            tc_dFoF = tc_filtered / tc_fit - 1
        else:
            tc_estim = tc_filtered - tc_fit
            tc_base = tc_slidingbase(tc_filtered, sampling_rate)
            tc_dFoF = tc_dFF(tc_estim, tc_base, b_percentile)
        tc_dFoF = tc_filling(tc_dFoF, n_frame_to_cut)
        tc_params = {i_coef: tc_coefs[i_coef] for i_coef in range(len(tc_coefs))}
    except Exception as e:
        logging.warning(
            f"Processing with method {method} failed with Error {e}. Setting dF/F to nans."
        )
        tc_dFoF = np.nan * tc
        tc_params = {
            i_coef: np.nan
            for i_coef in range({"poly": 5, "exp": 4, "bright": 9}[method])
        }
    tc_qualitymetrics = {"QC_metric": np.nan}
    tc_params.update(tc_qualitymetrics)

    return tc_dFoF, tc_params, tc_filling(tc_fit, n_frame_to_cut)


def motion_correct(
    dff: pd.DataFrame,
    fs: float = 20,
    cutoff_freq_motion: float = 0.05,
    cutoff_freq_noise: float = 3,
    M: RobustNorm = TukeyBiweight(3),
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
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
    M : statsmodels.robust.norms.RobustNorm, optional
        Robust criterion function used to downweight outliers.
        Default is TukeyBiweight(3).

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
    """
    if np.isnan(dff["Iso"]).any():
        c = {ch: np.nan for ch in dff.columns}
        return np.nan * dff, np.nan * dff, c, c
    sos = butter(N=2, Wn=cutoff_freq_motion, fs=fs, output="sos")
    dff_filt = sosfiltfilt(sos, dff, axis=0).T
    idx_iso = dff.columns.get_loc("Iso")
    motion = dff_filt[idx_iso]
    no_nans = ~np.isnan(dff_filt.sum(1))
    no_nans[idx_iso] = False  # skip regressing motion against motion, it's obviously 1
    if M is not None:
        coef = np.array(
            [RLM(d, add_constant(motion), M=M).fit().params for d in dff_filt[no_nans]]
        )
        intercept = coef[:, 0]
        coef = np.maximum(coef[:, 1:], 0)
    else:
        lr = LinearRegression(fit_intercept=True, positive=True).fit(
            motion[:, None], dff_filt[no_nans].T
        )
        coef = lr.coef_
        intercept = lr.intercept_
    motions = np.full_like(dff_filt, np.nan)
    motions[no_nans] = coef * dff["Iso"].values
    motions -= motions.mean(axis=1, keepdims=True)
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
    return dff_mc, dff_filt, coef, intercept
