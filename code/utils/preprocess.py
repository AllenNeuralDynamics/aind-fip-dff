import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """Obtain dF/F using median of values within sliding baseline."""
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
    """
    Fit with polynomial to remove bleaching artifact
    Args:
        tc: np.ndarray
            Fiber photometry signal
        sampling_rate: float
            Sampling rate of the signal
    Returns:
        tc_F0: np.ndarray
            Fitted baseline
        popt: np.ndarray
            Optimal values for the parameters of the preprocessing
    """
    time_seconds = np.arange(len(tc)) / sampling_rate
    coefs = np.polyfit(time_seconds, tc, deg=degree)
    tc_poly = np.polyval(coefs, time_seconds)
    return tc_poly, coefs


def tc_expfit(
    tc: np.ndarray, sampling_rate: float = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit with Biphasic exponential decay
    Args:
        tc: np.ndarray
            Fiber photometry signal
        sampling_rate: float
            Sampling rate of the signal
    Returns:
        tc_F0: np.ndarray
            Fitted baseline
        popt: np.ndarray
            Optimal values for the parameters of the preprocessing
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
    b_inf: float,
    b_slow: float = 0,
    b_fast: float = 0,
    b_rapid: float = 0,
    b_bright: float = 0,
    t_slow: float = np.inf,
    t_fast: float = np.inf,
    t_rapid: float = np.inf,
    t_bright: float = np.inf,
    T: int = 70000,
    fs: float = 20,
) -> np.ndarray:
    """Baseline with  Triphasic exponential decay (bleaching)
    x  increasing saturating exponential (brightening)"""
    tmp = -np.arange(T)
    return (
        b_inf
        * (
            1
            + b_slow * np.exp(tmp / (t_slow * fs))
            + b_fast * np.exp(tmp / (t_fast * fs))
            + b_rapid * np.exp(tmp / (t_rapid * fs))
        )
        * (1 - b_bright * np.exp(tmp / (t_bright * fs)))
    )


def plot_fit(x, trace, fs=20, title=None, color="C0"):
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
    fs: float = 20,
    rss_thresh: float | tuple[float, float] | str = (0.98, 0.995),
    M: RobustNorm | None = TukeyBiweight(3),
    maxiter: int = 10,
    tol: float = 1e-3,
    update_scale: bool = True,
    skewness_factor: float = 1.0,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit trace with above baseline (bleaching x brightening) using Ordinary
    Least Squares (OLS) or Iteratively Reweighted Least Squares (IRLS).
    More complex models that include brightening and/or a third exponential
    are only selected if they notably improve the fit by reducing the RSS.
    Args:
        trace: np.ndarray
            Fiber photometry signal
        fs: float
            Sampling rate of the signal
        rss_thresh: float, or (float, float), or str
            Factor(s) used for model selection.
            If a list then the order is (brightening, 3rd exponential)
            A more complex model (with 2 additional parameters)
            is accepted if the RSS decreases by at least this factor.
            Automatically calculated if "AIC" or "BIC".
        M : statsmodels.robust.norms.RobustNorm
            The robust criterion function for downweighting outliers.
            See statsmodels.robust.norms for more information.
        maxiter : int
            The maximum number of IRLS iterations to try.
            Has to be >0 for robust regression, 0 uses only OLS.
        tol : float
            The convergence tolerance of the estimate.
        skewness_factor : float
            Scaling factor to correct for bias by performing asymmetric
            robust regression based on skewness of the residuals.
        update_scale : bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.
    Returns:
        tc_dFoF: np.array
            Preprocessed fiber photometry signal
        popt: array
            Optimal values for the parameters of the preprocessing
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
                weights * (trace_ds - baseline(*params, T=T // ds, fs=fs / ds)) ** 2
            )

        bounds = np.array(
            [(0, np.inf)] * 5 + [(300, np.inf), (1, 1200), (1, 180), (60, np.inf)]
        )
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
            plot_fit(params, trace, fs)
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
        f"{np.sum((trace - baseline(*params, T=len(trace), fs=fs)) ** 2):.3f}"
    )
    logging.info(
        f"{CBOLD}Fit of original trace with {'triple-exp' if include_3rd else 'double-exp'} "
        f"and {'' if include_bright else 'no '}brightening{CEND}"
    )
    x, cost, success, msg = optimize(trace, x0)

    # robust fit down-weighting outliers using IRLS
    # see https://github.com/statsmodels/statsmodels/blob/main/statsmodels/robust/robust_linear_model.py#L196
    if maxiter > 0 and M is not None and cost > 0:
        f0 = baseline(*x, T=T, fs=fs)
        resid = trace - f0
        scl = scale.mad(resid[None if skewness_factor == 0 else resid < 0], center=0)
        deviance = M(resid / scl).sum()
        iteration = 0
        converged = False
        while not converged:
            iteration += 1
            if scl == 0.0:
                import warnings

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
            f0 = baseline(*x, T=T, fs=fs)
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

    return baseline(*x, T=T, fs=fs), x


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
) -> tuple[np.ndarray, dict]:
    """
    Calculates dF/F of the fiber photometry signal.
    Args:
        tc: np.ndarray
            Fiber photometry signal
        method: str
            Method to preprocess the data. Options: poly, exp, bright
        n_frame_to_cut: int
            Number of frames to crop from the beginning of the signal
        kernel_size: int
            Size of the kernel for median filtering
        sampling_rate: float
            Sampling rate of the signal
        degree: int
            Degree of the polynomial to fit
        b_percentile: float
            Percentile to calculate the baseline
        robust: bool
            Whether to fit baseline using IRLS (robust regression, only 'bright' method)
    Returns:
        tc_F0: np.ndarray
            dF/F of fiber photometry signal
        tc_params: dict
            Dictionary with the parameters of the preprocessing
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
) -> pd.DataFrame:
    """
    Perform motion correction on a fiber's dF/F traces by regressing out
    the isosbestic traces.
    Args:
        dff: pd.DataFrame
            DataFrame containing the dF/F traces of the fiber photometry signals.
        fs: float
            Sampling rate of the signal, in Hz.
        cutoff_freq_motion: float
            Cutoff frequency of the lowpass Butterworth filter that's only
            applied for estimating the regression coefficient, in Hz.
        cutoff_freq_noise: float
            Cutoff frequency of the lowpass Butterworth filter
            that's applied to filter out noise, in Hz.
        M: statsmodels.robust.norms.RobustNorm
            Robust criterion function used to downweight outliers.
            Refer to `statsmodels.robust.norms` for more details.
    Returns:
        dff_mc: pd.DataFrame
            Preprocessed fiber photometry signal with motion correction applied
            (dF/F + motion correction).
        dff_filt: pd.DataFrame
            Low-pass filtered dF/F fiber photometry signal.
        coeffs: dict
            The regression coefficients.
        intercepts: dict
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
