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
            maxfev=10000,
        )
    except RuntimeError:
        popt, pcov = curve_fit(func, timestamps, tc, maxfev=10000)
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
    rss_thresh: float | tuple[float, float] | str = (0.98, 0.995),
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
        Factor(s) used for model selection. Default is (0.98, 0.995).
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
            f0 = baseline(timestamps, *x)
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
            plot_fit(x, trace, timestamps)

    return baseline(timestamps, *x), x


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
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Calculate dF/F of the fiber photometry signal.

    Parameters
    ----------
    tc : np.ndarray
        Fiber photometry signal.
    timestamps : np.ndarray
        Fiber photometry timestamps.
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
        if method == "bright":
            tc_fit, tc_coefs = tc_brightfit(tc_filtered, ts)
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
        tc_dFoF = np.full(tc.shape, np.nan)
        tc_fit = np.full(tc_filtered.shape, np.nan)
        tc_params = {
            i_coef: np.nan
            for i_coef in range(
                {"poly": 5, "exp": 4, "tri-exp": 7, "bright": 9}[method]
            )
        }
    # tc_qualitymetrics = {"QC_metric": np.nan}
    # tc_params.update(tc_qualitymetrics)

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


def motion_correct(
    dff: pd.DataFrame,
    fs: float = 20,
    cutoff_freq_motion: float = 0.05,
    cutoff_freq_noise: float = 3,
    M: RobustNorm = AsymmetricTukeyBiweight(2),
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
        Default is AsymmetricTukeyBiweight(2).

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
    if np.isnan(dff["Iso"]).any():
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
            model = RLM(d, add_constant(motion), M=M).fit()
            coef[i] = model.params
            w[i] = model.weights
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
    return dff_mc, dff_filt, coef, intercept, weights
