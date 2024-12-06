import itertools

import numpy as np
import pandas as pd
from aind_ophys_utils.signal_utils import noise_std
from scipy.optimize import curve_fit, minimize
from scipy.signal import butter, medfilt, sosfiltfilt
from sklearn.linear_model import LinearRegression
from statsmodels.api import RLM
from statsmodels.robust import scale
from statsmodels.robust.norms import RobustNorm, TukeyBiweight


def tc_crop(tc: np.ndarray, n_frame_to_cut: int) -> np.ndarray:
    """Remove the first few seconds of the time course."""
    return tc[n_frame_to_cut:]


def tc_medfilt(tc: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply median filtering to remove electrical artifact."""
    return medfilt(tc, kernel_size=kernel_size)


def tc_lowcut(tc: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Apply lowpass filter with zero phase filtering to avoid distorting the signal."""
    sos = butter(2, 9, btype="low", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, tc)


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
        tc_dFoF: np.ndarray
            Preprocessed fiber photometry signal
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
        tc_dFoF: np.ndarray
            Preprocessed fiber photometry signal
        popt: np.ndarray
            Optimal values for the parameters of the preprocessing
    """

    def func(x, a, b, c, d):
        return a * np.exp(-b * x) + c * np.exp(-d * x)

    time_seconds = np.arange(len(tc)) / sampling_rate
    try:  # try first providing initial estimates
        tc0 = tc[: int(sampling_rate)].mean()
        popt, pcov = curve_fit(
            func, time_seconds, tc, (0.9 * tc0, 1 / 3600, 0.1 * tc0, 1 / 200)
        )
    except:
        popt, pcov = curve_fit(func, time_seconds, tc)
    tc_exp = func(time_seconds, *popt)
    return tc_exp, popt


def tc_brightfit(
    tc: np.ndarray, sampling_rate: float = 20, robust: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit with  Biphasic exponential decay (bleaching)  x  increasing saturating exponential (brightening)
    Args:
        tc: np.array
            Fiber photometry signal
        sampling_rate: float
            Sampling rate of the signal
        robust: bool
            Whether to fit baseline using IRLS (robust regression)
    Returns:
        tc_dFoF: np.array
            Preprocessed fiber photometry signal
        popt: array
            Optimal values for the parameters of the preprocessing
    """
    popt = (
        fit_trace_robust(tc, sampling_rate) if robust else fit_trace(tc, sampling_rate)
    )
    return baseline(*popt, T=len(tc)), popt


def baseline(
    b_inf: float,
    b_slow: float = 0,
    b_fast: float = 0,
    b_bright: float = 0,
    t_slow: float = np.inf,
    t_fast: float = np.inf,
    t_bright: float = np.inf,
    T: int = 70000,
    fs: float = 20,
) -> np.ndarray:
    """Baseline with  Biphasic exponential decay (bleaching)  x  increasing saturating exponential (brightening)"""
    tmp = -np.arange(T)
    return (
        b_inf
        * (
            1
            + b_slow * np.exp(tmp / (t_slow * fs))
            + b_fast * np.exp(tmp / (t_fast * fs))
        )
        * (1 - b_bright * np.exp(tmp / (t_bright * fs)))
    )


def fit_trace(trace: np.ndarray, fs: float = 20):
    """
    Oridinary Least Squares (OLS) fit using above baseline (bleaching x brightening)
    Args:
        trace: np.ndarray
            Fiber photometry signal
        fs: float
            Sampling rate of the signal
    Returns:
        x: np.ndarray
            Optimal values for the parameters of the preprocessing
    """

    def optimize(trace, x0, ds=1, maxiter=20000):
        T = len(trace)
        trace_ds = trace[: T // ds * ds].reshape(-1, ds).mean(1)

        def objective(params):
            return np.sum(
                (trace_ds - baseline(*params, T=len(trace_ds), fs=fs / ds)) ** 2
            )

        return minimize(
            objective,
            x0,
            bounds=[
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (1, np.inf),
                (1 / ds, np.inf),
                (1, np.inf),
            ],
            method="Nelder-Mead",
            options={"maxiter": maxiter},
        )

    # optimize on decimated data to quickly get good initial estimates
    res100 = optimize(
        trace, (trace[-1000:].mean(), 0.35, 0.2, 0.25, 3600.0, 200.0, 2000.0), 100, 2000
    )
    res10 = optimize(trace, res100.x, 10, 1000)
    # optimize on full data
    res = optimize(trace, res10.x)
    x = res.x
    if np.allclose(x[3], 0):  # no brightening
        x[-1] = np.inf
        x[3] = 0
    if np.allclose(x[2], 0):  # no fast decay
        x[-2] = np.inf
        x[2] = 0
    if x[-2] > x[-3]:  # swap t_slow and t_fast if optimization returns t_fast > t_slow
        x[1], x[2], x[-2], x[-3] = x[2], x[1], x[-3], x[-2]
    return x


def fit_trace_robust(
    trace: np.ndarray,
    fs: float = 20,
    M=TukeyBiweight(2),
    maxiter: int = 50,
    tol: float = 1e-5,
    update_scale: bool = True,
    asymmetric: bool = True,
    scale_est: str = "welch",
) -> np.ndarray:
    """
    Iteratively Reweighted Least Squares (IRLS) fit using above baseline (bleaching x brightening)
    see https://github.com/statsmodels/statsmodels/blob/main/statsmodels/robust/robust_linear_model.py#L196
    Args:
        trace: np.ndarray
            Fiber photometry signal
        fs: float
            Sampling rate of the signal
        M : statsmodels.robust.norms.RobustNorm
            The robust criterion function for downweighting outliers.
            See statsmodels.robust.norms for more information.
        maxiter : int
            The maximum number of iterations to try.
        tol : float
            The convergence tolerance of the estimate.
        update_scale : bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.
        asymmetric : bool
            If `asymmetric` is True then only positive outliers are reweighted.
        scale_est : str
            'mad' or 'welch'
            Indicates the estimate to use for scaling the weights in the IRLS.
    Returns:
        params: np.ndarray
            Optimal values for the parameters of the preprocessing
    """

    def optimize_robust(trace, x0, weights):
        T = len(trace)

        def objective(params):
            return np.sum(weights * (trace - baseline(*params, T=T, fs=fs)) ** 2)

        return minimize(
            objective,
            x0,
            bounds=[
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (1, np.inf),
                (1, np.inf),
                (1, np.inf),
            ],
        )

    # init with OLS fit
    params = fit_trace(trace)
    f0 = baseline(*params, T=len(trace), fs=fs)
    resid = trace - f0
    scl = (
        scale.mad(resid, center=0)
        if scale_est == "mad"
        else noise_std(resid, method="welch")
    )
    deviance = M(resid / scl).sum()

    iteration = 1
    converged = False
    while not converged:
        if scl == 0.0:
            import warnings

            warnings.warn(
                "Estimated scale is 0.0 indicating that the most"
                " last iteration produced a perfect fit of the "
                "weighted data."
            )
            break
        weights = M.weights(resid / scl)
        if asymmetric:
            weights[resid < 0] = 1
        wls_results = optimize_robust(trace, params, weights)
        params = wls_results.x
        f0 = baseline(*params, T=len(trace), fs=fs)
        resid = trace - f0
        if update_scale is True:
            scl = (
                scale.mad(resid, center=0)
                if scale_est == "mad"
                else noise_std(resid, method="welch")
            )
        dev_pre = deviance
        deviance = M(resid / scl).sum()
        iteration += 1
        converged = iteration >= maxiter or np.abs(deviance - dev_pre) < tol

    return params


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
        tc_dFoF: np.ndarray
            dF/F of fiber photometry signal
        tc_params: dict
            Dictionary with the parameters of the preprocessing
    """
    tc_cropped = tc_crop(tc, n_frame_to_cut)
    tc_filtered = medfilt(tc_cropped, kernel_size=kernel_size)
    tc_filtered = tc_lowcut(tc_filtered, sampling_rate)
    try:
        if method == "poly":
            tc_fit, tc_coefs = tc_polyfit(tc_filtered, sampling_rate, degree)
        if method == "exp":
            tc_fit, tc_coefs = tc_expfit(tc_filtered, sampling_rate)
        if method == "bright":
            tc_fit, tc_coefs = tc_brightfit(tc_filtered, sampling_rate, robust)
            tc_dFoF = tc_filtered / tc_fit - 1
        else:
            tc_estim = tc_filtered - tc_fit
            tc_base = tc_slidingbase(tc_filtered, sampling_rate)
            tc_dFoF = tc_dFF(tc_estim, tc_base, b_percentile)
        tc_dFoF = tc_filling(tc_dFoF, n_frame_to_cut)
        tc_params = {i_coef: tc_coefs[i_coef] for i_coef in range(len(tc_coefs))}
    except:
        print(f"Processing with method {method} failed. Setting dF/F to nans.")
        tc_dFoF = np.nan * tc
        tc_params = {
            i_coef: np.nan
            for i_coef in range({"poly": 5, "exp": 4, "bright": 7}[method])
        }
    tc_qualitymetrics = {"QC_metric": np.nan}
    tc_params.update(tc_qualitymetrics)

    return tc_dFoF, tc_params


def motion_correct(
    dff: pd.DataFrame, fs: float = 20, M: RobustNorm = TukeyBiweight(1)
) -> pd.DataFrame:
    """
    Perform motion correction on a fiber's dF/F traces by regressing out
    the filtered isosbestic traces.
    Args:
        dff: pd.DataFrame
            DataFrame containing the dF/F traces of the fiber photometry signals.
        fs: float
            Sampling rate of the signal, in Hz.
        M: statsmodels.robust.norms.RobustNorm
            Robust criterion function used to downweight outliers.
            Refer to `statsmodels.robust.norms` for more details.
    Returns:
        dff_mc : pd.DataFrame
            Preprocessed fiber photometry signal with motion correction applied
            (dF/F + motion correction).
    """
    if np.isnan(dff["Iso"]).any():
        return np.nan * dff
    sos = butter(N=2, Wn=0.3, fs=fs, output="sos")
    dff_filt = sosfiltfilt(sos, dff, axis=0).T
    motion = dff_filt[dff.columns.get_loc("Iso")]
    motions = np.nan * dff_filt.T
    no_nans = ~np.isnan(dff_filt.sum(1))
    if M is not None:
        motions[:, no_nans] = (
            np.maximum([RLM(d, motion, M=M).fit().params for d in dff_filt[no_nans]], 0)
            * motion
        ).T
    else:
        motions[:, no_nans] = (
            LinearRegression(fit_intercept=False, positive=True)
            .fit(motion[:, None], dff_filt[no_nans].T)
            .predict(motion[:, None])
        )
    motions -= motions.mean(0)
    dff_mc = dff - motions
    return dff_mc


def batch_processing(
    df_fip: pd.DataFrame, methods: list[str] = ["poly", "exp", "bright"]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the fiber photometry signal (dF/F + motion correction).
    Args:
        df_fib: pd.DataFrame
            Fiber photometry signal
        methods: list[str]
            Methods to preprocess the data. Options: poly, exp, bright
    Returns:
        df_fip_pp: pd.DataFrame
            dF/F of fiber photometry signal
        df_pp_params: pd.DataFrame
            Dataframe with the parameters of the preprocessing
        df_fip_mc: pd.DataFrame
            Preprocessed (dF/F + motion correction) of fiber photometry signal
    """
    df_fip_pp = pd.DataFrame()
    df_pp_params = pd.DataFrame()
    df_mc = pd.DataFrame()

    if len(df_fip) == 0:
        return df_fip, df_pp_params

    sessions = pd.unique(df_fip["session"].values)
    sessions = sessions[~pd.isna(sessions)]
    fiber_numbers = np.unique(df_fip["fiber_number"].values)
    channels = pd.unique(df_fip["channel"])  # ['G', 'R', 'Iso']
    channels = channels[~pd.isna(channels)]
    for pp_name in methods:
        if pp_name in ["poly", "exp", "bright"]:
            for i_iter, (session, fiber_number) in enumerate(
                itertools.product(sessions, fiber_numbers)
            ):
                # dF/F
                df_1fiber = pd.DataFrame()
                for channel in channels:
                    df_fip_iter = df_fip[
                        (df_fip["session"] == session)
                        & (df_fip["fiber_number"] == fiber_number)
                        & (df_fip["channel"] == channel)
                    ].copy()
                    if len(df_fip_iter) == 0:
                        continue

                    NM_values = df_fip_iter["signal"].values
                    NM_preprocessed, NM_fitting_params = chunk_processing(
                        NM_values, method=pp_name
                    )
                    df_fip_iter.loc[:, "signal"] = NM_preprocessed
                    df_fip_iter.loc[:, "preprocess"] = pp_name
                    df_fip_pp = pd.concat([df_fip_pp, df_fip_iter], ignore_index=True)
                    df_1fiber = pd.concat([df_1fiber, df_fip_iter], ignore_index=True)

                    NM_fitting_params.update(
                        {
                            "preprocess": pp_name,
                            "channel": channel,
                            "fiber_number": fiber_number,
                            "session": session,
                        }
                    )
                    df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
                    df_pp_params = pd.concat([df_pp_params, df_pp_params_ses], axis=0)

                # motion correction
                if len(df_1fiber) == 0:
                    continue
                # convert to #frames x #channels
                df_dff_iter = pd.DataFrame(
                    np.column_stack(
                        [
                            df_1fiber[df_1fiber["channel"] == c]["signal"].values
                            for c in channels
                        ]
                    ),
                    columns=channels,
                )
                # run motion correction
                df_mc_iter = motion_correct(df_dff_iter)
                # convert back to a table with columns channel and signal
                df_mc_iter = df_mc_iter.melt(var_name="channel", value_name="signal")
                df_mc = pd.concat([df_mc, df_mc_iter], ignore_index=True)
    df_fip_mc = df_fip_pp.copy()
    df_fip_mc["signal"] = df_mc["signal"]

    return df_fip_pp, df_pp_params, df_fip_mc
