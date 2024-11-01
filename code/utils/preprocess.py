import glob
import itertools
import os
import re

import numpy as np
import pandas as pd
from aind_ophys_utils.signal_utils import noise_std
from hdmf_zarr.nwb import NWBZarrIO
from pynwb import NWBHDF5IO
from scipy.optimize import curve_fit, minimize
from scipy.signal import butter, filtfilt, medfilt, sosfilt
from sklearn.linear_model import LinearRegression
from statsmodels.api import RLM
from statsmodels.robust import scale
from statsmodels.robust.norms import HuberT, TukeyBiweight


# Preprocessing functions
# ---------------------------------------------------------------------------------------------
# removing first few seconds
def tc_crop(tc, nFrame2cut):
    tc_cropped = tc[nFrame2cut:]
    return tc_cropped


# Median filtering to remove electrical artifact.
def tc_medfilt(tc, kernelSize):
    tc_filtered = medfilt(tc, kernel_size=kernelSize)
    return tc_filtered


# Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
def tc_lowcut(tc, sampling_rate):
    b, a = butter(2, 9, btype="low", fs=sampling_rate)
    tc_filtered = filtfilt(b, a, tc)
    return tc_filtered


# setting up sliding baseline to calculate dF/F
def tc_slidingbase(tc, sampling_rate):
    b, a = butter(2, 0.0001, btype="low", fs=sampling_rate)
    tc_base = filtfilt(b, a, tc, padtype="even")
    return tc_base


# obtain dF/F using median of values within sliding baseline
def tc_dFF(tc, tc_base, b_percentile):
    tc_dFoF = tc / tc_base
    sort = np.sort(tc_dFoF)
    b_median = np.median(sort[0 : round(len(sort) * b_percentile)])
    tc_dFoF = tc_dFoF - b_median
    return tc_dFoF


# fill in the gap left by cropping out the first few timesteps
def tc_filling(tc, nFrame2cut):
    tc_filled = np.append(np.ones([nFrame2cut, 1]) * tc[0], tc)
    return tc_filled


def tc_polyfit(tc, sampling_rate, degree):
    """
    Fit with polynomial to remove bleaching artifact
    Args:
        tc: np.array
            Fiber photometry signal
        sampling_rate: float
            Sampling rate of the signal
    Returns:
        tc_dFoF: np.array
            Preprocessed fiber photometry signal
        popt: array
            Optimal values for the parameters of the preprocessing
    """
    time_seconds = np.arange(len(tc)) / sampling_rate
    coefs = np.polyfit(time_seconds, tc, deg=degree)
    tc_poly = np.polyval(coefs, time_seconds)
    return tc_poly, coefs


def tc_expfit(tc, sampling_rate=20):
    """
    Fit with Biphasic exponential decay
    Args:
        tc: np.array
            Fiber photometry signal
        sampling_rate: float
            Sampling rate of the signal
    Returns:
        tc_dFoF: np.array
            Preprocessed fiber photometry signal
        popt: array
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


def tc_brightfit(tc, sampling_rate=20, robust=True):
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
    b_inf,
    b_slow=0,
    b_fast=0,
    b_bright=0,
    t_slow=np.inf,
    t_fast=np.inf,
    t_bright=np.inf,
    T=70000,
    fs=20,
):
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


def fit_trace(trace, fs=20):
    """
    Oridinary Least Squares (OLS) fit using above baseline (bleaching x brightening)
    Args:
        trace: np.array
            Fiber photometry signal
        fs: float
            Sampling rate of the signal
    Returns:
        x: array
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
    trace,
    fs=20,
    M=TukeyBiweight(2),
    maxiter=50,
    tol=1e-5,
    update_scale=True,
    asymmetric=True,
    scale_est="welch",
):
    """
    Iteratively Reweighted Least Squares (IRLS) fit using above baseline (bleaching x brightening)
    see https://github.com/statsmodels/statsmodels/blob/main/statsmodels/robust/robust_linear_model.py#L196
    Args:
        trace: np.array
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
        params: array
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
    tc,
    method="poly",
    nFrame2cut=100,
    kernelSize=1,
    sampling_rate=20,
    degree=4,
    b_percentile=0.7,
    robust=True,
):
    """
    Calculates dF/F of the fiber photometry signal.
    Args:
        tc: np.array
            Fiber photometry signal
        method: str
            Method to preprocess the data. Options: poly, exp, bright
        nFrame2cut: int
            Number of frames to crop from the beginning of the signal
        kernelSize: int
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
        tc_dFoF: np.array
            dF/F of fiber photometry signal
        tc_params: dict
            Dictionary with the parameters of the preprocessing
    """
    tc_cropped = tc_crop(tc, nFrame2cut)
    tc_filtered = medfilt(tc_cropped, kernel_size=kernelSize)
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
        tc_dFoF = tc_filling(tc_dFoF, nFrame2cut)
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


def motion_correct(dff, fs=20, M=TukeyBiweight(1)):
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
    sos = butter(N=2, Wn=0.3, fs=fs, output="sos")
    dff_filt = sosfilt(sos, dff, axis=0).T
    motion = dff_filt[dff.columns.get_loc("Iso")]
    if M is not None:
        motion = (
            np.maximum([RLM(d, motion, M=M).fit().params for d in dff_filt], 0) * motion
        ).T
    else:
        motion = (
            LinearRegression(fit_intercept=False, positive=True)
            .fit(motion[:, None], dff_filt.T)
            .predict(motion[:, None])
        )
    motion -= motion.mean(0)
    dff_mc = dff - motion
    return dff_mc


# run the total preprocessing (dF/F + motion correction) on multiple sessions
# -- future iteration: collect exceptions in a log file
def batch_processing(df_fip, methods=["poly", "exp", "bright"]):
    """
    Preprocesses the fiber photometry signal (dF/F + motion correction).
    Args:
        df_fib: pd.DataFrame
            Fiber photometry signal
        methods: list of str
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
            # dF/F
            for i_iter, (channel, fiber_number, session) in enumerate(
                itertools.product(channels, fiber_numbers, sessions)
            ):
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
                df_fip_pp = pd.concat([df_fip_pp, df_fip_iter], axis=0)

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
            for i_iter, (fiber_number, session) in enumerate(
                itertools.product(fiber_numbers, sessions)
            ):
                df_fip_iter = df_fip_pp[
                    (df_fip_pp["session"] == session)
                    & (df_fip_pp["fiber_number"] == fiber_number)
                    & (df_fip_pp["preprocess"] == pp_name)
                ].copy()
                if len(df_fip_iter) == 0:
                    continue

                # convert to #frames x #channels
                df_dff_iter = pd.DataFrame(
                    np.column_stack(
                        [
                            df_fip_iter[df_fip_iter["channel"] == c]["signal"].values
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


# Below are obsolete Processing Functions that used the NPM system instead of NWB
# ---------------------------------------------------------------------------------------------
# Function to create the input to the batch processing function
def load_Homebrew_fip_data(filenames, fibers_per_file=2):
    """
    This function loops over the filenames for the channels
    in the NPM system 'L415', 'L470', 'L560'
    The created dataframe has the following fields:
        - session
        - time
        - signal
        - fiber_number
        - channel
        - excitation
        - camera
        - system
    """

    df_fip = pd.DataFrame()
    # df_data_acquisition = pd.DataFrame()
    save_fip_channels = np.arange(1, fibers_per_file + 1)
    for filename in filenames:
        subject_id, session_date, session_time = (
            re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename)
            .group()
            .split("_")
        )
        session_name = subject_id + "_" + session_date + "_" + session_time
        header = os.path.basename(filename).split("/")[-1]
        channel = ("_".join(header.split("_")[:2])).replace("FIP_Data", "")
        try:
            df_fip_file = pd.read_csv(filename, header=None)  # read the CSV file
        except pd.errors.EmptyDataError:
            continue
        except FileNotFoundError:
            continue
        df_file = pd.DataFrame()
        for col in df_fip_file.columns[save_fip_channels]:
            df_fip_file_renamed = df_fip_file[[0, col]].rename(
                columns={0: "time_fip", col: "signal"}
            )
            channel_number = int(col)
            df_fip_file_renamed["fiber_number"] = channel_number
            df_fip_file_renamed.loc[:, "frame_number"] = df_fip_file.index.values
            df_file = pd.concat([df_file, df_fip_file_renamed])
            # df_data_acquisition = pd.concat([df_data_acquisition, pd.DataFrame({'session':ses_idx, 'system':'FIP', channel+str(channel_number):1.,'N_files':len(filenames)}, index=[0])])
        df_file["channel"] = channel
        camera = {"Iso": "G", "G": "G", "R": "R"}[channel]
        excitation = {"Iso": 415, "G": 470, "R": 560}[channel]
        df_file["excitation"] = excitation
        df_file["camera"] = camera
        df_fip = pd.concat([df_fip, df_file], axis=0)

    if len(df_fip) > 0:
        df_fip["system"] = "FIP"
        df_fip["preprocess"] = "None"
        df_fip["session"] = subject_id + "_" + session_date + "_" + session_time
        df_fip_ses = df_fip.loc[
            :,
            [
                "session",
                "frame_number",
                "time_fip",
                "signal",
                "channel",
                "fiber_number",
                "excitation",
                "camera",
                "system",
                "preprocess",
            ],
        ]
    else:
        df_fip_ses = df_fip
    return df_fip_ses


# Function to get the preprocessed (pp) dataframe without the nwb generation
# -- used to check if the new method is working
def gen_pp_df_old_version(AnalDir="../trial_data/700708_2024-06-14_08-38-31/"):

    # define the files with the traces from each of the channels
    filenames = []
    for name in ["FIP_DataG", "FIP_DataR", "FIP_DataIso"]:
        if (
            bool(
                glob.glob(AnalDir + os.sep + "**" + os.sep + name + "*", recursive=True)
            )
            == True
        ):
            filenames.extend(
                glob.glob(AnalDir + os.sep + "**" + os.sep + name + "*", recursive=True)
            )

    # create the df for input to the batch preprocessing function and then preprocess it
    df_fip_ses = load_Homebrew_fip_data(filenames=filenames)
    df_fip_pp, df_PP_params = batch_processing(df_fip=df_fip_ses)

    return df_fip_ses, df_fip_pp, df_PP_params
