# aind-fip-dff

This capsule processes input NWB files containing raw fiber photometry data by generating baseline-corrected (ΔF/F) and motion-corrected traces, which are then appended back to the NWB file.

## Method
### Baseline (F<sub>0</sub>) Estimation Methods for Calculating ΔF/F
The following baseline estimation methods are implemented (the default is `poly`, `exp` and `bright-v2`; `tri-exp` and `bright` are opt-in via `--dff_methods`):
- "poly": Fits a 4th-order polynomial using Ordinary Least Squares (OLS).
- "exp": Fits a biphasic exponential using OLS: $a\cdot e^{-b\cdot t} + c\cdot e^{-d\cdot t}$
- "tri-exp": Fits a triphasic exponential using OLS: $a\cdot e^{-b\cdot t} + c\cdot e^{-d\cdot t} + e\cdot e^{-f\cdot t} + g$
- "bright": Robust fit with  [Bi- or Tri-phasic exponential decay (bleaching)]  x  [Increasing saturating exponential (brightening)] using Iteratively Reweighted Least Squares (IRLS):  $b_{\infty} \cdot (1 + b_{slow} e^{-t/\tau_{slow}} + b_{fast} e^{-t/\tau_{fast}} + b_{rapid} e^{-t/\tau_{rapid}}) \cdot (1-b_{bright} e^{-t/\tau_{bright}}))$  
  Fitting starts with a biphasic exponential, and the brightening and/or 3rd exponential are only included if they substantially improve the fit.
- "bright-v2": Robust fit with a sum of exponentials, $b_{\infty} + \sum_{i=1}^{3} b_i e^{-t/\tau_i} + b_{bright} e^{-t/\tau_{bright}}$ (bleaching terms with an optional negative-amplitude brightening term), selected on a decimated trace and refit at full resolution with an asymmetric Tukey biweight M-estimator ($c_{pos}=c_{neg}=4$) via `aind_ophys_utils.nonlinear_fit`. The scale of the weights is a MAD-based noise estimate (`noise_std(method="mad")`), annealed in from the OLS residuals over 6 steps. The fits take about 50x less compute than "bright"; on a long session the whole capsule ran in about 3 min instead of 19 min with "bright" (where the slowest trace dominates the wall time).

For "poly", "exp" and "tri-exp" the OLS fit is pulled up by transients, so ΔF/F is computed as F / fit − r, where r is the median of the lowest `b_percentile` fraction of the ratio F / fit (`--b_percentile`, default 0.7, i.e. roughly the 35th percentile of all values; `mode` subtracts the half-sample mode of the ratio instead). A low percentile sits below the true baseline, so ΔF/F can carry a small positive offset (about 0.3 % on one example session), which the pre-GoCue drift, baseline level and calibration QC metrics below make visible. "bright" and "bright-v2" fit the baseline robustly and ignore `--b_percentile`.

### Motion Correction
Motion correction is carried out in three to four steps:
1. **Motion filtering**: A second-order Butterworth filter is applied to smooth the traces (parameter: `cutoff_freq_motion`).
2. **Estimating motion attenuation**: Robust regression is used to estimate the coefficients for each filtered trace.
3. **Removing the motion component**: The (unfiltered) isosbestic motion signal, scaled by the corresponding coefficients, is subtracted from all channels.
4. **Noise filtering**: Optionally, a second-order Butterworth filter is applied to filter out noise (parameter: `cutoff_freq_noise`).


## Input

`code/run_capsule.py` processes the one attached asset (`python run_capsule.py [parameters]`); the parameters are defined with argparse in `main()` (the input and output paths keep their defaults and are not App Panel parameters):
- `--dff_methods`: Method(s) to be used for calculating ΔF/F. Default: `poly exp bright-v2`
- `--b_percentile`: Reference level of "poly", "exp" and "tri-exp": a fraction in (0, 1] (median of the lowest fraction of the ratio F / fit) or `mode`. No effect on "bright" and "bright-v2". Default: `0.7`
- `--cutoff_freq_motion`: Cutoff frequency of the lowpass Butterworth filter that's only applied for estimating the regression coefficient, in Hz. Default: `0.05`
- `--cutoff_freq_noise`: Cutoff frequency of the lowpass Butterworth filter that's applied to filter out noise, in Hz. If `None` or greater than the Nyquist frequency (`10` for `20`Hz sampling rate) no noise filtering is performed. Default: `3`
- `--serial`: Do not parallelize fibers and channels.
- `--no_qc`: Skip the QC plots and metrics.

## Running

**Code Ocean App Panel.** The parameters are passed to `code/run` as named flags (`--name=value`), so their order in the panel does not matter. `--dff_methods` takes one quoted string (e.g. `"poly exp bright-v2"`), `--serial` and `--no_qc` take `true` or `false`. The `entrypoint` line at the top of `code/run` selects the script: `run_capsule.py` (default, single asset) or `reprocess.py` (one-off batch run over many attached assets; `--serial` then has no effect, as `reprocess.py` parallelizes across assets itself).

**Reprocessing many assets.** `code/reprocess.py` redoes the ΔF/F step for already processed assets, e.g. `python reprocess.py --source_pattern "/data/*/nwb/*.nwb" --output-dir /results --dff_methods poly exp bright-v2`. It takes the same method and filter parameters as `run_capsule.py` (`--parallel` opts in to parallelizing fibers and channels within an asset). Each asset gets its own folder `<output-dir>/<asset name>/` with the copied metadata, the updated NWB file (`fib.nwb.zarr`), `processing.json`, `quality_control.json` and `dff-qc`. If an asset fails, its partial output is removed, the other assets are unaffected, and the run ends with a non-zero exit code and a list of the failed assets (the single-asset case keeps its partial output).

## Output

The primary output is the updated NWB file, which includes the newly processed data:
- **Baseline-corrected traces (ΔF/F)**: Stored as `[Channel]_[ROI]_dff-[method]` (e.g., `G_1_dff-poly`). 
- **Fully preprocessed traces (ΔF/F with motion correction)**: Stored as `[Channel]_[ROI]_dff-[method]_mc-[method]` (e.g., `G_1_dff-poly_mc-iso-IRLS`). 

Besides the NWB file, the output directory contains:
- `processing.json`: the processing record with this step appended.
- `data_description.json`: the derived data description, upgraded from the input asset's metadata. If the upgrade fails (e.g. a `funding_source` entry with fields the schema does not allow), the run fails and the log names the file; the metadata has to be fixed. A missing `data_description.json` is only logged.
- `dff_timing.csv`: fit time of every method, ROI and channel.

For quality control (QC), the `dff-qc` subdirectory contains visualizations of the processing steps for each fiber and ΔF/F method.   
- The ΔF/F figures (e.g., `ROI0_dff-bright-v2.png`) display raw traces with fitted baseline (F<sub>0</sub>) and ΔF/F traces of all channels.   
- The motion-correction figures (e.g., `ROI0_dff-bright-v2_mc-iso-IRLS.png`) display, for each non-isosbestic channel:
  * On the left:
    - The ΔF/F trace of the regressed and low-pass filtered isosbestic ΔF/F traces (the estimated motion component)
    - Low-pass filtered ΔF/F traces of the color and isosbestic channels
    - The motion corrected, and optionally noise filtered, ΔF/F trace
  * In the middle: The corresponding power spectral densities with motion and (if applicable) noise cutoff frequencies indicated as dashed vertical lines
  * On the right: A scatter plot of color vs isosbestic ΔF/F values for both the original and low-pass filtered (`cutoff_freq_motion`) data with the fitted regression line
- The pre-event regression figures (e.g., `ROI0_dff-bright-v2_pregocue-regression.png`, and `..._pregocue-regression_mc-iso-IRLS.png` after motion correction) plot, per channel, the mean ΔF/F of each trial's pre-event window against time with an OLS trend line, a check for baseline drift within the session. The window is each trial's delay period before the GoCue, or, for sessions without one, the second before the reward.

Within the `dff-qc` subdirectory is also the `quality_control.json` for the [aind-qc-portal](https://github.com/AllenNeuralDynamics/aind-qc-portal). For every ROI and method it holds these metrics, each with an automatic status (the worst channel wins; values between the pass and fail limits stay Pending):

| Metric | What it checks | Pass | Fail |
|---|---|---|---|
| Calibration ratio | Median negative residual of the fit relative to what Gaussian noise (MAD estimate) would give, at the ΔF/F stage; about 1 for a well-placed baseline | 0.9 – 1.4 | < 0.8 or > 2.5 |
| Baseline level | Mode of the 1 Hz low-passed ΔF/F in the first, middle and last 5 min; 0 if F<sub>0</sub> follows the baseline, negative (positive) if F<sub>0</sub> is too high (low). Needs ≥ 90 % baseline samples and a recording of at least 1000 s | worst &#124;level&#124; ≤ 1 % | > 5 % |
| Motion correction | Regression coefficient and noise inflation (noise of the motion-corrected over the uncorrected trace) of each non-isosbestic channel | < 1.1 | > 2.0 |
| Pre-GoCue ΔF/F drift | Mean pre-event ΔF/F and its total linear drift across trials, before and after motion correction; needs ≥ 20 trials | both &#124;.&#124; < 1 % | either &#124;.&#124; > 5 % |

The baseline-correction parameters of each fit are stored as additional (Pending) metrics.
