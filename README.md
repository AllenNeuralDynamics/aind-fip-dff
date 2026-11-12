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
- "bright-v2": Robust fit with a sum of exponentials, $b_{\infty} + \sum_{i=1}^{3} b_i e^{-t/\tau_i} + b_{bright} e^{-t/\tau_{bright}}$ (bleaching terms with an optional negative-amplitude brightening term), selected on a decimated trace and refit at full resolution with an asymmetric Tukey biweight M-estimator ($c_{pos}=c_{neg}=4$) via `aind_ophys_utils.nonlinear_fit`. The scale of the weights is a MAD-based noise estimate (`noise_std(method="mad")`), annealed in from the OLS residuals over 6 steps. Much faster than "bright" (about 50x less fit compute).

For "poly", "exp" and "tri-exp", ΔF/F = F / fit − r, where r is the median of the lowest `b_percentile` fraction of F / fit (default 0.7, about the 35th percentile; `mode` uses the half-sample mode). Such a low reference can leave a small positive offset in ΔF/F, which the QC metrics below show. "bright" and "bright-v2" ignore `--b_percentile`.

### Motion Correction
Motion correction is carried out in three to four steps:
1. **Motion filtering**: A second-order Butterworth filter is applied to smooth the traces (parameter: `cutoff_freq_motion`).
2. **Estimating motion attenuation**: Robust regression is used to estimate the coefficients for each filtered trace.
3. **Removing the motion component**: The (unfiltered) isosbestic motion signal, scaled by the corresponding coefficients, is subtracted from all channels.
4. **Noise filtering**: Optionally, a second-order Butterworth filter is applied to filter out noise (parameter: `cutoff_freq_noise`).


## Input

`code/run_capsule.py` processes the attached asset. Parameters (argparse in `main()`; the input and output paths keep their defaults and are not App Panel parameters):
- `--dff_methods`: Method(s) for calculating ΔF/F (see Method).
- `--b_percentile`: Reference level of "poly", "exp" and "tri-exp" (see Method): a fraction in (0, 1] or `mode`. Default: `0.7`
- `--cutoff_freq_motion`: Cutoff frequency of the lowpass Butterworth filter that's only applied for estimating the regression coefficient, in Hz. Default: `0.05`
- `--cutoff_freq_noise`: Cutoff frequency of the lowpass Butterworth filter that's applied to filter out noise, in Hz. If `None` or greater than the Nyquist frequency (`10` for `20`Hz sampling rate) no noise filtering is performed. Default: `3`
- `--serial`: Do not parallelize fibers and channels.
- `--no_qc`: Skip the QC plots and metrics.

In the Code Ocean App Panel the parameters are passed to `code/run` as named flags (`--name=value`). Its `entrypoint` line switches between `run_capsule.py` (default) and `reprocess.py`, a one-off batch run over many attached assets that redoes the ΔF/F step with the same method and filter parameters, plus `--source_pattern` and `--output-dir`. It writes one folder per asset; a failing asset's partial output is removed and the asset is listed, the others are kept, and the exit code is non-zero.

## Output

The primary output is the updated NWB file, `fiber.nwb.zarr` in the output directory, which includes the newly processed data:
- **Baseline-corrected traces (ΔF/F)**: Stored as `[Channel]_[ROI]_dff-[method]` (e.g., `G_1_dff-poly`). 
- **Fully preprocessed traces (ΔF/F with motion correction)**: Stored as `[Channel]_[ROI]_dff-[method]_mc-[method]` (e.g., `G_1_dff-poly_mc-iso-IRLS`). 

Besides the NWB file, the output directory contains `processing.json` and `dff_timing.csv`. The capsule does not write a `data_description.json`.

For quality control (QC), the `dff-qc` subdirectory contains visualizations of the processing steps for each fiber and ΔF/F method.   
- The ΔF/F figures (e.g., `ROI0_dff-bright-v2.png`) display raw traces with fitted baseline (F<sub>0</sub>) and ΔF/F traces of all channels.   
- The motion-correction figures (e.g., `ROI0_dff-bright-v2_mc-iso-IRLS.png`) display, for each non-isosbestic channel:
  * On the left:
    - The ΔF/F trace of the regressed and low-pass filtered isosbestic ΔF/F traces (the estimated motion component)
    - Low-pass filtered ΔF/F traces of the color and isosbestic channels
    - The motion corrected, and optionally noise filtered, ΔF/F trace
  * In the middle: The corresponding power spectral densities with motion and (if applicable) noise cutoff frequencies indicated as dashed vertical lines
  * On the right: A scatter plot of color vs isosbestic ΔF/F values for both the original and low-pass filtered (`cutoff_freq_motion`) data with the fitted regression line

Within the `dff-qc` subdirectory is also the `quality_control.json` for the [aind-qc-portal](https://github.com/AllenNeuralDynamics/aind-qc-portal), with one status per ROI and method (worst channel wins). The metrics are:
- **Calibration ratio**: the fit's median negative residual over what Gaussian noise would give; about 1 for a well-placed baseline.
- **Baseline level**: the mode of the 1 Hz low-passed ΔF/F at the start, middle and end of the session; negative if F<sub>0</sub> is too high.
- **Motion correction**: noise inflation of the corrected over the uncorrected trace.

Each metric's description in `quality_control.json` gives its pass and fail limits (constants in `run_capsule.py`).
