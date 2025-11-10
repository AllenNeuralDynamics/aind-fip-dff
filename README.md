# aind-fip-dff

This capsule processes input NWB files containing raw fiber photometry data by generating baseline-corrected (ΔF/F) and motion-corrected traces, which are then appended back to the NWB file.

## Method
### Baseline (F<sub>0</sub>) Estimation Methods for Calculating ΔF/F
Three baseline estimation methods are implemented:
- "poly": Fits a 4th-order polynomial using Ordinary Least Squares (OLS).
- "exp": Fits a biphasic exponential using OLS: $a\cdot e^{-b\cdot t} + c\cdot e^{-d\cdot t}$
- "tri-exp": Fits a triphasic exponential using OLS: $a\cdot e^{-b\cdot t} + c\cdot e^{-d\cdot t} + e\cdot e^{-f\cdot t} + g$
- "bright": Robust fit with  [Bi- or Tri-phasic exponential decay (bleaching)]  x  [Increasing saturating exponential (brightening)] using Iteratively Reweighted Least Squares (IRLS):  $b_{\infty} \cdot (1 + b_{slow} e^{-t/\tau_{slow}} + b_{fast} e^{-t/\tau_{fast}} + b_{rapid} e^{-t/\tau_{rapid}}) \cdot (1-b_{bright} e^{-t/\tau_{bright}}))$  
  Fitting starts with a biphasic exponential, and the brightening and/or 3rd exponential are only included if they substantially improve the fit.

### Motion Correction
Motion correction is carried out in three to four steps:
1. **Motion filtering**: A second-order Butterworth filter is applied to smooth the traces (parameter: `cutoff_freq_motion`).
2. **Estimating motion attenuation**: Robust regression is used to estimate the coefficients for each filtered trace.
3. **Removing the motion component**: The (unfiltered) isosbestic motion signal, scaled by the corresponding coefficients, is subtracted from all channels.
4. **Noise filtering**: Optionally, a second-order Butterworth filter is applied to filter out noise (parameter: `cutoff_freq_noise`).


## Input

All parameters are passed to `run_capule.py` using `python run_capule.py [parameters]`.
Parameters are defined in `__main__` using argparse.  
Key parameters include:  
- '--dff_methods': Defines the method(s) to be used for calculating ΔF/F. Default: `["poly", "exp", "bright"]`
- '--cutoff_freq_motion': Cutoff frequency of the lowpass Butterworth filter that's only applied for estimating the regression coefficient, in Hz. Default: `0.05`
- '--cutoff_freq_noise': Cutoff frequency of the lowpass Butterworth filter that's applied to filter out noise, in Hz. If `None` or greater than the Nyquist frequency (`10` for `20`Hz sampling rate) no noise filtering is performed. Default: `3`

## Output

The primary output is the updated NWB file, which includes the newly processed data:
- **Baseline-corrected traces (ΔF/F)**: Stored as `[Channel]_[ROI]_dff-[method]` (e.g., `G_1_dff-poly`). 
- **Fully preprocessed traces (ΔF/F with motion correction)**: Stored as `[Channel]_[ROI]_dff-[method]_mc-[method]` (e.g., `G_1_dff-poly_mc-iso-IRLS`). 

For quality control (QC), the `dff-qc` subdirectory contains visualizations of the two processing steps for each fiber and ΔF/F method.   
- The ΔF/F figures (e.g., `ROI0_dff-bright.png`) display raw traces with fitted baseline (F<sub>0</sub>) and ΔF/F traces of all channels.   
- The motion-correction figures (e.g., `ROI0_dff-bright_mc-iso-IRLS.png`) display, for each non-isosbestic channel:
  * On the left:
    - The ΔF/F trace of the regressed and low-pass filtered isosbestic ΔF/F traces (the estimated motion component)
    - Low-pass filtered ΔF/F traces of the color and isosbestic channels
    - The motion corrected, and optionally noise filtered, ΔF/F trace
  * In the middle: The corresponding power spectral densities with motion and (if applicable) noise cutoff frequencies indicated as dashed vertical lines
  * On the right: A scatter plot of color vs isosbestic ΔF/F values for both the original and low-pass filtered (`cutoff_freq_motion`) data with the fitted regression line

Within the `dff-qc` subdirectory is also the `quality_control.json` for the [aind-qc-portal](https://github.com/AllenNeuralDynamics/aind-qc-portal).
