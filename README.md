# aind-fip-dff

This capsule processes input NWB files containing raw fiber photometry data by generating baseline-corrected (ΔF/F) and motion-corrected traces, which are then appended back to the NWB file.

## Method
### Baseline (F<sub>0</sub>) Estimation Methods for Calculating ΔF/F
Three baseline estimation methods are implemented:
- "poly": Fits a 4th-order polynomial using Ordinary Least Squares (OLS).
- "exp": Fits a biphasic exponential using OLS: $a\cdot e^{-b\cdot t} + c\cdot e^{-d\cdot t}$
- "bright": Robust fit with  [Bi- or Tri-phasic exponential decay (bleaching)]  x  [Increasing saturating exponential (brightening)] using Iteratively Reweighted Least Squares (IRLS):  $b_{\infty} \cdot (1 + b_{slow} e^{-t/\tau_{slow}} + b_{fast} e^{-t/\tau_{fast}} + b_{rapid} e^{-t/\tau_{rapid}}) \cdot (1-b_{bright} e^{-t/\tau_{bright}}))$  
  Fitting starts with a biphasic exponential, and the brightening and/or 3rd exponential are only included if they substantially improve the fit.

### Motion Correction
Motion correction is carried out in three steps:
1. **Filtering**: A second-order Butterworth filter is applied to smooth the traces.
2. **Estimating motion attenuation**: Robust regression is used to estimate the coefficients for each filtered trace.
3. **Removing the motion component**: The (unfiltered) isosbestic motion signal, scaled by the corresponding coefficients, is subtracted from all channels.


## Input

All parameters are passed to `run_capule.py` using `python run_capule.py [parameters]`.
Parameters are defined in `__main__` using argparse.  
Key parameters include:  
- '--source_pattern': Specifies the regular expression used to locate input NWB files containing raw fiber data.
- '--dff_methods': Defines the method(s) to be used for calculating ΔF/F.

## Output

The primary output is the updated NWB file, which includes the newly processed data:
- **Baseline-corrected traces (ΔF/F)**: Stored as `[Channel]_[ROI]_dff-[method]` (e.g., `G_1_dff-poly`). 
- **Fully preprocessed traces (ΔF/F with motion correction)**: Stored as `[Channel]_[ROI]_preprocessed-[method]` (e.g., `G_1_preprocessed-poly`). 

For quality control (QC), the `dff-qc` subdirectory contains visualizations for each fiber and ΔF/F method. These figures display raw, ΔF/F, and preprocessed (ΔF/F with motion correction) traces of all channels. For example, `ROI1_bright.png`. Within the `dff-qc` subdirectory is also the `quality_control.json` for the [aind-qc-portal](https://github.com/AllenNeuralDynamics/aind-qc-portal).
