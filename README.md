# aind-fip-dff

This capsule processes input NWB files containing raw fiber photometry data by generating baseline-corrected (ΔF/F) and motion-corrected traces, which are then appended back to the NWB file.

## Method
### Baseline (F<sub>0</sub>) Estimation Methods for Calculating ΔF/F
Three baseline estimation methods are implemented:
- "poly": Fits a 4th-order polynomial using Ordinary Least Squares (OLS).
- "exp": Fits a biphasic exponential using OLS: $a\cdot e^{-b\cdot t} + c\cdot e^{-d\cdot t}$
- "bright": Robust fit with  [Biphasic exponential decay (bleaching)]  x  [Increasing saturating exponential (brightening)] using Iteratively Reweighted Least Squares (IRLS):  $b_{\infty} \cdot (1 + b_{slow} e^{-t/\tau_{slow}} + b_{fast} e^{-t/\tau_{fast}}) \cdot (1-b_{bright} e^{-t/\tau_{bright}}))$

### Motion Correction
Motion correction is carried out in two steps:
1. **Estimating motion attenuation**: A second-order Butterworth filter is applied to smooth the isosbestic trace.
2. **Removing the motion component**: Robust regression is used to subtract the motion signal from all channels.


## Input

All parameters are passed to `run_capule.py` using `python run_capule.py [parameters]`.
Parameters are defined in `__main__` using argparse.  
Key parameters include:  
- '--source_pattern': Specifies the regular expression used to locate input NWB files containing raw fiber data.
- '--dff_methods': Defines the method(s) to be used for calculating ΔF/F.

## Output

The primary output is the updated NWB file, which includes the newly processed data:
- **Baseline-corrected traces (ΔF/F)**: Stored as `[Channel]_[Fiber]_dff-[method]` (e.g., `G_1_dff-poly`). 
- **Fully preprocessed traces (ΔF/F with motion correction)**: Stored as `[Channel]_[Fiber]_preprocessed-[method]` (e.g., `G_1_preprocessed-poly`). 

For quality control (QC), the `qc` subdirectory contains visualizations for each fiber and ΔF/F method. These figures display raw, ΔF/F, and preprocessed (ΔF/F with motion correction) traces of all channels. For example, `Fiber1_bright.png`. Within the `qc` subdirectory is also the `quality_control.json` for the [aind-qc-portal](https://github.com/AllenNeuralDynamics/aind-qc-portal).
