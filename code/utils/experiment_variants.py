"""Experimental `tc_brightfit_v2` variants for the snap / tolerance sweep.

Every arm pins its L-BFGS-B options explicitly, for all fits (OLS model
selection and the final robust fit alike), so results do not depend on which
`aind-ophys-utils` version is installed.

- `bright_a`: `tc_brightfit_v2` as shipped (degenerate-term snap, tau caps),
  scipy-default options.
- `nosnap_scipy`: no snap, `tau1` / `tau_bright` caps lifted to
  `UNBOUNDED_TAU`, scipy-default options. Fitted slow taus can then exceed
  the session length by orders of magnitude and are not meaningful.
- `nosnap_ftol1e10`: as `nosnap_scipy` with `ftol=1e-10`.

Not part of aind-fip-dff PR #91.
"""

import numpy as np

from . import preprocess as _pp

SCIPY_LBFGSB = {"maxiter": 15000, "ftol": 2.220446049250313e-09, "gtol": 1e-5}
UNBOUNDED_TAU = 1e7


def _no_snap(params, n_exp, include_bright, tc, ts, near_bound_frac=0.95):
    return params, n_exp, include_bright, []


def _fit_variant(trace, timestamps, snap, tau_cap, options, **kwargs):
    orig_fit = _pp.nonlinear_fit
    saved = (_pp._snap_degenerate_slow_terms, _pp.TAU1_CAP, _pp.TAU_BRIGHT_CAP)

    def fit(*args, **kw):
        kw["optimizer_options"] = options
        return orig_fit(*args, **kw)

    _pp.nonlinear_fit = fit
    if not snap:
        _pp._snap_degenerate_slow_terms = _no_snap
    if tau_cap is not None:
        _pp.TAU1_CAP = _pp.TAU_BRIGHT_CAP = tau_cap
    try:
        return _pp.tc_brightfit_v2(trace, timestamps, **kwargs)
    finally:
        _pp.nonlinear_fit = orig_fit
        _pp._snap_degenerate_slow_terms, _pp.TAU1_CAP, _pp.TAU_BRIGHT_CAP = saved


def tc_brightfit_v2_a(
    trace: np.ndarray, timestamps: np.ndarray, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Shipped fit (snap, tau caps), scipy-default options."""
    return _fit_variant(
        trace, timestamps, snap=True, tau_cap=None, options=SCIPY_LBFGSB, **kwargs
    )


def tc_brightfit_v2_nosnap_scipy(
    trace: np.ndarray, timestamps: np.ndarray, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """No snap, unbounded slow taus, scipy-default options."""
    return _fit_variant(
        trace,
        timestamps,
        snap=False,
        tau_cap=UNBOUNDED_TAU,
        options=SCIPY_LBFGSB,
        **kwargs,
    )


def tc_brightfit_v2_nosnap_ftol1e10(
    trace: np.ndarray, timestamps: np.ndarray, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """As `tc_brightfit_v2_nosnap_scipy` with ``ftol=1e-10``."""
    return _fit_variant(
        trace,
        timestamps,
        snap=False,
        tau_cap=UNBOUNDED_TAU,
        options={**SCIPY_LBFGSB, "ftol": 1e-10},
        **kwargs,
    )


FITS = {
    "bright_a": tc_brightfit_v2_a,
    "nosnap_scipy": tc_brightfit_v2_nosnap_scipy,
    "nosnap_ftol1e10": tc_brightfit_v2_nosnap_ftol1e10,
}
assert set(FITS) == set(_pp.VARIANT_METHODS)
