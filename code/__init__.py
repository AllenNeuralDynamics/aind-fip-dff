import os

from . import utils
from .run_capsule import *

__version__ = "20.0"  # keep in sync with ENV VERSION in environment/Dockerfile

# The capsule's Dockerfile sets these; `write_output_metadata` records them in processing.json.
os.environ.setdefault("VERSION", __version__)
os.environ.setdefault("DFF_EXTRACTION_URL", "https://github.com/AllenNeuralDynamics/aind-fip-dff")

__all__ = [
    # Submodule
    "utils",
    # Version info
    "__version__",
    # From run_capsule.py (functions users might want to use programmatically)
    "create_metric",
    "plot_dff",
    "plot_motion_correction",
    "plot_raw_dff_mc",
    "process_nwb_file",
    "write_output_metadata",
]
