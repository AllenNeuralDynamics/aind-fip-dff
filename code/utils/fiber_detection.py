"""Decide from an NWB file whether it holds fiber photometry data."""

import zarr

#: Acquisition entries of fiber photometry channels ("G_0", "Iso_1", "R_2", ...).
FIBER_PREFIXES = ("G_", "R_", "Iso_", "Signal_")


def nwb_has_fiber_channels(nwb_path) -> bool:
    """Whether the NWB zarr at `nwb_path` has fiber photometry channels.

    The NWB file is what gets processed, so this is the criterion for
    running the dF/F step; the raw asset's folders are not consulted.

    Parameters
    ----------
    nwb_path : str or Path
        Path to the NWB zarr directory.

    Returns
    -------
    bool
        True if the acquisition group has an entry starting with one of
        `FIBER_PREFIXES`.
    """
    root = zarr.open(str(nwb_path), mode="r")
    acquisition = root.get("acquisition")
    if acquisition is None:
        return False
    return any(name.startswith(FIBER_PREFIXES) for name in acquisition.keys())
