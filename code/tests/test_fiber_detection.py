"""Run from `code/`: python -m pytest tests"""

import sys
from pathlib import Path

import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.fiber_detection import nwb_has_fiber_channels  # noqa: E402


def _nwb(tmp_path, acquisition=None):
    """Minimal NWB-like zarr; `acquisition` lists the acquisition entries
    (None: no acquisition group at all)."""
    path = tmp_path / "nwb.zarr"
    root = zarr.open(str(path), mode="w")
    if acquisition is not None:
        group = root.create_group("acquisition")
        for name in acquisition:
            group.create_group(name)
    return path


@pytest.mark.parametrize("entry", ["G_0", "Iso_1", "R_3", "Signal_0"])
def test_fiber_channel_found(tmp_path, entry):
    assert nwb_has_fiber_channels(_nwb(tmp_path, ["left_lick_time", entry]))


def test_behavior_only(tmp_path):
    path = _nwb(tmp_path, ["left_lick_time", "right_lick_time", "FIP_rising_time"])
    assert not nwb_has_fiber_channels(path)


def test_empty_acquisition(tmp_path):
    assert not nwb_has_fiber_channels(_nwb(tmp_path, []))


def test_no_acquisition_group(tmp_path):
    assert not nwb_has_fiber_channels(_nwb(tmp_path))
