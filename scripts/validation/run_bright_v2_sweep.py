"""Attach a stage/date-stratified sample of fiber photometry assets, run the
aind-fip-dff capsule (on feat/nonlinear-fit-bright-v2) once per parameter
combination, and download each asset's quality_control.json per run into a
layout aggregate_bright_v2_qc.py's RUNS dict can read directly.

Adapted from p-values_recent_data.py's asset-query/attach/run/download
pattern. Two things to confirm before running the full sweep:
  - CAPSULE_ID below must point at a duplicate of the aind-fip-dff capsule
    checked out on feat/nonlinear-fit-bright-v2 (not the PSTH capsule the
    original script targeted).
  - Whether this capsule's run command consumes `parameters` (positional
    passthrough, e.g. `python code/reprocess.py "$@"`) or needs
    `named_parameters` instead depends on how its Code Ocean run command is
    configured -- test with ONE asset first (see `SMOKE_TEST_FIRST` below)
    before dispatching the full ~100-asset sweep.
"""

import os
from pathlib import Path

import requests
from aind_data_access_api.document_db import MetadataDbClient
from codeocean import CodeOcean
from codeocean.computation import DataAssetsRunParam, RunParams
from dotenv import load_dotenv

load_dotenv()

client = CodeOcean(
    domain="https://codeocean.allenneuraldynamics.org",
    token=os.environ["CODEOCEAN_TOKEN"],
)

# ── Configure ─────────────────────────────────────────────────────────────
CAPSULE_ID = "55239f42-10a9-46f1-bb4d-f87a6266b2c2"  # aind-fip-dff dup, feat/nonlinear-fit-bright-v2
OUT_ROOT = Path("bright_v2_sweep_results")
RIGS = ("446_6D", "446_7D", "446_8D", "447_1D", "447_2D", "447_3D")
# Approximate training-stage proxy (see aind-fip-dff investigation notes):
# DelayMin 0.1 -> warmup/stage1, 0.3 -> stage2, 0.5 -> stage3, 1.0 -> final/graduated.
# Sample across all of these rather than filtering to one, so the pre-GoCue
# window (built on each trial's own Delay period) gets exercised at both the
# short (0.1-0.3s) and long (1.0s) end.
DELAY_MINS = ("0.1", "0.3", "0.5", "1.0")
MONTHS = [f"2025-{m:02d}-" for m in range(1, 13)] + [f"2026-{m:02d}-" for m in range(1, 9)]
N_PER_BUCKET = 5  # ~ N_PER_BUCKET x len(DELAY_MINS) x len(MONTHS worth sampled) assets

RUNS = {
    # These two are independent of which per-trace correction (if any) wins
    # in brightfit-comparison.ipynb's Section 3e-iv rerun -- safe to run now.
    "demean": {
        "parameters": [
            "--dff_methods", "bright", "bright_legacy", "poly",
            "--motion_correction_mode", "demean",
            "--parallel",
        ],
    },
    "intercept": {
        "parameters": [
            "--dff_methods", "bright", "bright_legacy", "poly",
            "--motion_correction_mode", "intercept",
            "--parallel",
        ],
    },
    # PENDING the notebook rerun (HSM-based modecorr + new pctcorr70 vs. the
    # already-tested mediancorr) -- don't run this one yet. The capsule's CLI
    # flag was generalized (aind-fip-dff@8b819cb) from --median_correct to
    # --correction {median,pct70}, so once the notebook settles which
    # correction (if any) is worth testing broadly, just set the value below
    # and uncomment.
    # "demean_correction": {
    #     "parameters": [
    #         "--dff_methods", "bright",
    #         "--motion_correction_mode", "demean",
    #         "--correction", "median",  # or "pct70" -- set after the notebook rerun
    #         "--parallel",
    #     ],
    # },
}

SMOKE_TEST_FIRST = True  # run RUNS["demean"] on ONE asset before the full sweep


def retrieve_fiber_records(rig, month, delay_min, data_level="derived"):
    db = MetadataDbClient(
        host="api.allenneuraldynamics.org",
        database="metadata_index",
        collection="data_assets",
    )
    # NOTE: the original script filtered DelayMin AFTER the query (a plain
    # Python list-comprehension), not in filter_query -- this dot-indexed
    # path (standard MongoDB array-index syntax) should work the same way
    # DocDB is queried elsewhere here, but if retrieve_docdb_records rejects
    # or silently ignores it, fall back to querying without this key and
    # filtering `recs` in Python afterward, as the original script did.
    return db.retrieve_docdb_records(
        filter_query={
            "name": {"$regex": "behavior", "$options": "i"},
            "data_description.data_level": data_level,
            "data_description.modality.abbreviation": {"$regex": "fib", "$options": "i"},
            "rig.rig_id": {"$regex": rig, "$options": "i"},
            "session.session_start_time": {"$regex": month, "$options": "i"},
            "session.stimulus_epochs.0.output_parameters.task_parameters.DelayMin": delay_min,
        }
    )


def gather_stratified_sample() -> list[dict]:
    """Sample up to N_PER_BUCKET records per (delay_min x month) bucket,
    pooling across all configured rigs -- spreads the sample across
    training stage and time rather than taking whatever is most recent."""
    records = []
    for delay_min in DELAY_MINS:
        for month in MONTHS:
            bucket = []
            for rig in RIGS:
                bucket += retrieve_fiber_records(rig, month, delay_min)
            bucket = sorted(bucket, key=lambda r: r["session"]["session_start_time"])
            records += bucket[:N_PER_BUCKET]
    # De-duplicate (a session could in principle match more than one query)
    seen, unique = set(), []
    for r in records:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)
    return unique


def to_data_assets(records: list[dict]) -> list[DataAssetsRunParam]:
    ids = [r["external_links"]["Code Ocean"][0] for r in records]
    return [DataAssetsRunParam(id=i, mount=r["name"]) for i, r in zip(ids, records)]


def run_and_download(run_label: str, parameters: list[str], records: list[dict]) -> None:
    print(f"\n=== Run '{run_label}': {len(records)} assets, parameters={parameters} ===")
    data_assets = to_data_assets(records)
    run_params = RunParams(
        capsule_id=CAPSULE_ID, data_assets=data_assets, parameters=parameters
    )
    computation = client.computations.run_capsule(run_params)
    computation = client.computations.wait_until_completed(computation)
    print(f"  computation {computation.id} finished with state {computation.state}")

    run_root = OUT_ROOT / run_label
    n_ok, n_fail = 0, 0
    for r in records:
        asset_name = r["name"]
        try:
            url = client.computations.get_result_file_download_url(
                computation.id, f"{asset_name}/quality_control.json"
            )
            response = requests.get(url.url)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            out_dir = run_root / asset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "quality_control.json").write_bytes(response.content)
            n_ok += 1
        except Exception as e:
            print(f"  [skip] {asset_name}: {e}")
            n_fail += 1
    print(f"  downloaded {n_ok} quality_control.json files ({n_fail} failed/skipped)")


if __name__ == "__main__":
    assert CAPSULE_ID != "REPLACE_ME", "Set CAPSULE_ID to your aind-fip-dff duplicate first."

    print("Querying DocDB for a stage/date-stratified sample...")
    records = gather_stratified_sample()
    print(f"Found {len(records)} unique assets across {len(DELAY_MINS)} stage buckets "
          f"x {len(MONTHS)} months.")

    if SMOKE_TEST_FIRST:
        print("\nSmoke test: running 'demean' params on a single asset first...")
        run_and_download("demean_smoketest", RUNS["demean"]["parameters"], records[:1])
        input("Check bright_v2_sweep_results/demean_smoketest/ looks right, "
              "then press Enter to continue with the full sweep (Ctrl-C to abort)...")

    for run_label, cfg in RUNS.items():
        run_and_download(run_label, cfg["parameters"], records)

    print(f"\nDone. Point aggregate_bright_v2_qc.py's RUNS dict at "
          f"{OUT_ROOT.resolve()}/<run_label> for each run above.")
