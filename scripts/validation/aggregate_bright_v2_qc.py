#!/usr/bin/env python3
"""Aggregate aind-fip-dff QC metrics across many processed assets and across
the 4 (motion_correction_mode x median_correct) combinations, for validating
feat/nonlinear-fit-bright-v2's bright vs. bright_legacy swap at scale.

Point RUNS below at the 4 reprocess.py output roots (one per run), each
containing one subfolder per processed asset with its own
quality_control.json -- reprocess.py's process1dataset writes it at
<output_dir>/<asset_name>/quality_control.json. Then run this script.

Produces, under ./bright_v2_qc_aggregation/:
  - calibration_ratio.csv, pregocue_drift.csv, motion_coef.csv (long-format,
    one row per asset x fiber x channel x method, ready for further analysis)
  - a handful of comparison plots (bright vs. bright_legacy, across runs)
  - a printed summary table (median calibration ratio / mean dF/F / slope,
    by run x method x channel)
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configure: point each label at that run's output root. run_bright_v2_sweep.py
# downloads to bright_v2_sweep_results/<run_label>/<asset_name>/quality_control.json
# (relative to wherever that script was run) -- adjust below if you copied/
# downloaded the results elsewhere. Add the third (correction) run here once
# run_bright_v2_sweep.py's commented-out "demean_correction" entry is filled
# in and run.
RUNS = {
    "demean": Path("bright_v2_sweep_results/demean"),
    "intercept": Path("bright_v2_sweep_results/intercept"),
}
METHODS = ("bright", "bright_legacy", "poly")

_CALIB_RE = re.compile(r"^Calibration ratio of ROI (\S+) using method '(\S+)'$")
_DRIFT_RE = re.compile(r"^Pre-(\S+) dF/F drift of ROI (\S+) using method '(\S+)'$")
_MOTION_RE = re.compile(r"^Motion correction of ROI (\S+) using method '(\S+)'$")


def iter_asset_qc_files(run_root: Path):
    """Yield (asset_name, quality_control.json path) for every processed
    asset under `run_root` (one subfolder per asset, each with its own
    quality_control.json at its top level)."""
    if not run_root.exists():
        print(f"  [warning] run root does not exist: {run_root}")
        return
    for qc_path in sorted(run_root.glob("*/quality_control.json")):
        yield qc_path.parent.name, qc_path


def parse_asset_qc(qc_path: Path, run_label: str, asset_name: str):
    """Extract calibration-ratio, pre-event-drift, and motion-correction
    rows for `METHODS` from one asset's quality_control.json."""
    calib_rows, drift_rows, motion_rows = [], [], []
    try:
        qc = json.loads(qc_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [skip] {qc_path}: {e}")
        return calib_rows, drift_rows, motion_rows

    for ev in qc.get("evaluations", []):
        for metric in ev.get("metrics", []):
            name = metric.get("name", "")
            value = metric.get("value")

            m = _CALIB_RE.match(name)
            if m:
                fiber, method = m.groups()
                if method in METHODS and isinstance(value, dict):
                    for channel, stages in value.items():
                        if not isinstance(stages, dict):
                            continue
                        for stage, ratios in stages.items():
                            if not isinstance(ratios, dict):
                                continue
                            for depth, ratio in ratios.items():
                                calib_rows.append(
                                    dict(
                                        run=run_label,
                                        asset=asset_name,
                                        fiber=fiber,
                                        method=method,
                                        channel=channel,
                                        stage=stage,
                                        depth=depth,
                                        ratio=ratio,
                                    )
                                )
                continue

            m = _DRIFT_RE.match(name)
            if m:
                event_label, fiber, method = m.groups()
                if method in METHODS and isinstance(value, dict):
                    for channel, stages in value.items():
                        if not isinstance(stages, dict):
                            continue
                        for stage, stats in stages.items():
                            if not isinstance(stats, dict):
                                continue
                            drift_rows.append(
                                dict(
                                    run=run_label,
                                    asset=asset_name,
                                    fiber=fiber,
                                    method=method,
                                    channel=channel,
                                    stage=stage,
                                    event=event_label,
                                    **stats,
                                )
                            )
                continue

            m = _MOTION_RE.match(name)
            if m:
                fiber, method = m.groups()
                if method in METHODS:
                    motion_rows.append(
                        dict(
                            run=run_label,
                            asset=asset_name,
                            fiber=fiber,
                            method=method,
                            max_coef=value,
                        )
                    )
                continue

    return calib_rows, drift_rows, motion_rows


def aggregate(runs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calib_rows, drift_rows, motion_rows = [], [], []
    for run_label, run_root in runs.items():
        n_assets = 0
        for asset_name, qc_path in iter_asset_qc_files(run_root):
            n_assets += 1
            c, d, m = parse_asset_qc(qc_path, run_label, asset_name)
            calib_rows += c
            drift_rows += d
            motion_rows += m
        print(f"{run_label:<24} {n_assets:>4} assets found under {run_root}")

    return (
        pd.DataFrame(calib_rows),
        pd.DataFrame(drift_rows),
        pd.DataFrame(motion_rows),
    )


def plot_summary(
    df_calib: pd.DataFrame, df_drift: pd.DataFrame, df_motion: pd.DataFrame, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_labels = list(RUNS)
    colors = {"bright": "C0", "bright_legacy": "C1", "poly": "C2"}

    # Both metrics carry a "stage" column: "dff" (before motion correction --
    # isolates the baseline-fit method) and "motion_corrected" (after -- the
    # actual production output, where demean vs. intercept differ). Plot
    # each stage separately rather than pooling them.
    stages = sorted(set(df_calib.get("stage", pd.Series(dtype=str))) | set(
        df_drift.get("stage", pd.Series(dtype=str))
    )) or ["dff", "motion_corrected"]

    for stage in stages:
        calib_s = df_calib[df_calib.stage == stage] if not df_calib.empty else df_calib
        drift_s = df_drift[df_drift.stage == stage] if not df_drift.empty else df_drift

        if not calib_s.empty:
            k0 = calib_s[calib_s.depth == "k0"]
            fig, axes = plt.subplots(
                1, len(run_labels), figsize=(4 * len(run_labels), 3.5), sharey=True
            )
            axes = np.atleast_1d(axes)
            for ax, run in zip(axes, run_labels):
                sub = k0[k0.run == run]
                for method in METHODS:
                    vals = sub[sub.method == method].ratio.dropna()
                    if len(vals):
                        ax.hist(vals, bins=30, alpha=0.5, label=method, color=colors[method])
                ax.axvline(1.0, c="k", ls="--", lw=0.8)
                ax.set_title(run, fontsize=9)
                ax.set_xlabel("k0 calibration ratio")
            axes[0].set_ylabel("count")
            axes[0].legend(fontsize=8)
            plt.suptitle(f"stage = {stage}", y=1.02)
            plt.tight_layout()
            plt.savefig(out_dir / f"calibration_ratio_k0_by_run_{stage}.png", dpi=150)
            plt.close()

            fig, ax = plt.subplots(figsize=(5, 5))
            for method in METHODS:
                piv = calib_s[calib_s.method == method].pivot_table(
                    index=["run", "asset", "fiber", "channel"], columns="depth", values="ratio"
                )
                if {"k0", "k1"}.issubset(piv.columns):
                    ax.scatter(piv.k0, piv.k1, s=8, alpha=0.4, label=method, color=colors[method])
            lims = (0.5, 1.8)
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("k0 ratio")
            ax.set_ylabel("k1 ratio")
            ax.set_title(f"stage = {stage}", fontsize=9)
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"calibration_ratio_k0_vs_k1_{stage}.png", dpi=150)
            plt.close()

        if not drift_s.empty:
            # Distributions, faceted by run (like the calibration-ratio plot) --
            # mean_dff on top, slope on bottom, one column per run.
            fig, axes = plt.subplots(
                2, len(run_labels), figsize=(4 * len(run_labels), 7), sharex="row"
            )
            axes = np.atleast_2d(axes)
            for col, run in enumerate(run_labels):
                sub_run = drift_s[drift_s.run == run]
                for method in METHODS:
                    sub = sub_run[sub_run.method == method]
                    axes[0, col].hist(
                        (sub.mean_dff * 100).dropna(), bins=25, alpha=0.5, label=method,
                        color=colors[method],
                    )
                    axes[1, col].hist(
                        (sub.slope * 100).dropna(), bins=25, alpha=0.5, label=method,
                        color=colors[method],
                    )
                axes[0, col].axvline(0, c="k", ls="--", lw=0.8)
                axes[1, col].axvline(0, c="k", ls="--", lw=0.8)
                axes[0, col].set_title(run, fontsize=9)
                axes[1, col].set_xlabel("pre-event slope [%/trial]")
            axes[0, 0].set_ylabel("pre-event mean dF/F [%]\ncount")
            axes[1, 0].set_ylabel("count")
            axes[0, 0].legend(fontsize=8)
            plt.suptitle(f"stage = {stage}", y=1.01)
            plt.tight_layout()
            plt.savefig(out_dir / f"pregocue_drift_by_run_{stage}.png", dpi=150)
            plt.close()

            # Paired comparison: bright vs. bright_legacy on the SAME
            # asset/fiber/channel/run -- the actual head-to-head that
            # matters for the swap decision, not just two marginal
            # distributions.
            piv_mean = drift_s.pivot_table(
                index=["run", "asset", "fiber", "channel"], columns="method", values="mean_dff"
            )
            piv_slope = drift_s.pivot_table(
                index=["run", "asset", "fiber", "channel"], columns="method", values="slope"
            )
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            for ax, piv, label in (
                (axes[0], piv_mean, "pre-event mean dF/F [%]"),
                (axes[1], piv_slope, "pre-event slope [%/trial]"),
            ):
                if {"bright", "bright_legacy"}.issubset(piv.columns):
                    x = piv["bright_legacy"] * 100
                    y = piv["bright"] * 100
                    ax.scatter(x, y, s=10, alpha=0.4, c="C2")
                    lim = np.nanmax(np.abs(np.concatenate([x, y]))) * 1.05 or 1.0
                    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
                    ax.axhline(0, c="gray", lw=0.6)
                    ax.axvline(0, c="gray", lw=0.6)
                    ax.set_xlim(-lim, lim)
                    ax.set_ylim(-lim, lim)
                    n_closer = int((y.abs() < x.abs()).sum())
                    ax.set_title(
                        f"{label}\n|bright| < |bright_legacy| in {n_closer}/{x.notna().sum()}",
                        fontsize=9,
                    )
                ax.set_xlabel("bright_legacy")
                ax.set_ylabel("bright")
                ax.legend(fontsize=8)
            plt.suptitle(f"stage = {stage}", y=1.02)
            plt.tight_layout()
            plt.savefig(out_dir / f"pregocue_drift_paired_bright_vs_legacy_{stage}.png", dpi=150)
            plt.close()

    if not df_motion.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        linestyles = ("-", "--", "-.", ":")
        for method in METHODS:
            for run, ls in zip(run_labels, linestyles):
                vals = df_motion[
                    (df_motion.method == method) & (df_motion.run == run)
                ].max_coef.dropna()
                if len(vals):
                    ax.hist(
                        vals,
                        bins=30,
                        histtype="step",
                        label=f"{method}/{run}",
                        linestyle=ls,
                        color=colors[method],
                    )
        ax.set_xlabel("max motion-correction regression coefficient")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "motion_coef_by_run.png", dpi=150)
        plt.close()


def print_summary_table(df_calib: pd.DataFrame, df_drift: pd.DataFrame) -> None:
    if not df_calib.empty:
        print("\n=== Calibration ratio (k0), median [IQR] by stage x run x method x channel ===")
        g = df_calib[df_calib.depth == "k0"].groupby(
            ["stage", "run", "method", "channel"]
        ).ratio
        summary = g.agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            n="count",
        )
        print(summary.round(4))

    if not df_drift.empty:
        print("\n=== Pre-event mean dF/F [%], median by stage x run x method x channel ===")
        g = (
            df_drift.assign(mean_dff_pct=df_drift.mean_dff * 100)
            .groupby(["stage", "run", "method", "channel"])
            .mean_dff_pct
        )
        print(g.median().unstack("method").round(4))

        print("\n=== Pre-event slope [%/trial], median by stage x run x method x channel ===")
        g = (
            df_drift.assign(slope_pct=df_drift.slope * 100)
            .groupby(["stage", "run", "method", "channel"])
            .slope_pct
        )
        print(g.median().unstack("method").round(6))


if __name__ == "__main__":
    df_calib, df_drift, df_motion = aggregate(RUNS)

    out_dir = Path("bright_v2_qc_aggregation")
    out_dir.mkdir(exist_ok=True)
    df_calib.to_csv(out_dir / "calibration_ratio.csv", index=False)
    df_drift.to_csv(out_dir / "pregocue_drift.csv", index=False)
    df_motion.to_csv(out_dir / "motion_coef.csv", index=False)
    print(
        f"\nSaved {len(df_calib)} calibration rows, {len(df_drift)} drift rows, "
        f"{len(df_motion)} motion rows to {out_dir}/"
    )

    print_summary_table(df_calib, df_drift)
    plot_summary(df_calib, df_drift, df_motion, out_dir)
    print(f"\nPlots saved to {out_dir}/")
