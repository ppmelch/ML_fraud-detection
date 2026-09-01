#!/usr/bin/env python3
"""Check prediction-horizon viability against execution latency.

The project brief states the constraint directly: the horizon must exceed the
latency, or the order reaches the market after the move has already happened.
Horizons are measured in events and latency in milliseconds, so answering the
question requires converting between them using measured event density.

This script also detects the degenerate case where a dataset's event gaps are
so wide that every latency level converts to zero events — in which case
latency cannot be studied on that data at all, and saying so is the honest
outcome.

Usage:
    python analyze_horizons.py lob_reconstructed.csv \\
        --horizons 1 3 5 --latencies 0 1 10 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# A horizon should exceed latency by at least this factor to be worth trading:
# arriving exactly as the move completes captures none of it.
SAFETY_FACTOR = 2.0


def measure_event_density(timestamps: pd.Series) -> dict:
    """Inter-arrival statistics, in milliseconds."""
    gaps = timestamps.diff().dropna()
    gaps = gaps[gaps >= 0]

    if gaps.empty:
        return {"median_gap_ms": np.nan, "mean_gap_ms": np.nan,
                "p25_gap_ms": np.nan, "p75_gap_ms": np.nan,
                "events_per_second": np.nan, "n_events": len(timestamps)}

    median = float(gaps.median())
    return {
        "median_gap_ms": median,
        "mean_gap_ms": float(gaps.mean()),
        "p25_gap_ms": float(gaps.quantile(0.25)),
        "p75_gap_ms": float(gaps.quantile(0.75)),
        "events_per_second": 1000.0 / median if median > 0 else np.inf,
        "n_events": len(timestamps),
        "span_ms": float(timestamps.iloc[-1] - timestamps.iloc[0]),
    }


def latency_to_events(latency_ms: float, timestamps: pd.Series, sample_points: int = 200) -> dict:
    """How many events elapse during `latency_ms`, measured locally.

    A single global median treats a quiet stretch and a burst identically.
    Sampling the actual forward walk at many points respects local density and
    reports the spread, which is what reveals whether the conversion is stable.
    """
    if latency_ms <= 0:
        return {"mean_events": 0.0, "median_events": 0.0, "max_events": 0,
                "degenerate": True, "note": "zero latency"}

    ts = timestamps.to_numpy()
    n = len(ts)
    if n < 2:
        return {"mean_events": 0.0, "median_events": 0.0, "max_events": 0,
                "degenerate": True, "note": "too few events"}

    idx = np.linspace(0, n - 2, min(sample_points, n - 1), dtype=int)
    # searchsorted gives the first index past the deadline; the offset is how
    # many events fit strictly inside the latency window.
    offsets = np.searchsorted(ts, ts[idx] + latency_ms, side="right") - idx - 1
    offsets = np.clip(offsets, 0, None)

    median_events = float(np.median(offsets))
    mean_events = float(offsets.mean())

    # Degenerate when latency spans no events for a typical order. Requiring
    # max == 0 would be too strict: with exponentially distributed gaps a rare
    # tight cluster lets one sample advance, while latency remains irrelevant
    # for the overwhelming majority of orders. The median is what determines
    # whether a latency sweep can produce differentiated results.
    degenerate = median_events == 0

    if degenerate and mean_events > 0:
        note = (f"latency spans no events for the median order "
                f"(mean {mean_events:.3f} events — affects only a small tail)")
    elif degenerate:
        note = "latency spans no events on this data"
    else:
        note = ""

    return {
        "mean_events": mean_events,
        "median_events": median_events,
        "max_events": int(offsets.max()),
        "degenerate": degenerate,
        "note": note,
    }


def analyze(
    timestamps: pd.Series,
    horizons: list[int],
    latencies: list[float],
) -> tuple[pd.DataFrame, dict]:
    """Build the horizon-by-latency viability grid."""
    density = measure_event_density(timestamps)
    median_gap = density["median_gap_ms"]

    rows = []
    for h in horizons:
        horizon_ms = h * median_gap if not np.isnan(median_gap) else np.nan

        for lat in latencies:
            conv = latency_to_events(lat, timestamps)

            if lat == 0:
                viable, reason = True, "zero latency"
            elif np.isnan(horizon_ms):
                viable, reason = False, "event density unknown"
            elif horizon_ms >= lat * SAFETY_FACTOR:
                viable, reason = True, f"horizon {horizon_ms:.0f}ms >= {SAFETY_FACTOR}x latency"
            elif horizon_ms > lat:
                viable, reason = False, f"horizon {horizon_ms:.0f}ms exceeds latency but under {SAFETY_FACTOR}x safety factor"
            else:
                viable, reason = False, f"horizon {horizon_ms:.0f}ms below latency {lat}ms"

            rows.append(
                {
                    "horizon_events": h,
                    "horizon_ms": round(horizon_ms, 1) if not np.isnan(horizon_ms) else np.nan,
                    "latency_ms": lat,
                    "latency_events_mean": round(conv["mean_events"], 2),
                    "latency_events_max": conv["max_events"],
                    "latency_degenerate": conv["degenerate"],
                    "viable": viable,
                    "reason": reason,
                }
            )

    return pd.DataFrame(rows), density


def print_report(grid: pd.DataFrame, density: dict, latencies: list[float]) -> None:
    """Human-readable summary."""
    print("=" * 78)
    print("HORIZON VIABILITY ANALYSIS")
    print("=" * 78)
    print()
    print("Event density")
    print(f"  events              {density['n_events']}")
    print(f"  span                {density.get('span_ms', float('nan')) / 1000:.1f} s")
    print(f"  median gap          {density['median_gap_ms']:.1f} ms")
    print(f"  mean gap            {density['mean_gap_ms']:.1f} ms")
    print(f"  IQR                 {density['p25_gap_ms']:.1f} – {density['p75_gap_ms']:.1f} ms")
    print(f"  rate                {density['events_per_second']:.2f} events/s")
    print()

    nonzero = [lat for lat in latencies if lat > 0]
    degenerate = grid[(grid["latency_ms"] > 0) & grid["latency_degenerate"]]["latency_ms"].unique()

    if nonzero and len(degenerate) == len(nonzero):
        print("!" * 78)
        print("DEGENERATE EVENT DENSITY")
        print()
        print(f"Every configured latency ({', '.join(f'{l:g}ms' for l in nonzero)}) converts to")
        print("zero events on this dataset. All latency levels would produce identical")
        print("simulation results.")
        print()
        print("This is a property of the data, not a bug. Two honest responses:")
        print("  1. Report that latency is not measurable on this dataset.")
        print("  2. Simulate a compressed-density variant, clearly labelled as such,")
        print("     and never present its results as coming from the primary data.")
        print()
        print("What is NOT acceptable: presenting a flat latency curve as evidence")
        print("that latency does not matter in markets.")
        print("!" * 78)
        print()

    print("Viability grid")
    display = grid[["horizon_events", "horizon_ms", "latency_ms",
                    "latency_events_mean", "viable", "reason"]]
    print(display.to_string(index=False))
    print()

    print("-" * 78)
    max_lat = max(latencies) if latencies else 0
    at_max = grid[grid["latency_ms"] == max_lat]
    viable_at_max = at_max[at_max["viable"]]["horizon_events"].tolist()

    if viable_at_max:
        # Shortest viable horizon: accuracy generally falls as horizon grows,
        # so the shortest one that clears the constraint is preferred.
        print(f"RECOMMENDATION: horizon {min(viable_at_max)} events")
        print(f"  Shortest horizon viable at the maximum configured latency ({max_lat:g}ms).")
        print("  Accuracy generally degrades with horizon, so prefer the shortest")
        print("  that satisfies the constraint.")
    else:
        print(f"RECOMMENDATION: no configured horizon is viable at {max_lat:g}ms latency.")
        print("  Either accept a lower latency ceiling, or evaluate longer horizons")
        print("  and accept the accuracy cost. Record whichever choice is made.")
    print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lob_file", type=Path, help="reconstructed LOB CSV with a timestamp column")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5],
                        help="prediction horizons in events")
    parser.add_argument("--latencies", type=float, nargs="+", default=[0, 1, 10, 100],
                        help="latency levels in milliseconds")
    parser.add_argument("--timestamp-col", default="timestamp", help="timestamp column name")
    parser.add_argument("--output", type=Path, help="write the grid to CSV")
    args = parser.parse_args()

    if not args.lob_file.exists():
        print(f"ERROR: file not found: {args.lob_file}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.lob_file)
    if args.timestamp_col not in df.columns:
        print(f"ERROR: column '{args.timestamp_col}' not in file", file=sys.stderr)
        print(f"available: {list(df.columns)[:15]}", file=sys.stderr)
        return 2

    timestamps = df[args.timestamp_col].sort_values().reset_index(drop=True)
    grid, density = analyze(timestamps, args.horizons, args.latencies)
    print_report(grid, density, args.latencies)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.output, index=False)
        print(f"\nGrid written to {args.output}")

    max_lat = max(args.latencies) if args.latencies else 0
    any_viable = grid[(grid["latency_ms"] == max_lat) & grid["viable"]].shape[0] > 0
    return 0 if any_viable else 1


if __name__ == "__main__":
    sys.exit(main())
