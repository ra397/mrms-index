"""
Reshape per-radar JSON files into a compact column-oriented format.

Input  shape (per file in output/):
  [
    {"timestamp": "YYYYMMDD-HHMMSS",
     "area_rain_m2": float, "total_area_m2": float,
     "total_rain": float,   "mean_rain": float | null},
    ...
  ]

Output shape (per file in output_compact/):
  {
    "timestamps":     "<base64 of uint32  little-endian array, unix epoch seconds UTC>",
    "area_rain_km2":  "<base64 of float32 little-endian array>",
    "total_area_km2": "<base64 of float32 little-endian array>",
    "total_rain_mm":  "<base64 of float32 little-endian array>",
    "mean_rain_mm":   "<base64 of float32 little-endian array>"
  }

Unit conversions:
  - area_rain_m2  -> area_rain_km2   (divide by 1e6)
  - total_area_m2 -> total_area_km2  (divide by 1e6)
  - total_rain stays in original units, renamed total_rain_mm
  - mean_rain  stays in original units, renamed mean_rain_mm

Sentinel handling:
  - -1 / -2 sentinels are preserved as float32 -1.0 / -2.0 (areas are also
    divided by 1e6, so -1 m^2 becomes -1e-6 km^2 — which would collide with
    a near-zero real value). To keep sentinels detectable, we DO NOT divide
    sentinel values; we copy them through as-is. Real values are converted.
  - mean_rain null becomes float32 NaN.

Run:
  python reshape.py
or override:
  python reshape.py --input output --output output_compact
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SENTINELS = (-1, -2, -1.0, -2.0)


def _parse_stamp_to_epoch(stamp: str) -> int:
    """'YYYYMMDD-HHMMSS' (UTC) -> unix epoch seconds (uint32-safe)."""
    dt = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _b64(arr: np.ndarray) -> str:
    """Base64-encode a numpy array's raw bytes (little-endian, contiguous)."""
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _to_float32_array(values: list, *, scale: float = 1.0) -> np.ndarray:
    """
    Convert a list of numbers (possibly with None for nulls) to float32.

    Sentinels (-1, -2) pass through unscaled so downstream code can match
    them exactly. Real values are multiplied by `scale` (use 1e-6 for m^2
    -> km^2). None becomes NaN.
    """
    out = np.empty(len(values), dtype="<f4")
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
        elif v in SENTINELS:
            out[i] = float(v)
        else:
            out[i] = float(v) * scale
    return out


def reshape_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    """Read one per-radar JSON, write its compact form. Returns (in_bytes, out_bytes)."""
    with open(in_path) as f:
        entries = json.load(f)

    if not entries:
        # Edge case: empty list. Emit empty arrays so consumers don't need a special case.
        compact = {
            "timestamps":     _b64(np.array([], dtype="<u4")),
            "area_rain_km2":  _b64(np.array([], dtype="<f4")),
            "total_area_km2": _b64(np.array([], dtype="<f4")),
            "total_rain_mm":  _b64(np.array([], dtype="<f4")),
            "mean_rain_mm":   _b64(np.array([], dtype="<f4")),
        }
    else:
        # Pull each column as a plain list first; vectorize the scaling once we
        # have arrays. Sentinels need per-element handling (don't scale them),
        # so we do that inside _to_float32_array.
        n = len(entries)
        ts = np.empty(n, dtype="<u4")
        ar, ta, tr, mr = [], [], [], []
        for i, e in enumerate(entries):
            ts[i] = _parse_stamp_to_epoch(e["timestamp"])
            ar.append(e["area_rain_m2"])
            ta.append(e["total_area_m2"])
            tr.append(e["total_rain"])
            mr.append(e["mean_rain"])

        compact = {
            "timestamps":     _b64(ts),
            "area_rain_km2":  _b64(_to_float32_array(ar, scale=1e-6)),
            "total_area_km2": _b64(_to_float32_array(ta, scale=1e-6)),
            "total_rain_mm":  _b64(_to_float32_array(tr, scale=1.0)),
            "mean_rain_mm":   _b64(_to_float32_array(mr, scale=1.0)),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(compact, f)
    tmp.replace(out_path)

    return in_path.stat().st_size, out_path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="output")
    ap.add_argument("--output", type=str, default="output_compact")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        print(f"input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Only reshape per-radar files; skip errors.log, _shards/, etc.
    in_files = sorted(p for p in in_dir.glob("*.json") if p.is_file())
    if not in_files:
        print(f"no *.json files in {in_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reshaping {len(in_files)} files: {in_dir} -> {out_dir}")

    total_in = 0
    total_out = 0
    for i, p in enumerate(in_files, 1):
        out_p = out_dir / p.name
        in_bytes, out_bytes = reshape_file(p, out_p)
        total_in += in_bytes
        total_out += out_bytes
        if i % 25 == 0 or i == len(in_files):
            pct = 100.0 * (1 - total_out / total_in) if total_in else 0.0
            print(
                f"  [{i}/{len(in_files)}] {p.name}: "
                f"{in_bytes/1024:.0f} KB -> {out_bytes/1024:.0f} KB  "
                f"(running savings: {pct:.1f}%)",
                flush=True,
            )

    pct = 100.0 * (1 - total_out / total_in) if total_in else 0.0
    print(
        f"\ndone. {total_in/1024/1024:.1f} MB -> {total_out/1024/1024:.1f} MB  "
        f"({pct:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()