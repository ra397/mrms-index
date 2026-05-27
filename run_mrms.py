"""
Runner: process MRMS MultiSensor_QPE_01H_Pass1_00.00 from
2020-10-14 21:00:00 UTC through 2026-05-26 21:00:00 UTC (1-hour increments)
using 6 worker processes, then merge per-worker shards into per-radar JSON files.

Output layout:
  output/
    <radar_id>.json         # final sorted list of entries (one per timestamp)
    errors.log              # missing/failed timestamps from all workers
    _shards/
      worker0/<radar_id>.json
      worker1/<radar_id>.json
      ...

Each final per-radar file is a JSON list of entries:
  [
    {"timestamp": "20201014-210000",
     "area_rain_m2": ..., "total_area_m2": ...,
     "total_rain": ...,   "mean_rain": ...},
    ...
  ]

Missing-file sentinels carry -1 in every numeric field (mean_rain may also
be -1; in non-missing entries mean_rain is null when no pixel exceeds the
rain threshold).

Run:
  python run_mrms.py
or, to override defaults:
  python run_mrms.py --workers 6 --output output \
      --pixel-areas pixel_areas/pixel_areas_m2.npy \
      --radar-indices radar_indices/indices.npz
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from mrms_worker import run_worker

# Inclusive bounds, 1-hour cadence.
START = datetime(2020, 10, 14, 21, 0, 0)
END = datetime(2026, 5, 26, 21, 0, 0)


def split_interval(start: datetime, end: datetime, n: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end] inclusive on 1-hour cadence into n contiguous chunks."""
    total_hours = int((end - start).total_seconds() // 3600) + 1
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n > total_hours:
        n = total_hours

    base, extra = divmod(total_hours, n)
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunk_start = cursor
        chunk_end = cursor + timedelta(hours=size - 1)
        chunks.append((chunk_start, chunk_end))
        cursor = chunk_end + timedelta(hours=1)
    return chunks


def merge_radar_shards(output_dir: Path, n_workers: int) -> None:
    """
    Concatenate per-worker shards into a single per-radar JSON list.

    Workers own disjoint intervals so concatenation in worker order yields
    chronological order; we still sort by timestamp as a safety net.
    """
    shard_root = output_dir / "_shards"
    if not shard_root.exists():
        print("no _shards directory found — nothing to merge", file=sys.stderr)
        return

    radar_ids: set[str] = set()
    for w in range(n_workers):
        wdir = shard_root / f"worker{w}"
        if wdir.is_dir():
            radar_ids.update(p.stem for p in wdir.glob("*.json"))

    print(f"merging {len(radar_ids)} radars across {n_workers} workers...", flush=True)

    for rid in sorted(radar_ids):
        combined: list[dict] = []
        for w in range(n_workers):
            shard_path = shard_root / f"worker{w}" / f"{rid}.json"
            if not shard_path.exists():
                continue
            with open(shard_path) as f:
                combined.extend(json.load(f))

        combined.sort(key=lambda e: e["timestamp"])

        out_path = output_dir / f"{rid}.json"
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(combined, f)
        tmp.replace(out_path)

    print("merge complete", flush=True)


def _worker_entry(args: tuple) -> None:
    """Thin wrapper so multiprocessing can pickle the call."""
    run_worker(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=str, default="output")
    ap.add_argument("--pixel-areas", type=str, default="pixel_areas/pixel_areas_m2.npy")
    ap.add_argument("--radar-indices", type=str, default="radar_indices/indices.npz")
    ap.add_argument(
        "--keep-shards",
        action="store_true",
        help="don't delete _shards/ after a successful merge",
    )
    ap.add_argument(
        "--skip-process",
        action="store_true",
        help="skip processing and only merge existing shards (useful for resuming)",
    )
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "errors.log"

    chunks = split_interval(START, END, args.workers)
    total_hours = sum(
        int((e - s).total_seconds() // 3600) + 1 for s, e in chunks
    )
    print(f"Total hours to process: {total_hours}")
    for i, (s, e) in enumerate(chunks):
        hrs = int((e - s).total_seconds() // 3600) + 1
        print(f"  worker {i}: {s.isoformat()}  ..  {e.isoformat()}  ({hrs} hours)")

    if not args.skip_process:
        # Fresh error log per run.
        if error_log.exists():
            error_log.unlink()
        error_log.touch()

        t0 = time.time()
        worker_args = [
            (
                i,
                s.isoformat(),
                e.isoformat(),
                str(output_dir),
                args.pixel_areas,
                args.radar_indices,
                str(error_log),
            )
            for i, (s, e) in enumerate(chunks)
        ]

        # 'spawn' is the safe default on every platform for GDAL + numpy.
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker_entry, args=(wa,)) for wa in worker_args]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        bad = [i for i, p in enumerate(procs) if p.exitcode != 0]
        if bad:
            print(
                f"WARNING: workers {bad} exited non-zero — merging what exists, "
                "but check errors.log and shard contents before trusting output.",
                file=sys.stderr,
            )

        elapsed = time.time() - t0
        print(f"\nall workers joined in {elapsed/3600:.2f} h ({elapsed:.0f} s)")

    merge_radar_shards(output_dir, args.workers)

    if not args.keep_shards:
        shard_root = output_dir / "_shards"
        if shard_root.exists():
            shutil.rmtree(shard_root)
            print("removed _shards/ (pass --keep-shards to retain)")


if __name__ == "__main__":
    main()