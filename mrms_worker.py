"""
Worker process for MRMS MultiSensor_QPE_01H_Pass1_00.00 batch processing.

Each worker owns a contiguous interval of hourly timestamps. For every
timestamp it:
  1. Fetches the gzipped GRIB2 directly from the NOAA S3 bucket via GDAL
     (/vsigzip//vsicurl/) — no local download step.
  2. Slices per-radar pixel values + per-row pixel areas.
  3. Computes area_with_rain, total_area, total_rain, mean_rain per radar.
  4. Appends one entry per radar to in-memory shard buffers.
  5. Flushes shards to disk every CHECKPOINT_EVERY timestamps and at end.

Missing/failed files: a sentinel entry with -1 in every numeric field is
written for that timestamp, so the per-radar series stays gap-free and
downstream code can detect missing data by checking any numeric field == -1.

Shards are written to: <output_dir>/_shards/worker<i>/<radar_id>.json
The runner merges them into per-radar files after all workers finish.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from osgeo import gdal

from summary import area_with_rain, total_area, total_rain, mean_rain

# --- GDAL plugin path (Windows conda env) — only set if it exists ----------
_WIN_PLUGINS = r"C:\Users\ralaya\Miniforge3\envs\mrms-index\Library\lib\gdalplugins"
if "GDAL_DRIVER_PATH" not in os.environ and os.path.isdir(_WIN_PLUGINS):
    os.environ["GDAL_DRIVER_PATH"] = _WIN_PLUGINS
gdal.UseExceptions()

# Hint GDAL/CURL to retry transient S3 hiccups instead of failing the timestamp.
gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "3")
gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "1")
gdal.SetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO")

URL_TEMPLATE = (
    "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass1_00.00/"
    "{ymd}/MRMS_MultiSensor_QPE_01H_Pass1_00.00_{ymd}-{hms}.grib2.gz"
)

CHECKPOINT_EVERY = 100  # flush shards every N timestamps


def _url_for(ts: datetime) -> str:
    ymd = ts.strftime("%Y%m%d")
    hms = ts.strftime("%H%M%S")
    return URL_TEMPLATE.format(ymd=ymd, hms=hms)


def _stamp(ts: datetime) -> str:
    return ts.strftime("%Y%m%d-%H%M%S")


def _missing_entry(ts: datetime) -> dict:
    return {
        "timestamp": _stamp(ts),
        "area_rain_m2": -1,
        "total_area_m2": -1,
        "total_rain": -1,
        "mean_rain": -1,
    }


def _flush(shards: dict[str, list[dict]], shard_dir: Path) -> None:
    """Atomically rewrite each per-radar shard file with full buffer contents."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    for rid, entries in shards.items():
        path = shard_dir / f"{rid}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(entries, f)
        os.replace(tmp, path)


def _process_one(
    ts: datetime,
    radar_indices: dict[str, np.ndarray],
    pixel_areas: np.ndarray,
    shards: dict[str, list[dict]],
) -> bool:
    """Process a single timestamp. Returns True on success, False on missing/error."""
    url = _url_for(ts)
    vsi_path = f"/vsigzip//vsicurl/{url}"

    try:
        ds = gdal.Open(vsi_path)
        if ds is None:
            raise RuntimeError("gdal.Open returned None")
        array = ds.ReadAsArray().astype(np.float32).ravel()
        num_cols = ds.RasterXSize
        ds = None  # close
    except Exception:
        missing = _missing_entry(ts)
        for rid in radar_indices:
            shards[rid].append(missing)
        return False

    stamp = _stamp(ts)
    for rid, indices in radar_indices.items():
        rain = array[indices]
        rows = indices // num_cols
        areas = pixel_areas[rows]

        a_rain = float(area_with_rain(rain, areas))
        a_total = float(total_area(areas))
        r_total = float(total_rain(rain))
        r_mean = mean_rain(rain)
        # mean_rain may return np.nan when no pixel exceeds threshold; JSON-safe it.
        r_mean = float(r_mean) if not np.isnan(r_mean) else None

        shards[rid].append({
            "timestamp": stamp,
            "area_rain_m2": a_rain,
            "total_area_m2": a_total,
            "total_rain": r_total,
            "mean_rain": r_mean,
        })
    return True


def run_worker(
    worker_id: int,
    start_iso: str,
    end_iso: str,
    output_dir: str,
    pixel_areas_path: str,
    radar_indices_path: str,
    error_log_path: str,
) -> None:
    """
    Process every hourly timestamp in [start_iso, end_iso] inclusive.

    Timestamps are ISO strings (e.g. '2020-10-14T21:00:00') so they survive
    the multiprocessing pickle boundary cleanly.
    """
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)

    pixel_areas = np.load(pixel_areas_path).ravel()
    npz = np.load(radar_indices_path)
    # Materialize to plain dict so the npz file handle can close.
    radar_indices = {rid: npz[rid] for rid in npz.files}
    npz.close()

    shard_dir = Path(output_dir) / "_shards" / f"worker{worker_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards: dict[str, list[dict]] = {rid: [] for rid in radar_indices}

    err_f = open(error_log_path, "a", buffering=1)  # line-buffered
    err_f.write(f"# worker {worker_id} starting interval {start_iso} .. {end_iso}\n")

    total_hours = int((end - start).total_seconds() // 3600) + 1
    processed = 0
    failed = 0
    ts = start

    while ts <= end:
        ok = _process_one(ts, radar_indices, pixel_areas, shards)
        if not ok:
            failed += 1
            err_f.write(f"{_stamp(ts)}\tmissing_or_failed\n")

        processed += 1

        if processed % CHECKPOINT_EVERY == 0:
            _flush(shards, shard_dir)
            pct = 100.0 * processed / total_hours
            print(
                f"[worker {worker_id}] {processed}/{total_hours} "
                f"({pct:.1f}%) ok={processed - failed} fail={failed} last={_stamp(ts)}",
                flush=True,
            )

        ts += timedelta(hours=1)

    _flush(shards, shard_dir)
    err_f.write(f"# worker {worker_id} done: {processed} processed, {failed} failed\n")
    err_f.close()
    print(
        f"[worker {worker_id}] DONE {processed}/{total_hours} ok={processed - failed} fail={failed}",
        flush=True,
    )


if __name__ == "__main__":
    # CLI entry point so the runner can spawn workers via subprocess if it prefers
    # that over multiprocessing.Process. Args mirror run_worker() positional args.
    if len(sys.argv) != 8:
        print(
            "usage: mrms_worker.py <worker_id> <start_iso> <end_iso> "
            "<output_dir> <pixel_areas_path> <radar_indices_path> <error_log>",
            file=sys.stderr,
        )
        sys.exit(2)
    run_worker(
        worker_id=int(sys.argv[1]),
        start_iso=sys.argv[2],
        end_iso=sys.argv[3],
        output_dir=sys.argv[4],
        pixel_areas_path=sys.argv[5],
        radar_indices_path=sys.argv[6],
        error_log_path=sys.argv[7],
    )