"""Benchmark sequential image-to-result requests to a LOCAL Inference server.

Example:
  python code/benchmark_edge.py test-images --model YOUR_MODEL_ID \
      --confidence 0.20 --output edge-benchmark.json

Use a cutoff chosen on validation data; 0.20 is only an example.
Capturing camera frames, display, routing and machine actions are not timed.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be between 0 and 1.")
    if args.warmup < 1:
        parser.error("Use at least one warmup request.")
    if args.output.exists():
        parser.error("Output exists; choose a new output filename.")
    if not args.images.is_dir():
        parser.error("The input must be a directory of test images.")
    files = sorted(p for p in args.images.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not files:
        parser.error("No supported images found.")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        parser.error("Set ROBOFLOW_API_KEY first.")

    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    client = InferenceHTTPClient(api_url="http://localhost:9001", api_key=api_key).configure(
        InferenceConfiguration(api_key_transport="header", confidence_threshold=args.confidence))
    for i in range(args.warmup):
        client.infer(str(files[i % len(files)]), model_id=args.model)

    durations, failures, rows = [], [], []
    start_all = time.perf_counter()
    for path in files:
        start = time.perf_counter()
        try:
            result = client.infer(str(path), model_id=args.model)
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            rows.append({"image": path.name, "seconds": elapsed, "result": result})
        except Exception as exc:
            failures.append({"image": path.name, "seconds": time.perf_counter() - start,
                             "error_type": type(exc).__name__})
    wall = time.perf_counter() - start_all
    summary = {
        "model": args.model, "confidence": args.confidence,
        "measurement": "sequential local image-to-result requests; no camera capture",
        "attempted_images": len(files), "successful_images": len(durations),
        "failed_images": len(failures), "warmup_requests": args.warmup,
        "median_ms_successful": statistics.median(durations) * 1000 if durations else None,
        "p95_ms_successful": percentile(durations, 0.95) * 1000 if durations else None,
        "wall_seconds_including_failures": wall,
        "completed_images_per_second": len(durations) / wall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "failures": failures,
                                       "requests": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
