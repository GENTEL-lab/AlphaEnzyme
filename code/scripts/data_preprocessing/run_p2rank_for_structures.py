#!/usr/bin/env python
"""Run P2Rank on Protenix-predicted CIF structures with resume support."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P2Rank for predicted CIF structures")
    parser.add_argument("--structure_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--p2rank_bin", type=Path, required=True)
    parser.add_argument("--java_home", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads_per_job", type=int, default=2)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()


def run_one(cif_path: Path, args: argparse.Namespace) -> dict[str, str]:
    expected = args.output_dir / f"{cif_path.name}_predictions.csv"
    if expected.exists():
        return {"status": "skip", "cif": str(cif_path), "output": str(expected)}

    env = {
        "JAVA_HOME": str(args.java_home),
        "PATH": f"{args.java_home / 'bin'}:{subprocess.os.environ.get('PATH', '')}",
        **subprocess.os.environ,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.p2rank_bin),
        "predict",
        "-f",
        str(cif_path),
        "-o",
        str(args.output_dir),
        "-threads",
        str(args.threads_per_job),
        "-visualizations",
        "0",
    ]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        return {"status": "fail", "cif": str(cif_path), "returncode": str(proc.returncode), "log": proc.stdout[-4000:]}
    return {"status": "ok", "cif": str(cif_path), "output": str(expected)}


def main() -> None:
    args = parse_args()
    cif_files = sorted(args.structure_dir.glob("*/predictions/*_sample_0.cif"))
    end_idx = len(cif_files) if args.end_idx < 0 else min(args.end_idx, len(cif_files))
    cif_files = cif_files[args.start_idx:end_idx]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, cif_path, args) for cif_path in cif_files]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "total": len(results),
        "ok": sum(r["status"] == "ok" for r in results),
        "skip": sum(r["status"] == "skip" for r in results),
        "fail": sum(r["status"] == "fail" for r in results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "p2rank_resume_report.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
