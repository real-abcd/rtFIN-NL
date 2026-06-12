#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_results.py <run_dir>", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1])
    results = {}

    candidates = {
        "bge": run_dir / "bge" / "eval_results_bge.json",
        "qwen": run_dir / "qwen" / "eval_results_qwen.json",
    }

    for name, path in candidates.items():
        data = load_json(path)
        if data is not None:
            results[name] = data

    summary = {
        "run_dir": str(run_dir),
        "results": results,
    }

    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
