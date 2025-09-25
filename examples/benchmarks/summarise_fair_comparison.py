#!/usr/bin/env python3
import json
import os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "fair_comparison_results.json")


def main() -> None:
    if not os.path.exists(RESULTS):
        print("fair_comparison_results.json not found at:", RESULTS)
        return

    with open(RESULTS, "r") as f:
        data = json.load(f)

    # Expect structure: { benchmark_name: { method: [times] or metrics } }
    print("=== Fair Comparison Summary ===")
    for bench, entries in data.items():
        if isinstance(entries, dict):
            print(f"\n[{bench}]")
            for method, vals in entries.items():
                if isinstance(vals, list) and vals and all(isinstance(x, (int, float)) for x in vals):
                    print(f"- {method}: n={len(vals)}, avg={mean(vals):.6f}")
                elif isinstance(vals, dict):
                    # Print key metrics if numeric
                    metrics = {k: v for k, v in vals.items() if isinstance(v, (int, float))}
                    if metrics:
                        summary = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                        print(f"- {method}: {summary}")
                    else:
                        print(f"- {method}: (non-numeric summary)")
                else:
                    print(f"- {method}: (unrecognized format)")
        else:
            print(f"{bench}: (unexpected format)")


if __name__ == "__main__":
    main()
