#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone


def load_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs("reports", exist_ok=True)

    # --- Simulierte Evaluation (später ersetzt du das mit echter Logik) ---
    total = 10
    correct = 8
    accuracy = correct / total

    # Quality Gate: Mindestqualität
    threshold = float(os.getenv("EVAL_THRESHOLD", "0.70"))

    # Regression Gate: wieviel schlechter darf es werden (z.B. 0.02 = 2 Prozentpunkte)
    max_drop = float(os.getenv("EVAL_MAX_DROP", "0.02"))

    # Baseline laden (committed Reference)
    baseline_path = os.getenv("EVAL_BASELINE_PATH", "reports/baseline.json")
    baseline = load_json(baseline_path)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "threshold": threshold,
        "baseline_path": baseline_path,
        "baseline_accuracy": None,
        "max_drop": max_drop,
        "regression_drop": None,
        "regression_status": "SKIP",
        "status": "PASS" if accuracy >= threshold else "FAIL",
    }

    # Regression Check (nur wenn Baseline existiert)
    if baseline and isinstance(baseline, dict) and "accuracy" in baseline:
        base_acc = float(baseline["accuracy"])
        drop = base_acc - accuracy  # positiv = schlechter geworden
        payload["baseline_accuracy"] = base_acc
        payload["regression_drop"] = drop
        payload["regression_status"] = "PASS" if drop <= max_drop else "FAIL"

    # JSON Report
    with open("reports/latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Markdown Report
    md = f"""# LLM Eval Report

- **Timestamp (UTC):** {payload["timestamp_utc"]}
- **Total cases:** {payload["total"]}
- **Correct:** {payload["correct"]}
- **Accuracy:** {payload["accuracy"]:.2%}
- **Threshold:** {payload["threshold"]:.2%}
- **Status:** {payload["status"]}

## Regression Gate

- **Baseline file:** `{payload["baseline_path"]}`
- **Baseline accuracy:** {("N/A" if payload["baseline_accuracy"] is None else f'{payload["baseline_accuracy"]:.2%}')}
- **Allowed max drop:** {payload["max_drop"]:.2%}
- **Actual drop:** {("N/A" if payload["regression_drop"] is None else f'{payload["regression_drop"]:.2%}')}
- **Regression status:** {payload["regression_status"]}
"""
    with open("reports/latest.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n=== EVAL SUMMARY ===")
    print(f"Score:     {accuracy:.4f}")
    print(f"Threshold: {threshold:.4f}")

    # 1) Mindestqualität
    if accuracy < threshold:
        print(f"❌ FAIL: score {accuracy:.4f} is below threshold {threshold:.4f}")
        sys.exit(1)

    # 2) Regression Gate (wenn Baseline existiert)
    if payload["regression_status"] == "FAIL":
        base_acc = payload["baseline_accuracy"]
        drop = payload["regression_drop"]
        print(f"❌ FAIL: regression detected. Baseline={base_acc:.4f}, score={accuracy:.4f}, drop={drop:.4f} > max_drop={max_drop:.4f}")
        sys.exit(1)

    print("✅ PASS: quality gate satisfied (threshold + regression)")
    sys.exit(0)


if __name__ == "__main__":
    main()
