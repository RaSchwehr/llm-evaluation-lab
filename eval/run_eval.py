#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone


def main():
    os.makedirs("reports", exist_ok=True)

    # --- Simulierte Evaluation ---
    total = 10
    correct = 8
    accuracy = correct / total

    threshold = float(os.getenv("EVAL_THRESHOLD", "0.70"))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "threshold": threshold,
        "status": "PASS" if accuracy >= threshold else "FAIL",
    }

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
"""
    with open("reports/latest.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n=== EVAL SUMMARY ===")
    print(f"Score:     {accuracy:.4f}")
    print(f"Threshold: {threshold:.4f}")

    if accuracy < threshold:
        print(f"❌ FAIL: score {accuracy:.4f} is below threshold {threshold:.4f}")
        sys.exit(1)
    else:
        print(f"✅ PASS: score {accuracy:.4f} meets threshold {threshold:.4f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
