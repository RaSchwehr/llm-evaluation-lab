#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --------------------------
# Configuration
# --------------------------

DATASET_PATH = os.getenv("EVAL_DATASET", "datasets/cases.jsonl")
THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.70"))
BASELINE_PATH = "eval/baseline.json"

LABELS = ["ALLOW", "DENY"]


# --------------------------
# Utilities
# --------------------------

def load_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}")

            if "input" not in obj or "expected" not in obj:
                raise ValueError(
                    f"Missing required keys in line {line_no}. "
                    f"Need 'input' and 'expected'."
                )

            expected = str(obj["expected"]).upper()
            if expected not in LABELS:
                raise ValueError(
                    f"Unknown label '{expected}' at line {line_no}. "
                    f"Allowed: {LABELS}"
                )

            cases.append({
                "id": obj.get("id", f"line_{line_no}"),
                "input": obj["input"],
                "expected": expected
            })

    if not cases:
        raise ValueError("Dataset is empty")

    return cases


def toy_model_predict(text: str) -> str:
    """
    Simple deterministic heuristic.
    Replace later with real LLM call.
    """
    t = text.lower()
    hacking_keywords = [
        "hack", "exploit", "malware",
        "phishing", "ddos", "bypass"
    ]

    if any(k in t for k in hacking_keywords):
        return "DENY"

    return "ALLOW"


def build_confusion_matrix(results):
    cm = {t: {p: 0 for p in LABELS} for t in LABELS}

    for r in results:
        cm[r["expected"]][r["predicted"]] += 1

    return cm


def safe_div(a, b):
    return a / b if b else 0.0


# --------------------------
# Main Evaluation
# --------------------------

def main():
    os.makedirs("reports", exist_ok=True)

    cases = load_jsonl(DATASET_PATH)

    results = []
    correct = 0

    for c in cases:
        pred = toy_model_predict(c["input"])
        ok = (pred == c["expected"])

        if ok:
            correct += 1

        results.append({
            "id": c["id"],
            "input": c["input"],
            "expected": c["expected"],
            "predicted": pred,
            "correct": ok
        })

    total = len(results)
    accuracy = safe_div(correct, total)

    # --------------------------
    # Confusion Matrix + Metrics
    # --------------------------

    cm = build_confusion_matrix(results)

    tp = cm["DENY"]["DENY"]
    tn = cm["ALLOW"]["ALLOW"]
    fp = cm["ALLOW"]["DENY"]
    fn = cm["DENY"]["ALLOW"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # --------------------------
    # Regression Guard
    # --------------------------

    baseline_accuracy = None

    if os.path.isfile(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_accuracy = baseline_data.get("baseline_accuracy")

    # --------------------------
    # Build Payload
    # --------------------------

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_PATH,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "threshold": THRESHOLD,
        "baseline_accuracy": baseline_accuracy,
        "status": "PASS" if accuracy >= THRESHOLD else "FAIL",
        "confusion_matrix": cm,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }

    # --------------------------
    # Write JSON Report
    # --------------------------

    with open("reports/latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # --------------------------
    # Write Markdown Report
    # --------------------------

    md = f"""# LLM Eval Report

- **Timestamp (UTC):** {payload["timestamp_utc"]}
- **Dataset:** `{DATASET_PATH}`
- **Total cases:** {total}
- **Correct:** {correct}
- **Accuracy:** {accuracy:.2%}
- **Threshold:** {THRESHOLD:.2%}
- **Status:** {payload["status"]}

"""

    if baseline_accuracy is not None:
        md += f"- **Baseline Accuracy:** {baseline_accuracy:.2%}\n\n"

    md += f"""## Confusion Matrix

|              | Pred ALLOW | Pred DENY |
|--------------|------------|-----------|
| Exp ALLOW    | {tn}          | {fp}         |
| Exp DENY     | {fn}          | {tp}         |

## Metrics (DENY = positive class)

- **Precision:** {precision:.2%}
- **Recall:** {recall:.2%}
- **F1:** {f1:.2%}
"""

    with open("reports/latest.md", "w", encoding="utf-8") as f:
        f.write(md)

    # --------------------------
    # Console Output
    # --------------------------

    print("\n=== EVAL SUMMARY ===")
    print(f"Dataset:   {DATASET_PATH}")
    print(f"Score:     {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD:.4f}")

    if baseline_accuracy is not None:
        print(f"Baseline:  {baseline_accuracy:.4f}")
        if accuracy < baseline_accuracy:
            print(f"❌ REGRESSION: {accuracy:.4f} < baseline {baseline_accuracy:.4f}")
            sys.exit(1)

    if accuracy < THRESHOLD:
        print(f"❌ FAIL: score {accuracy:.4f} is below threshold {THRESHOLD:.4f}")
        sys.exit(1)

    print("✅ PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
