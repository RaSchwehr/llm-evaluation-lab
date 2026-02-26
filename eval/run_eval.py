cat > eval/run_eval.py <<'EOF'
#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from collections import Counter

DATASET_PATH = os.getenv("EVAL_DATASET", "datasets/cases.jsonl")
THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.70"))

def load_jsonl(path: str):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases

def dummy_model_predict(case: dict) -> str:
    """
    Platzhalter-Model: simple heuristic.
    - Wenn 'hacking' im input => DENY, sonst ALLOW
    """
    text = (case.get("input") or "").lower()
    return "DENY" if "hack" in text else "ALLOW"

def main():
    os.makedirs("reports", exist_ok=True)

    if not os.path.isfile(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        sys.exit(2)

    cases = load_jsonl(DATASET_PATH)
    if not cases:
        print("❌ Dataset empty")
        sys.exit(2)

    y_true = []
    y_pred = []

    for c in cases:
        expected = c.get("expected")
        if expected is None:
            print(f"❌ Missing 'expected' in case: {c}")
            sys.exit(2)
        pred = dummy_model_predict(c)
        y_true.append(expected)
        y_pred.append(pred)

    total = len(cases)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / total

    labels = sorted(set(y_true) | set(y_pred))
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_PATH,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "threshold": THRESHOLD,
        "status": "PASS" if accuracy >= THRESHOLD else "FAIL",
        "labels": labels,
        "confusion_matrix": cm,
    }

    with open("reports/latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Markdown Summary
    md_lines = [
        "# LLM Eval Report",
        "",
        f"- **Timestamp (UTC):** {payload['timestamp_utc']}",
        f"- **Dataset:** `{payload['dataset']}`",
        f"- **Total cases:** {payload['total']}",
        f"- **Correct:** {payload['correct']}",
        f"- **Accuracy:** {payload['accuracy']:.2%}",
        f"- **Threshold:** {payload['threshold']:.2%}",
        f"- **Status:** {payload['status']}",
        "",
        "## Confusion Matrix",
        "",
    ]

    # Table header
    header = "| true \\ pred | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    md_lines.append(header)
    md_lines.append(sep)

    for t in labels:
        row = [t] + [str(cm[t][p]) for p in labels]
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("")
    with open("reports/latest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n=== EVAL SUMMARY ===")
    print(f"Dataset:   {DATASET_PATH}")
    print(f"Score:     {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD:.4f}")

    if accuracy < THRESHOLD:
        print(f"❌ FAIL: score {accuracy:.4f} is below threshold {THRESHOLD:.4f}")
        sys.exit(1)
    else:
        print(f"✅ PASS: score {accuracy:.4f} meets threshold {THRESHOLD:.4f}")
        sys.exit(0)

if __name__ == "__main__":
    main()
EOF
