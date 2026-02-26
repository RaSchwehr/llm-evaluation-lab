#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


DATASET_PATH = Path("datasets/cases.jsonl")
REPORT_DIR = Path("reports")
LATEST_JSON = REPORT_DIR / "latest.json"
LATEST_MD = REPORT_DIR / "latest.md"


def normalize_label(x: str) -> str:
    return str(x).strip().upper()


def load_cases(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}: {e}") from e

            if "input" not in obj or "expected" not in obj:
                raise ValueError(
                    f"Case missing required fields on line {lineno}: "
                    f"needs 'input' and 'expected'. Got keys={list(obj.keys())}"
                )

            obj["expected"] = normalize_label(obj["expected"])
            obj["id"] = str(obj.get("id", f"line-{lineno}"))
            cases.append(obj)

    if not cases:
        raise ValueError(f"No cases found in dataset: {path}")
    return cases


def predict_label(case: Dict) -> str:
    """
    PLACEHOLDER "model":
    - Replace this with a real model/LLM call later.
    - For now: deterministic heuristic so the pipeline is real.
    """
    text = case["input"].lower()

    # Example policy-ish heuristic (dummy but deterministic)
    deny_triggers = ["hack", "exploit", "password", "bypass", "malware", "ddos"]
    if any(t in text for t in deny_triggers):
        return "DENY"

    allow_triggers = ["refund", "delete my data", "privacy", "cancel", "support"]
    if any(t in text for t in allow_triggers):
        return "ALLOW"

    # default conservative
    return "DENY"


def evaluate(cases: List[Dict]) -> Tuple[float, List[Dict]]:
    rows = []
    correct = 0

    for c in cases:
        pred = normalize_label(predict_label(c))
        exp = c["expected"]
        ok = (pred == exp)
        correct += 1 if ok else 0

        rows.append(
            {
                "id": c["id"],
                "expected": exp,
                "predicted": pred,
                "correct": ok,
                "input": c["input"],
            }
        )

    total = len(cases)
    acc = correct / total
    return acc, rows


def write_reports(accuracy: float, threshold: float, rows: List[Dict]) -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    status = "PASS" if accuracy >= threshold else "FAIL"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "threshold": threshold,
        "status": status,
        "failed_cases": [
            {"id": r["id"], "expected": r["expected"], "predicted": r["predicted"]}
            for r in rows
            if not r["correct"]
        ],
    }

    # JSON
    with LATEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Markdown summary
    top_fail = payload["failed_cases"][:10]
    fail_lines = "\n".join(
        [f"- `{x['id']}` expected **{x['expected']}** got **{x['predicted']}**" for x in top_fail]
    ) or "_No failures_"

    md = f"""# LLM Eval Report

- **Timestamp (UTC):** {payload["timestamp_utc"]}
- **Dataset:** `{payload["dataset"]}`
- **Total cases:** {payload["total"]}
- **Correct:** {payload["correct"]}
- **Accuracy:** {payload["accuracy"]:.2%}
- **Threshold:** {payload["threshold"]:.2%}
- **Status:** **{payload["status"]}**

## Top failures (max 10)
{fail_lines}
"""
    with LATEST_MD.open("w", encoding="utf-8") as f:
        f.write(md)

    return payload


def main() -> int:
    threshold = float(os.getenv("EVAL_THRESHOLD", "0.70"))

    cases = load_cases(DATASET_PATH)
    accuracy, rows = evaluate(cases)
    payload = write_reports(accuracy, threshold, rows)

    print("\n=== EVAL SUMMARY ===")
    print(f"Dataset:   {payload['dataset']}")
    print(f"Score:     {accuracy:.4f}")
    print(f"Threshold: {threshold:.4f}")

    if accuracy < threshold:
        print(f"❌ FAIL: score {accuracy:.4f} is below threshold {threshold:.4f}")
        return 1
    else:
        print(f"✅ PASS: score {accuracy:.4f} meets threshold {threshold:.4f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
