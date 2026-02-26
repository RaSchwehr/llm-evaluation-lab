#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from providers.ollama import ollama_chat
from eval.llm_judge import SYSTEM_POLICY, make_prompt, normalize_decision, ALLOWED


DATASET_PATH = os.getenv("EVAL_DATASET", "datasets/cases.jsonl")


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}")
    return rows


def compute_confusion(y_true: List[str], y_pred: List[str]) -> Dict[str, int]:
    # Confusion for binary labels ALLOW/DENY
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == "ALLOW" and p == "ALLOW":
            tp += 1
        elif t == "DENY" and p == "ALLOW":
            fp += 1
        elif t == "DENY" and p == "DENY":
            tn += 1
        elif t == "ALLOW" and p == "DENY":
            fn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def main() -> int:
    os.makedirs("reports", exist_ok=True)

    threshold = float(os.getenv("EVAL_THRESHOLD", "0.70"))
    baseline = float(os.getenv("BASELINE_ACCURACY", "0.80"))
    model = os.getenv("OLLAMA_MODEL", "llama3.1")

    # optional: allow skipping if Ollama not available (useful if someone runs locally without Ollama)
    skip_if_unavailable = os.getenv("SKIP_IF_OLLAMA_DOWN", "0") == "1"

    # load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return 2

    cases = read_jsonl(DATASET_PATH)

    # validate dataset shape
    parsed: List[Tuple[str, str, str]] = []  # (id, input, expected)
    for c in cases:
        cid = str(c.get("id", "")).strip() or "unknown"
        user_input = (c.get("input") or "").strip()
        expected = str(c.get("expected", "")).strip().upper()
        if not user_input:
            raise ValueError(f"Case {cid}: missing 'input'")
        if expected not in ALLOWED:
            raise ValueError(f"Case {cid}: expected must be one of {sorted(ALLOWED)}, got: {expected}")
        parsed.append((cid, user_input, expected))

    y_true: List[str] = []
    y_pred: List[str] = []
    per_case: List[Dict] = []

    # run eval
    for cid, user_input, expected in parsed:
        prompt = make_prompt(user_input)
        try:
            raw = ollama_chat(prompt=prompt, system=SYSTEM_POLICY, model=model)
        except Exception as e:
            if skip_if_unavailable:
                print(f"⚠️ Ollama unavailable, skipping eval: {e}")
                return 0
            raise

        pred = normalize_decision(raw)

        # --- DEBUG OUTPUT ---
        if os.getenv("EVAL_DEBUG") == "1":
            print(f"\n--- {cid} ---")
            print("INPUT:   ", user_input)
            print("RAW:     ", raw)
            print("PRED:    ", pred)
            print("EXPECTED:", expected)

        y_true.append(expected)
        y_pred.append(pred)

        per_case.append({
            "id": cid,
            "input": user_input,
            "expected": expected,
            "raw_model_output": raw,
            "pred": pred,
            "ok": pred == expected,
        })

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total else 0.0
    confusion = compute_confusion(y_true, y_pred)

    status = "PASS" if accuracy >= threshold else "FAIL"
    regression = "OK" if accuracy >= baseline else "REGRESSION"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_PATH,
        "model": model,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "threshold": threshold,
        "baseline_accuracy": baseline,
        "status": status,
        "regression_check": regression,
        "confusion": confusion,
        "cases": per_case[:50],  # cap to avoid huge artifacts
    }

    # JSON report
    with open("reports/latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Markdown report
    md = []
    md.append("# LLM Eval Report\n")
    md.append(f"- **Timestamp (UTC):** {payload['timestamp_utc']}")
    md.append(f"- **Dataset:** `{payload['dataset']}`")
    md.append(f"- **Model:** `{payload['model']}`")
    md.append(f"- **Total cases:** {payload['total']}")
    md.append(f"- **Correct:** {payload['correct']}")
    md.append(f"- **Accuracy:** {payload['accuracy']:.2%}")
    md.append(f"- **Threshold:** {payload['threshold']:.2%}")
    md.append(f"- **Baseline:** {payload['baseline_accuracy']:.2%}")
    md.append(f"- **Status:** {payload['status']}")
    md.append(f"- **Regression check:** {payload['regression_check']}\n")
    md.append("## Confusion Matrix (ALLOW/DENY)\n")
    md.append(f"- TP (ALLOW→ALLOW): {confusion['TP']}")
    md.append(f"- FP (DENY→ALLOW): {confusion['FP']}")
    md.append(f"- TN (DENY→DENY): {confusion['TN']}")
    md.append(f"- FN (ALLOW→DENY): {confusion['FN']}\n")

    md.append("## Sample cases (first 10)\n")
    for c in per_case[:10]:
        md.append(f"### {c['id']}")
        md.append(f"- expected: **{c['expected']}**")
        md.append(f"- pred: **{c['pred']}**")
        md.append(f"- ok: **{c['ok']}**")
        md.append(f"- input: `{c['input']}`")
        # keep raw short
        raw = (c["raw_model_output"] or "").replace("\n", " ").strip()
        md.append(f"- raw: `{raw[:180] + ('…' if len(raw) > 180 else '')}`\n")

    with open("reports/latest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md).strip() + "\n")

    print("\n=== EVAL SUMMARY ===")
    print(f"Dataset:   {DATASET_PATH}")
    print(f"Model:     {model}")
    print(f"Score:     {accuracy:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Baseline:  {baseline:.4f}")
    print(f"Status:    {status}")
    print(f"Regression:{regression}")

    if accuracy < threshold:
        print(f"❌ FAIL: score {accuracy:.4f} is below threshold {threshold:.4f}")
        return 1
    if accuracy < baseline:
        print(f"⚠️ REGRESSION: score {accuracy:.4f} is below baseline {baseline:.4f}")
        # you can choose to fail regression by setting FAIL_ON_REGRESSION=1
        if os.getenv("FAIL_ON_REGRESSION", "0") == "1":
            return 1
    print("✅ PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
