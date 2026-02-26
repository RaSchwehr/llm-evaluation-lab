#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

# Ralph Ray Schwehr
# ---- Config ----
PROMPTS_DIR = Path("prompts")
THRESHOLDS_FILE = Path("thresholds.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def load_thresholds():
    if not THRESHOLDS_FILE.exists():
        die(f"Missing {THRESHOLDS_FILE}. Create it (see template).")
    return json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))

def discover_prompt_files():
    if not PROMPTS_DIR.exists():
        die(f"Missing prompts folder: {PROMPTS_DIR}/")
    files = sorted(list(PROMPTS_DIR.rglob("*.txt")) + list(PROMPTS_DIR.rglob("*.md")))
    if not files:
        die(f"No prompt files found in {PROMPTS_DIR}/ (expected .txt or .md).")
    return files

def simple_score(text: str) -> float:
    """
    Placeholder scoring:
    - deterministic & cheap
    - replace later with real LLM calls or rubric-based scoring
    """
    # toy heuristic: length-based score clipped to [0, 1]
    n = len(text.strip())
    return max(0.0, min(1.0, n / 800.0))

def run_eval():
    thresholds = load_thresholds()
    prompt_files = discover_prompt_files()

    runs = []
    total = 0
    passed = 0

    for p in prompt_files:
        content = p.read_text(encoding="utf-8")
        score = simple_score(content)

        gate = thresholds.get("min_score", 0.5)
        ok = score >= gate

        total += 1
        passed += 1 if ok else 0

        runs.append({
            "prompt_file": str(p),
            "score": round(score, 4),
            "min_required": gate,
            "pass": ok
        })

    pass_rate = passed / total if total else 0.0

    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": thresholds.get("min_pass_rate", 1.0),
        "min_score": thresholds.get("min_score", 0.5),
        "results": runs,
    }

    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Gate 1: pass rate
    if pass_rate < thresholds.get("min_pass_rate", 1.0):
        die(f"Pass rate {pass_rate:.2%} < required {thresholds.get('min_pass_rate', 1.0):.2%}")

    # Gate 2: any failing prompt
    failing = [r for r in runs if not r["pass"]]
    if failing:
        die(f"{len(failing)} prompt(s) failed min_score gate.")

    print("OK: All thresholds satisfied.")

if __name__ == "__main__":
    run_eval()
