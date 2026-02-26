#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LATEST_JSON = REPORTS / "latest.json"
LATEST_MD = REPORTS / "latest.md"

MODELS_DEFAULT = ["gpt-oss:20b", "llama3.1"]

def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")

def run_one(model: str, threshold: str, debug: bool) -> dict:
    env = os.environ.copy()
    env["OLLAMA_MODEL"] = model
    env["EVAL_THRESHOLD"] = threshold
    if debug:
        env["EVAL_DEBUG"] = "1"
    else:
        env.pop("EVAL_DEBUG", None)

    print(f"\n==============================")
    print(f"RUN MODEL: {model}")
    print(f"THRESHOLD: {threshold} | DEBUG: {int(debug)}")
    print(f"==============================\n")

    # run eval
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "eval" / "run_eval.py")],
        env=env,
        cwd=str(REPO_ROOT),
        text=True,
    )

    # Ensure reports exist
    if not LATEST_JSON.exists():
        raise RuntimeError(f"Expected report not found: {LATEST_JSON}")

    with open(LATEST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["_model"] = model
    data["_exit_code"] = proc.returncode
    return data

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    threshold = os.getenv("EVAL_THRESHOLD", "0.70")
    debug = os.getenv("EVAL_DEBUG") == "1"

    models_env = os.getenv("COMPARE_MODELS", "").strip()
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else MODELS_DEFAULT

    results = []
    frozen = []

    for m in models:
        res = run_one(m, threshold, debug)
        results.append(res)

        tag = safe_name(m)
        frozen_json = REPORTS / f"latest_{tag}.json"
        frozen_md = REPORTS / f"latest_{tag}.md"

        shutil.copyfile(LATEST_JSON, frozen_json)
        if LATEST_MD.exists():
            shutil.copyfile(LATEST_MD, frozen_md)

        frozen.append((m, frozen_json.name, frozen_md.name if frozen_md.exists() else None))

    # Write compare report
    ts = datetime.now(timezone.utc).isoformat()
    compare = {
        "timestamp_utc": ts,
        "threshold": float(threshold),
        "models": results,
        "frozen_reports": [{"model": m, "json": j, "md": md} for (m, j, md) in frozen],
    }

    compare_json = REPORTS / "compare.json"
    with open(compare_json, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2)

    # Markdown summary
    lines = []
    lines.append("# Model Comparison")
    lines.append("")
    lines.append(f"- **Timestamp (UTC):** {ts}")
    lines.append(f"- **Threshold:** {float(threshold):.2%}")
    lines.append("")
    lines.append("| Model | Score | Status | Exit | Frozen JSON | Frozen MD |")
    lines.append("|---|---:|---|---:|---|---|")

    def fmt_status(r):
        # try common keys; fall back gracefully
        status = r.get("status") or r.get("Status") or ("PASS" if r.get("accuracy", 0) >= float(threshold) else "FAIL")
        return str(status)

    def fmt_score(r):
        # supports either accuracy or score key
        if "accuracy" in r:
            return float(r["accuracy"])
        if "score" in r:
            return float(r["score"])
        # if none exists, try to infer from correct/total
        if "correct" in r and "total" in r and r["total"]:
            return float(r["correct"]) / float(r["total"])
        return float("nan")

    for r in results:
        m = r.get("_model", "?")
        score = fmt_score(r)
        status = fmt_status(r)
        exit_code = r.get("_exit_code", -1)

        tag = safe_name(m)
        j = f"latest_{tag}.json"
        md = f"latest_{tag}.md" if (REPORTS / f"latest_{tag}.md").exists() else "-"
        lines.append(f"| `{m}` | {score:.4f} | **{status}** | {exit_code} | `{j}` | `{md}` |")

    compare_md = REPORTS / "compare.md"
    with open(compare_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n✅ Wrote:")
    print(f" - {compare_json}")
    print(f" - {compare_md}")
    print("\nTip: open reports/compare.md\n")

if __name__ == "__main__":
    main()
