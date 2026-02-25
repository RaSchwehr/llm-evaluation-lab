import json
from pathlib import Path

PROMPT_PATH = Path("prompts/decision_prompt.txt")
DATASET_PATH = Path("datasets/cases.jsonl")
OUT_PATH = Path("reports/results.jsonl")


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_cases():
    cases = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def naive_baseline_risk(s: str) -> str:
    s_low = s.lower()
    if "without human" in s_low or "no human" in s_low or "without human review" in s_low:
        return "HIGH"
    if "bank" in s_low or "credit" in s_low or "loan" in s_low:
        return "MEDIUM"
    return "LOW"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt()
    cases = load_cases()

    correct = 0
    total = 0

    for c in cases:
        pred = naive_baseline_risk(c["scenario"])
        expected = c.get("expected_risk_level")

        total += 1
        if expected and pred == expected:
            correct += 1

        record = {
            "id": c["id"],
            "scenario": c["scenario"],
            "expected_risk_level": expected,
            "predicted_risk_level": pred,
            "note": "baseline_heuristic",
            "prompt_used": "decision_prompt.txt",
        }
        OUT_PATH.write_text("", encoding="utf-8") if total == 1 else None
        with OUT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = correct / total if total else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "report_file": str(OUT_PATH),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
