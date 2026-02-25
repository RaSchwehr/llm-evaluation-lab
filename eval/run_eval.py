#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone

def main():
    os.makedirs("reports", exist_ok=True)

    # --- Dummy-Eval (hier später deine echte Logik einbauen) ---
    # Minimal: wir tun so, als hätten wir 10 Tests, 8 korrekt
    total = 10
    correct = 8
    accuracy = correct / total

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }

    # JSON Report
    json_path = "reports/latest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Markdown Report (für schnelle menschliche Sicht)
    md_path = "reports/latest.md"
    md = f"""# LLM Eval Report

- **Timestamp (UTC):** {payload["timestamp_utc"]}
- **Total cases:** {payload["total"]}
- **Correct:** {payload["correct"]}
- **Accuracy:** {payload["accuracy"]:.2%}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote {json_path} and {md_path}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
