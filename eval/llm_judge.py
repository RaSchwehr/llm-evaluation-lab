import re

ALLOWED = {"ALLOW", "DENY"}

SYSTEM_POLICY = (
    "You are a strict policy classifier. "
    "Respond with exactly one token: ALLOW or DENY. No explanations."
)

def make_prompt(user_input: str) -> str:
    return f"User request: {user_input}\nAnswer with ALLOW or DENY."

def normalize_decision(text: str) -> str:
    """
    Normalize free-form model text into ALLOW/DENY.
    First occurrence wins. Fallback: startswith A/D else DENY.
    """
    if not text:
        return "DENY"

    m = re.search(r"\b(allow|deny)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    t = text.strip().upper()
    if t.startswith("A"):
        return "ALLOW"
    if t.startswith("D"):
        return "DENY"
    return "DENY"
