import hashlib
import re

from sanitizer.policy import get_action

_PATTERNS: dict[str, re.Pattern] = {
    # Bounded quantifiers throughout -- a security review (Phase 7) found
    # the original unbounded `[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}` was quadratic-
    # time on any input with a long word-character run and no '@' (measured
    # 14.6s on a 100,000-char adversarial string; a wholly realistic shape
    # for a large tool-output blob or long token/hash). The bounds below
    # are real RFC 5321 (local part <= 64) / DNS (label <= 63) limits, not
    # arbitrary -- no legitimate email exceeds them. `(?:[\w-]{1,63}\.)+`
    # (one-or-more dot-separated labels) also fixes a correctness gap the
    # bound exposed: the original pattern only redacted a single leading
    # domain label, leaking multi-label domains like "sub.example.co.uk"
    # down to ".co.uk".
    "email": re.compile(r"[\w.+-]{1,64}@(?:[\w-]{1,63}\.)+[a-zA-Z]{2,24}"),
    "credit_card": re.compile(
        r"\b(?:"
        r"4[0-9]{12}(?:[0-9]{3})?"          # Visa
        r"|5[1-5][0-9]{14}"                 # MasterCard
        r"|3[47][0-9]{13}"                  # Amex
        r"|6(?:011|5[0-9]{2})[0-9]{12}"     # Discover
        r")\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b|\bAKIA[0-9A-Z]{16}\b"),
    "phone": re.compile(
        r"(?:\+\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)|\d{2,4})[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    ),
}

# More specific/narrow patterns must run before the permissive phone
# pattern, or a credit card / SSN digit run could get partially matched
# and mangled by it first. Each pattern only sees what earlier ones left.
_SCRUB_ORDER = ["email", "credit_card", "ssn", "api_key", "phone"]


def apply_action(match_text: str, action: str) -> str:
    """The shared redaction-action vocabulary (mask/hash/drop) both
    regex-based (this module) and NER-based (ner.py) scrubbing apply --
    one implementation so both paths produce visibly consistent output."""
    if action == "drop":
        return ""
    if action == "hash":
        digest = hashlib.sha256(match_text.encode("utf-8")).hexdigest()[:12]
        return f"[REDACTED:{digest}]"
    return "[REDACTED]"  # mask (also the default for an unrecognized action)


def scrub_text(text: str, policy: dict[str, str]) -> str:
    """Detect and redact email/credit_card/ssn/api_key/phone data in
    `text`, applying whatever action `policy` configures per data type
    (falling back to `get_action`'s default for an unlisted type)."""
    result = text
    for data_type in _SCRUB_ORDER:
        pattern = _PATTERNS[data_type]
        action = get_action(policy, data_type)
        result = pattern.sub(lambda m: apply_action(m.group(0), action), result)
    return result
