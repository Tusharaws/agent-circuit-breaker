import os
from pathlib import Path

_DEFAULT_POLICY_FILE = Path(__file__).parent / "config" / "redaction_policy.md"
_ENV_OVERRIDE_VAR = "SANITIZER_REDACTION_POLICY_PATH"

VALID_ACTIONS = {"mask", "hash", "drop"}


def _resolve_policy_path(path: Path | None) -> Path:
    if path is not None:
        return path
    override = os.environ.get(_ENV_OVERRIDE_VAR)
    if override:
        return Path(override)
    return _DEFAULT_POLICY_FILE


def load_redaction_policy(path: Path | None = None) -> dict[str, str]:
    """Parse `- data_type: action` bulleted entries out of a markdown
    config file into a {data_type: action} mapping.

    Resolution order: explicit `path` argument, else the
    `SANITIZER_REDACTION_POLICY_PATH` environment variable (an operator
    can point this at an external file to change scrub behavior without a
    code change or redeploy), else the bundled default file.
    """
    target = _resolve_policy_path(path)
    policy: dict[str, str] = {}
    for line in target.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        body = stripped[1:].strip()
        if ":" not in body:
            continue
        data_type, _, action = body.partition(":")
        data_type = data_type.strip()
        action = action.strip()
        if not data_type:
            continue
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"invalid redaction action {action!r} for data type {data_type!r}; "
                f"must be one of {sorted(VALID_ACTIONS)}"
            )
        policy[data_type] = action
    return policy


def get_action(policy: dict[str, str], data_type: str, default: str = "mask") -> str:
    """The configured action for `data_type`, or `default` if unlisted."""
    return policy.get(data_type, default)
