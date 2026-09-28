"""Tests for the redaction policy config (Phase 2).

Written before the implementation (sanitizer.policy does not exist yet) --
run `pytest` to see them fail with a collection error until
src/sanitizer/policy.py exists.
"""
import pytest

import sanitizer.policy as policy_module
from sanitizer.policy import VALID_ACTIONS, get_action, load_redaction_policy


# ---------------------------------------------------------------------------
# Happy path: bundled default config
# ---------------------------------------------------------------------------


def test_bundled_default_policy_loads_and_uses_only_valid_actions():
    policy = load_redaction_policy()
    assert policy  # non-empty -- ships with a real starting point
    assert set(policy.values()) <= VALID_ACTIONS


def test_valid_config_parses_into_expected_mapping(tmp_path):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text("# Redaction Policy\n\n- email: mask\n- ssn: drop\n- api_key: hash\n")

    policy = load_redaction_policy(path=config_file)

    assert policy == {"email": "mask", "ssn": "drop", "api_key": "hash"}


@pytest.mark.parametrize("action", sorted(VALID_ACTIONS))
def test_each_valid_action_is_accepted(tmp_path, action):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text(f"- some_type: {action}\n")

    policy = load_redaction_policy(path=config_file)

    assert policy["some_type"] == action


# ---------------------------------------------------------------------------
# Invalid config
# ---------------------------------------------------------------------------


def test_invalid_action_value_is_rejected(tmp_path):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text("- email: redact_completely\n")  # not mask/hash/drop

    with pytest.raises(ValueError):
        load_redaction_policy(path=config_file)


def test_invalid_action_is_case_sensitive(tmp_path):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text("- email: MASK\n")

    with pytest.raises(ValueError):
        load_redaction_policy(path=config_file)


# ---------------------------------------------------------------------------
# Edge cases in the file format
# ---------------------------------------------------------------------------


def test_empty_config_file_produces_empty_mapping(tmp_path):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text("# Redaction Policy\n\n(none configured)\n")

    assert load_redaction_policy(path=config_file) == {}


def test_blank_and_non_bullet_lines_are_ignored(tmp_path):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text(
        "# Redaction Policy\n\nSome prose explaining the format.\n\n- email: mask\n\n"
    )

    assert load_redaction_policy(path=config_file) == {"email": "mask"}


def test_entry_without_colon_is_ignored(tmp_path):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text("- just some bullet text, not a policy entry\n- ssn: drop\n")

    assert load_redaction_policy(path=config_file) == {"ssn": "drop"}


def test_whitespace_around_data_type_and_action_is_trimmed(tmp_path):
    config_file = tmp_path / "redaction_policy.md"
    config_file.write_text("-   email   :   mask   \n")

    assert load_redaction_policy(path=config_file) == {"email": "mask"}


# ---------------------------------------------------------------------------
# get_action(): default handling for unlisted data types
# ---------------------------------------------------------------------------


def test_get_action_returns_configured_action():
    policy = {"email": "mask", "ssn": "drop"}
    assert get_action(policy, "ssn") == "drop"


def test_get_action_returns_default_for_unlisted_type():
    policy = {"email": "mask"}
    assert get_action(policy, "unknown_type") == "mask"  # documented default


def test_get_action_accepts_custom_default():
    policy = {}
    assert get_action(policy, "unknown_type", default="drop") == "drop"


# ---------------------------------------------------------------------------
# Env-var override (the new capability this task adds over the earlier
# halt_reasons.md/event_types.md precedents): change policy without a
# code change or redeploy.
# ---------------------------------------------------------------------------


def test_env_var_override_is_used_when_no_explicit_path_given(tmp_path, monkeypatch):
    config_file = tmp_path / "external_policy.md"
    config_file.write_text("- credit_card: drop\n")
    monkeypatch.setenv("SANITIZER_REDACTION_POLICY_PATH", str(config_file))

    policy = load_redaction_policy()

    assert policy == {"credit_card": "drop"}


def test_explicit_path_argument_takes_precedence_over_env_var(tmp_path, monkeypatch):
    env_file = tmp_path / "env_policy.md"
    env_file.write_text("- email: hash\n")
    monkeypatch.setenv("SANITIZER_REDACTION_POLICY_PATH", str(env_file))

    explicit_file = tmp_path / "explicit_policy.md"
    explicit_file.write_text("- email: drop\n")

    policy = load_redaction_policy(path=explicit_file)

    assert policy == {"email": "drop"}


def test_bundled_default_used_when_no_path_and_no_env_var(monkeypatch):
    monkeypatch.delenv("SANITIZER_REDACTION_POLICY_PATH", raising=False)

    policy = load_redaction_policy()

    assert policy  # falls back to the shipped default, not an error
