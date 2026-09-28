"""Tests for the markdown-backed halt-reason config loader (Phase 0 follow-up).

`schemas.halt_signal.load_allowed_reasons` reads config/halt_reasons.md so
that adding/removing a valid `reason` value doesn't require touching Python
code. Written before the implementation, per this project's TDD workflow.
"""
import schemas.halt_signal as halt_signal_module
from schemas.halt_signal import load_allowed_reasons


def test_default_config_lists_expected_reasons():
    """Sanity check that the shipped config defines the expected reasons.
    manual_override added in Phase 5 (control-api's manual override
    trigger task) -- a human's manual halt shouldn't have to masquerade
    as one of the automated-detection reasons in the audit log."""
    assert load_allowed_reasons() == [
        "repeated_tool_calls",
        "repeated_generation",
        "runaway_loop",
        "manual_override",
    ]


def test_parses_custom_markdown_bullets(tmp_path):
    config_file = tmp_path / "halt_reasons.md"
    config_file.write_text("- alpha\n- beta\n- gamma\n")

    assert load_allowed_reasons(config_file) == ["alpha", "beta", "gamma"]


def test_ignores_non_bullet_lines(tmp_path):
    config_file = tmp_path / "halt_reasons.md"
    config_file.write_text(
        "# Halt Reasons\n"
        "\n"
        "Add or remove entries below to change what `reason` values are accepted.\n"
        "\n"
        "- alpha\n"
        "\n"
        "- beta\n"
    )

    assert load_allowed_reasons(config_file) == ["alpha", "beta"]


def test_bullet_with_extra_whitespace_is_trimmed(tmp_path):
    config_file = tmp_path / "halt_reasons.md"
    config_file.write_text("-    alpha   \n-beta\n")

    assert load_allowed_reasons(config_file) == ["alpha", "beta"]


def test_missing_config_file_raises(tmp_path):
    missing_file = tmp_path / "does_not_exist.md"
    try:
        load_allowed_reasons(missing_file)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for a missing config file")


def test_empty_config_file_returns_empty_list(tmp_path):
    config_file = tmp_path / "halt_reasons.md"
    config_file.write_text("")

    assert load_allowed_reasons(config_file) == []


def test_config_with_only_prose_returns_empty_list(tmp_path):
    config_file = tmp_path / "halt_reasons.md"
    config_file.write_text("# Halt Reasons\n\nNone defined yet.\n")

    assert load_allowed_reasons(config_file) == []


def test_duplicate_bullet_entries_are_preserved_as_listed(tmp_path):
    """The loader parses bullets as-is; it doesn't dedupe. Membership checks
    against the result are unaffected by duplicates either way."""
    config_file = tmp_path / "halt_reasons.md"
    config_file.write_text("- alpha\n- alpha\n")

    assert load_allowed_reasons(config_file) == ["alpha", "alpha"]


def test_no_arg_call_reads_current_module_level_default_path(tmp_path, monkeypatch):
    """Confirms the default path is looked up at call time (so tests/callers
    can monkeypatch it), not captured once at import time."""
    config_file = tmp_path / "halt_reasons.md"
    config_file.write_text("- patched_reason\n")
    monkeypatch.setattr(halt_signal_module, "_HALT_REASONS_FILE", config_file)

    assert load_allowed_reasons() == ["patched_reason"]
