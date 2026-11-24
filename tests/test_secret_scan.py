"""The scanner must catch real-looking secrets. Fake secrets are assembled at runtime
so this file itself never contains a token-shaped string."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "secret_scan", Path(__file__).parents[1] / "scripts" / "secret_scan.py"
)
assert _spec and _spec.loader
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)


@pytest.mark.parametrize(
    "label,value",
    [
        ("huggingface token", "hf" + "_" + "A" * 34),
        ("groq key", "gsk" + "_" + "b" * 52),
        ("cerebras key", "csk" + "-" + "c" * 48),
        ("google api key", "AI" + "za" + "D" * 35),
        ("openrouter key", "sk-or" + "-v1-" + "e" * 64),
        ("github token", "gh" + "o_" + "F" * 36),
    ],
)
def test_patterns_detect(label, value):
    assert scan.PATTERNS[label].search(f"key = {value}\n")


def test_env_example_allowed_env_blocked():
    assert scan.FORBIDDEN_NAMES.search(".env")
    assert scan.FORBIDDEN_NAMES.search("sub/kaggle.json")
    assert ".env.example" in scan.ALLOWED_NAMES


def test_attribution_trailer_detected():
    msg = "Fix bug\n\nCo-" + "Authored-By: " + "Claude <x@y>"
    assert scan.ATTRIBUTION.search(msg)
    assert not scan.ATTRIBUTION.search("Compare NLA output with Claude Haiku results")
