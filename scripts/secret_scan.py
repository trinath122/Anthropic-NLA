#!/usr/bin/env python3
"""Pre-push guard (stdlib only). Fails if anything git would push contains:

* API keys / tokens (HF, Groq, Cerebras, Google, OpenRouter, OpenAI-style, GitHub, Kaggle, W&B);
* forbidden files (.env, kaggle.json, private keys);
* AI attribution trailers in unpushed commit messages (project rule).

Usage: python3 scripts/secret_scan.py   (exit code 1 on any finding)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "huggingface token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "groq key": re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"),
    "cerebras key": re.compile(r"\bcsk-[A-Za-z0-9]{40,}\b"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "openrouter key": re.compile(r"\bsk-or-v1-[0-9a-f]{40,}\b"),
    "openai/anthropic-style key": re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{32,}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "kaggle key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "wandb key": re.compile(r"WANDB_API_KEY\s*=\s*[0-9a-f]{40}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..+)?|kaggle\.json|.*\.pem|id_rsa.*)$")
ALLOWED_NAMES = {".env.example"}
ATTRIBUTION = re.compile(
    r"co-authored-by:\s*claude|generated with \[?claude code|noreply@anthropic\.com",
    re.IGNORECASE,
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def files_to_check() -> list[str]:
    tracked = git("ls-files").splitlines()
    staged = git("diff", "--cached", "--name-only", "--diff-filter=ACM").splitlines()
    return sorted(set(tracked) | set(staged))


def unpushed_commit_messages() -> str:
    try:
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        rng = f"{upstream}..HEAD"
    except subprocess.CalledProcessError:
        rng = "HEAD"
    try:
        return git("log", "--format=%H%n%B", rng)
    except subprocess.CalledProcessError:
        return ""


def main() -> int:
    findings: list[str] = []
    for name in files_to_check():
        base = Path(name).name
        if base not in ALLOWED_NAMES and FORBIDDEN_NAMES.search(name):
            findings.append(f"forbidden file tracked/staged: {name}")
            continue
        p = Path(name)
        if not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pat in PATTERNS.items():
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                findings.append(f"{label} in {name}:{line}")
    msgs = unpushed_commit_messages()
    if ATTRIBUTION.search(msgs):
        findings.append("AI attribution trailer found in an unpushed commit message")
    if findings:
        print("secret_scan: BLOCKED")
        for f in findings:
            print("  -", f)
        return 1
    print(f"secret_scan: OK ({len(files_to_check())} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
