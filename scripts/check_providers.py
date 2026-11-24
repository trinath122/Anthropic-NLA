"""Day-1 check: which free providers / models in configs/llm.yaml actually answer.

For each provider with a key in .env: list its models, confirm every configured
model exists, and send ONE tiny ping per configured model (bypasses the cache on
purpose; costs one free request each). Never prints keys.

    uv run python scripts/check_providers.py --config configs/llm.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

from nla_audit.runtime.env import load_secrets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--no-ping", action="store_true", help="only list models")
    args = ap.parse_args()

    import openai

    load_secrets()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ok = True
    for pname, p in cfg["providers"].items():
        key = os.environ.get(p["api_key_env"], "")
        wanted = [k.split("/", 1)[1] for k in cfg["models"] if k.split("/", 1)[0] == pname]
        if not key:
            print(f"[{pname}] SKIP: {p['api_key_env']} not set ({len(wanted)} models configured)")
            continue
        client = openai.OpenAI(base_url=p["base_url"], api_key=key, max_retries=0, timeout=60)
        try:
            available = {m.id for m in client.models.list()}
        except Exception as e:
            print(f"[{pname}] ERROR listing models: {type(e).__name__}: {e}")
            ok = False
            continue
        print(f"[{pname}] {len(available)} models visible")
        for m in wanted:
            present = m in available or f"models/{m}" in available
            status = "present" if present else "MISSING"
            if present and not args.no_ping:
                try:
                    r = client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": "Reply with the single word OK."}],
                        max_tokens=5,
                        temperature=0,
                    )
                    status += f", ping -> {(r.choices[0].message.content or '').strip()[:20]!r}"
                except Exception as e:
                    status += f", ping FAILED: {type(e).__name__}"
                    ok = False
            elif not present:
                ok = False
            print(f"    {m}: {status}")
        if not args.no_ping:
            print("    (check this provider's console for current free-tier daily limits)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
