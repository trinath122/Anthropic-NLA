"""Deterministic document sampling for Step 1 (and later steps).

Conventions follow the NLA training data (model card: 50/50 WildChat + Ultra-FineWeb):
UFW is fed as raw text, WildChat through the Qwen chat template. We read the LAST
shard of each corpus (streamed, so only the rows we touch are downloaded) to reduce,
not eliminate, overlap with the NLA's training texts — the exact training subset is
not published, so Step 1 is a "roughly matches" check, not an exact replication.

WildChat filters: English, not toxic, not redacted (PII). Messages are reduced to
{role, content}. Nothing from these corpora is ever committed to git.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

UFW_REPO = "openbmb/Ultra-FineWeb"
WILDCHAT_REPO = "allenai/WildChat-1M"


@dataclass(frozen=True)
class Doc:
    doc_id: str
    source: str  # "ufw" | "wildchat"
    kind: str  # "text" | "chat"
    text: str | None = None
    messages: list[dict[str, str]] | None = field(default=None, hash=False)


def _stable_id(source: str, payload: str) -> str:
    return f"{source}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def ufw_docs(rows: Iterable[dict[str, Any]], min_chars: int = 400) -> Iterator[Doc]:
    for r in rows:
        text = r.get("content") or ""
        if len(text) >= min_chars:
            yield Doc(_stable_id("ufw", text), "ufw", "text", text=text)


def wildchat_docs(rows: Iterable[dict[str, Any]], language: str = "English") -> Iterator[Doc]:
    for r in rows:
        if r.get("toxic") or r.get("redacted") or r.get("language") != language:
            continue
        conv = r.get("conversation") or []
        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in conv
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if not msgs or msgs[0]["role"] != "user":
            continue
        key = r.get("conversation_hash") or repr(msgs)
        yield Doc(_stable_id("wildchat", str(key)), "wildchat", "chat", messages=msgs)


def shuffled_pool(docs: Iterable[Doc], pool: int, seed: int) -> list[Doc]:
    """The first `pool` qualifying docs (deduplicated), in a seeded random order.

    Callers walk this list and keep the first n docs that yield a valid position.
    Deterministic given the stream order, so every runtime picks the same docs.
    """
    seen: set[str] = set()
    buf: list[Doc] = []
    for d in docs:
        if d.doc_id in seen:
            continue
        seen.add(d.doc_id)
        buf.append(d)
        if len(buf) >= pool:
            break
    random.Random(seed).shuffle(buf)
    return buf


def stream_parquet(repo: str, data_file: str) -> Iterator[dict[str, Any]]:
    """Stream one parquet file of an HF dataset repo (needs HF_TOKEN for gated repos)."""
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{repo}/{data_file}",
        split="train",
        streaming=True,
    )
    yield from ds
