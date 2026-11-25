"""Detect where we run (kaggle / colab / local), pick a dtype, load secrets, record provenance."""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Mapping
from typing import Any

SECRET_NAMES = (
    "HF_TOKEN",
    "WANDB_API_KEY",
    "CEREBRAS_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


def runtime_name(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    if "KAGGLE_KERNEL_RUN_TYPE" in env or "KAGGLE_URL_BASE" in env:
        return "kaggle"
    if "COLAB_RELEASE_TAG" in env or "COLAB_GPU" in env:
        return "colab"
    return "local"


def load_secrets(names: tuple[str, ...] = SECRET_NAMES) -> list[str]:
    """Populate os.environ from .env (local), Kaggle Secrets or Colab Secrets.

    Returns the names that were found. Values are never printed or logged.
    """
    found: list[str] = []
    rt = runtime_name()
    if rt == "local":
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass
    for name in names:
        if os.environ.get(name):
            found.append(name)
            continue
        value: str | None = None
        try:
            if rt == "kaggle":
                from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

                value = UserSecretsClient().get_secret(name)
            elif rt == "colab":
                from google.colab import userdata  # type: ignore[import-not-found]

                value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value
            found.append(name)
    return found


def gpu_info() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append(
            {
                "index": i,
                "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "total_mem_gib": round(p.total_memory / 2**30, 2),
            }
        )
    return out


def gpu_max_memory(headroom_gib: float = 3.0) -> dict[Any, str] | None:
    """`max_memory` for device_map="auto" that leaves room for activations on every GPU.

    Without it, accelerate packs GPU 0 to the brim with weights and the first forward OOMs
    (seen with fp32 Qwen-7B blocks 0-20 on 2×T4). No "cpu" key: never silently offload.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    out: dict[Any, str] = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 2**30
        out[i] = f"{max(total - headroom_gib, 1.0):.1f}GiB"
    return out


def pick_dtype(prefer: str = "auto") -> Any:
    """bf16 on Ampere+ (L4/A100/RTX 40xx), fp16 on T4 (sm75), fp32 on CPU.

    Never a quantized dtype (plan rule).
    """
    import torch

    table = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if prefer != "auto":
        return table[prefer]
    if not torch.cuda.is_available():
        return torch.float32
    caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    return torch.bfloat16 if all(c >= (8, 0) for c in caps) else torch.float16


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def provenance(dtype: Any | None = None) -> dict[str, Any]:
    """Recorded alongside every artifact (dtype + GPU are mandatory per the plan)."""
    import importlib.metadata as md

    def ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    return {
        "runtime": runtime_name(),
        "gpus": gpu_info(),
        "dtype": None if dtype is None else str(dtype).replace("torch.", ""),
        "python": platform.python_version(),
        "torch": ver("torch"),
        "transformers": ver("transformers"),
        "git_commit": git_commit(),
    }
