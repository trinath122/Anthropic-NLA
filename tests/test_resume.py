from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nla_audit.runtime.resume import Checkpointer


class FakeHub:
    """In-memory stand-in for the private HF repo."""

    def __init__(self, store: Path) -> None:
        self.store = store
        self.pushes = 0

    def push(self, run_name: str, local_dir: Path) -> None:
        dst = self.store / run_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(local_dir, dst)
        self.pushes += 1

    def pull(self, run_name: str, root_dir: Path) -> None:
        src = self.store / run_name
        if src.exists():
            shutil.copytree(src, root_dir / run_name, dirs_exist_ok=True)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_interval_must_be_at_most_30_min(tmp_path):
    with pytest.raises(ValueError):
        Checkpointer("r", tmp_path, interval_s=31 * 60)


def test_syncs_on_interval_and_resumes_on_new_machine(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    clock = Clock()
    ck = Checkpointer("step1/seed0", tmp_path / "machine1", syncer=hub, interval_s=600, clock=clock)
    ck.shard_path("s0", ".txt").write_text("result 0")
    ck.mark_done("s0")
    assert hub.pushes == 0  # interval not reached
    clock.t = 601
    ck.shard_path("s1", ".txt").write_text("result 1")
    ck.mark_done("s1")
    assert hub.pushes == 1

    # Session dies; a fresh machine restores from the hub and skips finished shards.
    ck2 = Checkpointer("step1/seed0", tmp_path / "machine2", syncer=hub, clock=clock)
    assert ck2.restore() == 2
    assert ck2.is_done("s0") and ck2.is_done("s1") and not ck2.is_done("s2")
    assert ck2.shard_path("s1", ".txt").read_text() == "result 1"


def test_finalize_forces_sync(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    ck = Checkpointer("r", tmp_path / "m", syncer=hub, clock=Clock())
    ck.mark_done("a")
    ck.finalize()
    assert hub.pushes == 1


def test_local_resume_without_hub(tmp_path):
    ck = Checkpointer("r", tmp_path)
    ck.mark_done("x")
    assert Checkpointer("r", tmp_path).is_done("x")
