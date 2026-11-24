"""Resumable jobs: shard-level progress + sync to a PRIVATE Hugging Face repo.

Plan rule: every Kaggle/Colab job checkpoints to a private HF repo at least every
30 minutes and auto-resumes. Usage:

    ck = Checkpointer("step1_baseline/run-seed0", local_dir, repo_id=os.environ["HF_ARTIFACT_REPO"])
    ck.restore()                         # pulls previous progress, if any
    for shard_id in shard_ids:
        if ck.is_done(shard_id):
            continue
        path = ck.shard_path(shard_id, ".parquet")
        ... write results to `path` ...
        ck.mark_done(shard_id)           # syncs if >= interval since last sync
    ck.finalize()                        # final forced sync
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

MAX_SYNC_INTERVAL_S = 30 * 60


class Syncer(Protocol):
    def pull(self, run_name: str, root_dir: Path) -> None: ...
    def push(self, run_name: str, local_dir: Path) -> None: ...


class HFHubSyncer:
    """Private HF *dataset* repo; each run lives under a folder named after the run."""

    def __init__(self, repo_id: str, token: str | None = None) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = self.api.repo_info(repo_id, repo_type="dataset")
        if not getattr(info, "private", False):
            raise RuntimeError(
                f"{repo_id} is PUBLIC; artifact repos must stay private until release is approved"
            )

    def pull(self, run_name: str, root_dir: Path) -> None:
        """Download `<run_name>/**` into `root_dir/<run_name>/` (no-op if nothing there yet)."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            allow_patterns=[f"{run_name}/**"],
            local_dir=str(root_dir),
            token=self.api.token,
        )

    def push(self, run_name: str, local_dir: Path) -> None:
        self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=run_name,
            commit_message=f"checkpoint {run_name}",
        )


class Checkpointer:
    def __init__(
        self,
        run_name: str,
        root_dir: str | Path,
        *,
        syncer: Syncer | None = None,
        interval_s: float = 25 * 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if interval_s > MAX_SYNC_INTERVAL_S:
            raise ValueError("sync interval must be <= 30 minutes (plan rule)")
        self.run_name = run_name.strip("/")
        self.root_dir = Path(root_dir)
        self.local_dir = self.root_dir / self.run_name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.syncer = syncer
        self.interval_s = interval_s
        self._clock = clock
        self._last_sync = clock()
        self._state_path = self.local_dir / "progress.json"
        self._done: set[str] = set()
        self._load_state()

    # -- state -------------------------------------------------------------
    def _load_state(self) -> None:
        if self._state_path.exists():
            self._done = set(json.loads(self._state_path.read_text())["done"])

    def _save_state(self) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done": sorted(self._done)}, indent=1))
        tmp.replace(self._state_path)

    # -- API -----------------------------------------------------------------
    def restore(self) -> int:
        """Pull previous progress from the hub (if a syncer is set). Returns #shards done."""
        if self.syncer is not None:
            self.syncer.pull(self.run_name, self.root_dir)
            self._load_state()
        return len(self._done)

    def is_done(self, shard_id: str) -> bool:
        return shard_id in self._done

    def shard_path(self, shard_id: str, suffix: str) -> Path:
        d = self.local_dir / "shards"
        d.mkdir(exist_ok=True)
        return d / f"{shard_id}{suffix}"

    def mark_done(self, shard_id: str) -> None:
        self._done.add(shard_id)
        self._save_state()
        self.maybe_sync()

    def maybe_sync(self, force: bool = False) -> bool:
        if self.syncer is None:
            return False
        if not force and self._clock() - self._last_sync < self.interval_s:
            return False
        self.syncer.push(self.run_name, self.local_dir)
        self._last_sync = self._clock()
        return True

    def finalize(self) -> None:
        self.maybe_sync(force=True)

    @property
    def done(self) -> frozenset[str]:
        return frozenset(self._done)
