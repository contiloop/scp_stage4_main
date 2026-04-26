"""Thin storage interface. Local-only for now; HF/S3/R2 sync later.

All paths are relative to a `root`. Atomic writes use tmp-then-rename.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


class LocalStorage:
    def __init__(self, root: str | Path, cloud_uri: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cloud_uri = cloud_uri

    def path(self, rel: str | Path) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, rel: str | Path) -> bool:
        return (self.root / rel).exists()

    def append_jsonl(self, rel: str | Path, row: dict) -> None:
        p = self.path(rel)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_jsonl(self, rel: str | Path) -> Iterable[dict]:
        p = self.root / rel
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def write_atomic(self, rel: str | Path, data: bytes) -> Path:
        p = self.path(rel)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return p

    def write_parquet(self, rel: str | Path, df) -> Path:
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return self.write_atomic(rel, buf.getvalue())

    def read_parquet(self, rel: str | Path):
        import pandas as pd
        return pd.read_parquet(self.root / rel)

    def sync(self, commit_message: str | None = None) -> None:
        """Push run dir to remote. Currently supports hf://<repo_id>[/<subdir>]."""
        if not self.cloud_uri:
            return
        if self.cloud_uri.startswith("hf://"):
            self._sync_hf(commit_message=commit_message)
        else:
            raise ValueError(f"Unsupported cloud_uri scheme: {self.cloud_uri}")

    def _sync_hf(self, commit_message: str | None = None) -> None:
        from huggingface_hub import HfApi, create_repo
        rest = self.cloud_uri[len("hf://"):].strip("/")
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"hf:// uri must include user/repo: {self.cloud_uri}")
        repo_id = "/".join(parts[:2])
        path_in_repo = parts[2] if len(parts) == 3 else self.root.name
        api = HfApi()
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
        except Exception:
            pass
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(self.root),
            path_in_repo=path_in_repo,
            commit_message=commit_message or f"sync {self.root.name}",
            ignore_patterns=["*.tmp", "**/.DS_Store"],
        )
