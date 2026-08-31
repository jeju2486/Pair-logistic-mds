from __future__ import annotations

"""Atomic, versioned checkpoint storage for long KOVAR scans."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 2
_STAGES = ("score",)


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _normalise_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")


def fingerprint_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Return a content fingerprint suitable for strict resume validation."""

    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"Cannot fingerprint missing input file: {source}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return {
        "name": source.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def run_fingerprint(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class CheckpointStatistics:
    resumed: bool = False
    recovered_score_tasks: int = 0
    written_score_tasks: int = 0
    shards_read: int = 0
    shards_written: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    read_seconds: float = 0.0
    write_seconds: float = 0.0

    def fields(self) -> dict[str, int | float]:
        return {
            "resumed": int(self.resumed),
            "recovered_score_tasks": self.recovered_score_tasks,
            "written_score_tasks": self.written_score_tasks,
            "shards_read": self.shards_read,
            "shards_written": self.shards_written,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "read_seconds": self.read_seconds,
            "write_seconds": self.write_seconds,
        }


@dataclass
class CheckpointStore:
    root: Path
    identity: Mapping[str, Any]
    tool_version: str
    every: int = 100
    resume: bool = False
    statistics: CheckpointStatistics = field(default_factory=CheckpointStatistics)
    _manifest: dict[str, Any] = field(init=False, repr=False)
    _buffers: dict[str, list[dict[str, Any]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if int(self.every) < 1:
            raise CheckpointError("Checkpoint interval must be positive")
        self.every = int(self.every)
        self._buffers = {stage: [] for stage in _STAGES}
        self.statistics.resumed = bool(self.resume)
        manifest_path = self.root / "manifest.json"
        expected_fingerprint = run_fingerprint(self.identity)

        if self.resume:
            if not manifest_path.is_file():
                raise CheckpointError(
                    f"Resume requested but checkpoint manifest is missing: {manifest_path}"
                )
            try:
                self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointError(f"Cannot read checkpoint manifest: {exc}") from exc
            if not isinstance(self._manifest, dict):
                raise CheckpointError("Checkpoint manifest has an invalid structure")
            if self._manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointError("Checkpoint schema version is incompatible")
            if self._manifest.get("tool_version") != self.tool_version:
                raise CheckpointError(
                    "Checkpoint was created by a different KOVAR version"
                )
            if self._manifest.get("run_fingerprint") != expected_fingerprint:
                raise CheckpointError(
                    "Checkpoint inputs or analysis settings do not match this run"
                )
            if self._manifest.get("status") == "complete":
                raise CheckpointError(
                    "Checkpoint is already marked complete; use the final output files"
                )
        else:
            if manifest_path.exists() or (self.root.exists() and any(self.root.iterdir())):
                raise CheckpointError(
                    f"Checkpoint directory already contains data: {self.root}. "
                    "Use --resume for the matching run or choose another directory."
                )
            self.root.mkdir(parents=True, exist_ok=True)
            now = _utc_now()
            self._manifest = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "tool_version": self.tool_version,
                "run_fingerprint": expected_fingerprint,
                "identity": _normalise_json(self.identity),
                "status": "running",
                "created_utc": now,
                "updated_utc": now,
                "plans": {},
                "shards": [],
            }
            self._write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def fingerprint(self) -> str:
        return str(self._manifest["run_fingerprint"])

    def _write_manifest(self) -> None:
        self._manifest["updated_utc"] = _utc_now()
        _atomic_write_bytes(
            self.manifest_path,
            json.dumps(
                self._manifest,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
        )

    def set_plan(self, stage: str, task_count: int) -> None:
        self._validate_stage(stage)
        count = int(task_count)
        plans = self._manifest.setdefault("plans", {})
        existing = plans.get(stage)
        if existing is not None and int(existing) != count:
            raise CheckpointError(
                f"Checkpoint {stage} task count changed ({existing} != {count})"
            )
        if existing is None:
            plans[stage] = count
            self._write_manifest()

    def iter_stage(self, stage: str) -> Iterator[tuple[int, dict[str, Any]]]:
        self._validate_stage(stage)
        seen: set[int] = set()
        shards = self._manifest.get("shards", [])
        if not isinstance(shards, list):
            raise CheckpointError("Checkpoint shard manifest is invalid")
        for shard in shards:
            if not isinstance(shard, dict):
                raise CheckpointError("Checkpoint shard manifest entry is invalid")
            if shard.get("stage") != stage:
                continue
            started = time.perf_counter()
            filename = shard.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise CheckpointError("Checkpoint shard filename is invalid")
            path = self.root / filename
            try:
                compressed = path.read_bytes()
            except OSError as exc:
                raise CheckpointError(f"Cannot read checkpoint shard {path}: {exc}") from exc
            observed_hash = hashlib.sha256(compressed).hexdigest()
            if observed_hash != shard.get("sha256"):
                raise CheckpointError(f"Checkpoint shard checksum failed: {path}")
            try:
                document = json.loads(gzip.decompress(compressed).decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CheckpointError(f"Checkpoint shard is corrupt: {path}") from exc
            if not isinstance(document, dict):
                raise CheckpointError(f"Checkpoint shard structure is invalid: {path}")
            if document.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointError(f"Checkpoint shard schema is incompatible: {path}")
            if document.get("run_fingerprint") != self.fingerprint:
                raise CheckpointError(f"Checkpoint shard belongs to another run: {path}")
            if document.get("stage") != stage:
                raise CheckpointError(f"Checkpoint shard stage is inconsistent: {path}")
            records = document.get("records")
            if not isinstance(records, list):
                raise CheckpointError(f"Checkpoint shard records are invalid: {path}")
            self.statistics.shards_read += 1
            self.statistics.bytes_read += len(compressed)
            self.statistics.read_seconds += time.perf_counter() - started
            for record in records:
                if not isinstance(record, dict) or "task_id" not in record:
                    raise CheckpointError(
                        f"Checkpoint record in {path.name} is invalid"
                    )
                try:
                    task_id = int(record["task_id"])
                except (TypeError, ValueError) as exc:
                    raise CheckpointError(
                        f"Checkpoint task identifier in {path.name} is invalid"
                    ) from exc
                if task_id in seen:
                    raise CheckpointError(
                        f"Checkpoint contains duplicate {stage} task {task_id}"
                    )
                seen.add(task_id)
                self.statistics.recovered_score_tasks += 1
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    raise CheckpointError(
                        f"Checkpoint payload for {stage} task {task_id} is invalid"
                    )
                yield task_id, payload

    def record(self, stage: str, task_id: int, payload: Mapping[str, Any]) -> None:
        self._validate_stage(stage)
        self._buffers[stage].append(
            {"task_id": int(task_id), "payload": _normalise_json(payload)}
        )
        if len(self._buffers[stage]) >= self.every:
            self.flush(stage)

    def flush(self, stage: str) -> None:
        self._validate_stage(stage)
        records = self._buffers[stage]
        if not records:
            return
        sequence = sum(
            1 for shard in self._manifest.get("shards", []) if shard.get("stage") == stage
        )
        filename = f"{stage}-{sequence:06d}.json.gz"
        document = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_fingerprint": self.fingerprint,
            "stage": stage,
            "records": records,
        }
        started = time.perf_counter()
        # Preserve payload key order so resumed diagnostic tables retain the
        # exact same column order as uninterrupted tables. The shard checksum
        # protects bytes; canonical ordering is only needed for run identity.
        shard_json = json.dumps(
            _normalise_json(document),
            separators=(",", ":"),
            allow_nan=True,
        ).encode("utf-8")
        compressed = gzip.compress(shard_json, compresslevel=1, mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        _atomic_write_bytes(self.root / filename, compressed)
        self._manifest.setdefault("shards", []).append(
            {
                "stage": stage,
                "file": filename,
                "n_tasks": len(records),
                "sha256": digest,
                "size_bytes": len(compressed),
            }
        )
        self._write_manifest()
        self.statistics.written_score_tasks += len(records)
        self.statistics.shards_written += 1
        self.statistics.bytes_written += len(compressed)
        self.statistics.write_seconds += time.perf_counter() - started
        self._buffers[stage] = []

    def flush_all(self) -> None:
        for stage in _STAGES:
            self.flush(stage)

    def complete(self, *, cleanup: bool) -> bool:
        self.flush_all()
        self._manifest["status"] = "complete"
        self._manifest["completed_utc"] = _utc_now()
        self._write_manifest()
        if cleanup:
            resolved = self.root.resolve()
            filesystem_root = Path(resolved.anchor).resolve()
            home = Path.home().resolve()
            if (
                resolved == filesystem_root
                or resolved == home
                or not self.manifest_path.is_file()
            ):
                return False
            try:
                shutil.rmtree(self.root)
            except OSError:
                return False
        return True

    def metadata_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "enabled": 1,
            "directory": str(self.root),
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_fingerprint": self.fingerprint,
            "interval_tasks": self.every,
        }
        fields.update(self.statistics.fields())
        return fields

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in _STAGES:
            raise CheckpointError(f"Unknown checkpoint stage: {stage}")


def fingerprint_inputs(inputs: Iterable[tuple[str, str | Path]]) -> dict[str, Any]:
    return {label: fingerprint_file(path) for label, path in inputs}
