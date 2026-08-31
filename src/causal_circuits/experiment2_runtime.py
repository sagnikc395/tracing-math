"""Durable status, logging, and checkpoint helpers for Experiment 2."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import TypeVar

import pandas as pd

T = TypeVar("T")
LOGGER = logging.getLogger("causal_circuits.experiment2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(_jsonable(payload), indent=2, allow_nan=False))


def atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint_identity(path: Path, identity: dict[str, object]) -> None:
    """Refuse to combine a partial checkpoint with a scientifically different run."""
    normalized = _jsonable(identity)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(
                f"Checkpoint identity mismatch at {path}; use a new output directory "
                "or remove that stage's checkpoints explicitly"
            )
        return
    atomic_write_json(path, normalized)


def configure_logging(output_dir: Path, command: str) -> Path:
    """Append human-readable messages to both aggregate and per-command logs."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = __import__("time").gmtime
    for path in (log_dir / "experiment2.log", log_dir / f"{command}.log"):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return log_dir / f"{command}.log"


def stage_status(output_dir: Path) -> dict[str, object]:
    path = output_dir / "stage_status.json"
    if not path.exists():
        return {"stages": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"stages": {}}
    payload.setdefault("stages", {})
    return payload


def status_report(output_dir: Path) -> dict[str, object]:
    """Collect the small live status files without loading large result tables."""
    report = {"output_dir": str(output_dir), "stage_status": stage_status(output_dir)}
    progress_paths = {
        "semantic_extraction": output_dir / "semantic_extraction_progress.json",
        "verdict_audit": output_dir / "verdict_audit" / "progress.json",
        "causal_validation": output_dir / "causal_validation" / "progress.json",
    }
    progress = {}
    for name, path in progress_paths.items():
        if path.exists():
            progress[name] = json.loads(path.read_text(encoding="utf-8"))
    report["progress"] = progress
    report["logs"] = {
        "aggregate": str(output_dir / "logs" / "experiment2.log"),
        "directory": str(output_dir / "logs"),
    }
    return report


def update_stage_progress(output_dir: Path, stage: str, **progress: object) -> None:
    payload = stage_status(output_dir)
    stages = payload["stages"]
    entry = stages.setdefault(stage, {})
    now = utc_now()
    entry.update({"updated_at": now, **_jsonable(progress)})
    payload["current_stage"] = stage if entry.get("status") == "running" else None
    payload["updated_at"] = now
    atomic_write_json(output_dir / "stage_status.json", payload)


def run_stage(
    output_dir: Path,
    stage: str,
    operation: Callable[[], T],
    *,
    skip_completed: bool = False,
    identity: str | None = None,
) -> T | dict[str, object]:
    """Run one stage with durable start/failure/completion metadata."""
    previous = stage_status(output_dir).get("stages", {}).get(stage, {})
    if (
        skip_completed
        and previous.get("status") == "complete"
        and previous.get("run_identity") == identity
    ):
        LOGGER.info("Skipping completed stage %s", stage)
        return previous.get("result", {"status": "complete", "resumed": True})

    attempt = int(previous.get("attempt", 0)) + 1
    started_at = utc_now()
    started = monotonic()
    payload = stage_status(output_dir)
    entry = {
        "status": "running",
        "attempt": attempt,
        "started_at": started_at,
        "updated_at": started_at,
        "pid": os.getpid(),
        "run_identity": identity,
    }
    if "log_path" in previous:
        entry["log_path"] = previous["log_path"]
    payload["stages"][stage] = entry
    payload.update({"current_stage": stage, "updated_at": started_at})
    atomic_write_json(output_dir / "stage_status.json", payload)
    LOGGER.info("Starting stage %s (attempt %d)", stage, attempt)
    try:
        result = operation()
    except BaseException as error:
        elapsed = monotonic() - started
        update_stage_progress(
            output_dir,
            stage,
            status="failed",
            finished_at=utc_now(),
            elapsed_seconds=round(elapsed, 3),
            error={"type": type(error).__name__, "message": str(error)},
        )
        LOGGER.exception("Stage %s failed after %.1f seconds", stage, elapsed)
        raise
    elapsed = monotonic() - started
    update_stage_progress(
        output_dir,
        stage,
        status="complete",
        finished_at=utc_now(),
        elapsed_seconds=round(elapsed, 3),
        result=result,
    )
    LOGGER.info("Completed stage %s in %.1f seconds", stage, elapsed)
    return result


def write_progress(
    path: Path,
    *,
    completed: int,
    total: int,
    rows: int,
    status: str = "running",
    **extra: object,
) -> None:
    payload = {
        "status": status,
        "units_completed": completed,
        "units_total": total,
        "fraction_complete": completed / total if total else 1.0,
        "rows_checkpointed": rows,
        "updated_at": utc_now(),
        **extra,
    }
    atomic_write_json(path, payload)
    LOGGER.info(
        "%s: %d/%d units complete; %d rows checkpointed",
        path.parent.name,
        completed,
        total,
        rows,
    )


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return None
    return value
