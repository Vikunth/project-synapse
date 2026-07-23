"""Crash-resilient artifact persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from synapse_bench.models import RunArtifact


def write_artifact_atomic(path: Path, artifact: RunArtifact) -> None:
    """Replace an artifact atomically without ever serializing prompts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_model_atomic(path: Path, artifact: BaseModel) -> None:
    """Atomically checkpoint any versioned Pydantic artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)
