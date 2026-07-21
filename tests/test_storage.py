from pathlib import Path

from synapse_bench.models import RunArtifact
from synapse_bench.storage import write_artifact_atomic


def test_artifact_write_is_valid_and_contains_no_prompt_field(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    artifact = RunArtifact(run_id="test", environment={}, configuration={})

    write_artifact_atomic(path, artifact)
    content = path.read_text(encoding="utf-8")

    assert '"schema_version": "1.0"' in content
    assert '"prompt"' not in content
    assert list(tmp_path.glob("*.tmp")) == []
