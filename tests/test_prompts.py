from synapse_bench.prompts import prompt_fingerprint, prompt_for, stable_prefix


def test_stable_prefix_is_deterministic() -> None:
    assert stable_prefix(128, 42) == stable_prefix(128, 42)
    assert stable_prefix(128, 42) != stable_prefix(128, 43)


def test_suffix_changes_without_changing_prefix() -> None:
    prefix = stable_prefix(64, 42)
    first = prompt_for(prefix, 1)
    second = prompt_for(prefix, 2)

    assert first.startswith(prefix)
    assert second.startswith(prefix)
    assert prompt_fingerprint(first) != prompt_fingerprint(second)
