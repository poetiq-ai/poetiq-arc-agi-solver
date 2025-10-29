from arc_agi.types import RunResult


def canonical_test_key(results: list[RunResult]) -> str:
    return str([r["output"] for r in results])
