import numpy as np

from arc_agi.types import ARCAGIResult, RunResult


def canonical_test_key(results: list[RunResult]) -> str:
    return str([r.get("output", "") for r in results])

def _mean_soft(res: ARCAGIResult) -> float:
    trs = res.get("train_results", [])
    if not trs:
        return 0.0
    return float(np.mean([rr.get("soft_score", 0.0) for rr in trs]))
