def grids_equal(a, b) -> bool:
    """Strict structural equality for ARC grids (list[list[int]])."""
    return a == b


def score_task(kaggle_preds: list[dict], gt_outputs: list) -> float:
    """
    Fraction of test inputs correct for a task.
    Correct if attempt_1 == GT or attempt_2 == GT for each test input.
    """
    if not gt_outputs:
        return 0.0
    correct = 0
    for i, gt in enumerate(gt_outputs):
        if i >= len(kaggle_preds):
            continue
        pack = kaggle_preds[i] or {}
        a1 = pack.get("attempt_1")
        a2 = pack.get("attempt_2")
        if (a1 is not None and grids_equal(a1, gt)) or (
            a2 is not None and grids_equal(a2, gt)
        ):
            correct += 1
    return correct / max(len(gt_outputs), 1)
