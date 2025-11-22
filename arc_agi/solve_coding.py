import json
import re
import string
from typing import Any, Optional

import numpy as np

from arc_agi.llm import llm
from arc_agi.sandbox import run
from arc_agi.types import ARCAGIResult, ARCAGISolution, ExpertConfig, RunResult


async def solve_coding(
    *,
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    config: ExpertConfig,
    problem_id: str | None = None,
) -> ARCAGIResult:
    solver_prompt = config["solver_prompt"]
    feedback_prompt = config["feedback_prompt"]
    llm_model = config["llm_id"]
    max_iterations = int(config["max_iterations"])
    solver_temperature = float(config["solver_temperature"])
    max_solutions = int(config.get("max_solutions"))
    selection_probability = float(config.get("selection_probability"))
    seed = int(config.get("seed"))
    timeout_sandbox = float(config.get("timeout_s", 5))
    shuffle_examples = bool(config.get("shuffle_examples"))
    improving_order = bool(config.get("improving_order"))
    return_best = bool(config.get("return_best_result"))
    request_timeout = config.get("request_timeout")
    max_total_timeouts = config.get("max_total_timeouts")
    max_total_time = config.get("max_total_time")
    per_iteration_retries = config.get("per_iteration_retries")

    best_train_score = -1.0
    best_result: Optional[ARCAGIResult] = None
    last_train: list[RunResult] = [
        RunResult(
            success=False,
            output="",
            soft_score=0.0,
            error="Unexpected use of initial empty train result",
            code="",
        )
    ]
    last_test: Optional[list[RunResult]] = None

    rng = np.random.default_rng(seed)
    solutions: list[ARCAGISolution] = []

    for it in range(max_iterations):
        example = _make_example(train_in, train_out, test_in)
        problem_str = format_problem(example, shuffle_examples, seed + it)
        message = _build_prompt(solver_prompt, problem=problem_str)

        selected = []
        if solutions:
            mask = rng.uniform(size=len(solutions)) < selection_probability
            selected = [s for s, keep in zip(solutions, mask, strict=False) if keep]

        if selected:
            examples_block = create_examples(
                selected, max_examples=max_solutions, improving_order=improving_order
            )
            message += "\n\n" + _build_prompt(feedback_prompt, feedback=examples_block)

        try:
            response, duration, max_total_time, max_total_timeouts = await llm(
                llm_model,
                message=message,
                temperature=solver_temperature,
                request_timeout=request_timeout,
                max_remaining_time=max_total_time,
                max_remaining_timeouts=max_total_timeouts,
                problem_id=problem_id,
                retries=per_iteration_retries,
            )
        except Exception as e:
            if "Exceeded timeouts allotted to the request" in str(e) or "Exceeded time allotted to the request" in str(e):
                # Exceeded max_remaining_timeouts or max_remaining_time
                print("Exiting early due to exceeding allotted time or timeouts on problem", problem_id)
                break
            # Just exceeded per_iteration_retries, so try the next iteration
            continue

        code = _parse_code_from_llm(response)
        if not code:
            continue

        train_res, test_res = await _eval_on_train_and_test(
            code, train_in, train_out, test_in, timeout_s=timeout_sandbox
        )

        last_train, last_test = train_res, test_res

        if all(r["success"] for r in train_res):
            return ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1
            )

        feedback, score = _build_feedback(train_res, train_in, train_out)
        solutions.append(ARCAGISolution(code=code, feedback=feedback, score=score))

        if score >= best_train_score:
            best_train_score = score
            best_result = ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1
            )

    if return_best and best_result is not None:
        return best_result
    if last_test is None:
        last_test = [
            RunResult(
                success=False,
                output="",
                soft_score=0.0,
                error="Failed to generate any valid solutions.",
                code="",
            )
        ]
    return ARCAGIResult(
        train_results=last_train, results=last_test, iteration=max_iterations
    )


def create_examples(solutions, max_examples=3, improving_order: bool = False):
    template = string.Template("""
<solution_$index>
<solution_code>
```python
$code
```
</solution_code>
<solution_evaluation>
$feedback
</solution_evaluation>
<solution_score>
$score
</solution_score>
</solution_$index>
""")
    if not solutions:
        return ""
    scores = [x["score"] for x in solutions]
    inds = np.argsort(scores)[::-1]
    inds = inds[: min(max_examples, len(inds))]
    if improving_order:
        inds = inds[::-1]

    blocks: list[str] = []
    for k, idx in enumerate(inds, start=1):
        e = solutions[idx]
        blocks.append(
            template.substitute(
                index=k,
                code=e["code"],
                feedback=e["feedback"],
                score=f"{e['score']:.2f}",
            )
        )
    return "\n".join(blocks)


def _build_prompt(base_prompt: str, **fields: str) -> str:
    s = base_prompt
    for k, v in fields.items():
        s = s.replace(f"$${k}$$", v)
    return s


def _array_diff(arr1: np.ndarray, arr2: np.ndarray) -> str:
    rows, cols = arr1.shape
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if arr1[i, j] == arr2[i, j]:
                row.append(str(int(arr1[i, j])))
            else:
                row.append(f"{int(arr1[i, j])}/{int(arr2[i, j])}")
        out.append(" ".join(row))
    return "\n".join(out)


def _parse_code_from_llm(response: str) -> Optional[str]:
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def _soft_score(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape != truth.shape:
        return 0.0
    if truth.size == 0:
        return 1.0
    raw = np.mean(pred == truth)
    return float(np.nan_to_num(raw, posinf=0.0, neginf=0.0))


def _json_to_ndarray(s: str) -> Optional[np.ndarray]:
    try:
        obj = json.loads(s)
        arr = np.array(obj)
        if arr.ndim < 2:
            arr = np.expand_dims(arr, axis=list(range(2 - arr.ndim)))
        return arr.astype(int, copy=False)
    except Exception:
        return None


def _make_example(train_in, train_out, test_in) -> dict[str, Any]:
    train = [
        {"input": iin, "output": oout}
        for iin, oout in zip(train_in, train_out, strict=True)
    ]
    test = [{"input": iin} for iin in test_in]
    return {"train": train, "test": test}


def format_problem(
    problem: dict[str, Any],
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> str:
    train = list(problem["train"])
    test = list(problem["test"])

    if shuffle and len(train) > 1:
        rng = np.random.default_rng(seed if seed is not None else 0)
        perm = rng.permutation(len(train))
        train = [train[i] for i in perm]

    example_str = ""
    challenge_str = ""

    for example_num, example in enumerate(train, start=1):
        example_str += f"""
Example #{example_num}
Input:
<Diagram>
{_example_to_diagram(example["input"])}
</Diagram>

Output:
<Diagram>
{_example_to_diagram(example["output"])}
</Diagram>
"""

    for challenge_num, challenge in enumerate(test, start=1):
        challenge_str += f"""
Challenge #{challenge_num}
Input:
<Diagram>
{_example_to_diagram(challenge["input"])}
</Diagram>
"""

    return example_str + challenge_str


def _example_to_diagram(example: list[list[int]] | np.ndarray) -> str:
    """Converts an ARC-AGI example (list of lists) to a diagram (ascii grid)."""
    diagram = ""
    for row in example:
        row_str = " ".join([str(col) for col in row]) + "\n"
        diagram += row_str
    return diagram[:-1]  # Strip final \n


async def _eval_on_train_and_test(
    code: str,
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    *,
    timeout_s: float = 1.5,
) -> tuple[list[RunResult], list[RunResult]]:
    # Train
    train_results: list[RunResult] = []
    for i, (iin, oout) in enumerate(zip(train_in, train_out, strict=True)):
        ok, out_str = await run(code, iin, timeout_s=timeout_s)
        success = False
        soft = 0.0
        err: Optional[str] = None
        if not ok:
            err = out_str or "Execution failed."
        else:
            arr = _json_to_ndarray(out_str)
            if arr is None:
                err = (
                    f"Failed to parse output as JSON 2D array.\nOutput was:\n{out_str}"
                )
            else:
                truth = np.array(oout)
                success = bool(arr.shape == truth.shape and np.array_equal(arr, truth))
                soft = _soft_score(arr, truth)
        train_results.append(
            RunResult(success=success, output=out_str, soft_score=soft, error=err, code=code)
        )

    # Test
    test_results: list[RunResult] = []
    for i, iin in enumerate(test_in):
        ok, out_str = await run(code, iin, timeout_s=timeout_s)
        err = None if ok else (out_str or "Execution failed.")
        test_results.append(
            RunResult(success=False, output=out_str, soft_score=0.0, error=err, code=code)
        )
    return train_results, test_results


def _parse_json_array_no_expand(s: str) -> Optional[np.ndarray]:
    """Parse JSON into a NumPy array without changing rank or dtype."""
    try:
        return np.array(json.loads(s))
    except Exception:
        return None


def _build_feedback(
    train_results: list[RunResult], train_in, train_out
) -> tuple[str, float]:
    feedback_parts: list[str] = []
    per_example_scores: list[float] = []

    for i, rr in enumerate(train_results):
        if rr["success"]:
            feedback_parts.append(f"Solves Example #{i + 1} correctly. ")
            per_example_scores.append(1.0)
            continue

        msg_lines: list[str] = [f"Solves Example #{i + 1} incorrectly. "]

        pred_raw = _parse_json_array_no_expand(rr["output"]) if rr["output"] else None
        truth = np.array(train_out[i])

        if pred_raw is None:
            per_example_scores.append(0.0)
            msg_lines.append("\nThe output has to be a rectangular grid of numbers.\n")
        else:
            pred_for_display = pred_raw
            if pred_for_display.ndim < 2:
                pred_for_display = np.expand_dims(
                    pred_for_display, axis=list(range(2 - pred_for_display.ndim))
                )

            if pred_raw.shape != truth.shape:
                per_example_scores.append(0.0)
                msg_lines.append(
                    f"\n\nShape mismatch: your prediction's shape was {pred_raw.shape}, "
                    f"while the correct shape was {truth.shape}."
                )
            else:
                # Same shape: show diff grid and compute soft score.
                msg_lines.append(
                    "\nYour code's output does not match the expected output."
                    "\n\nBelow is a visualization of the 2D array your code produced as well as the expected output.\n"
                    "Correctly predicted values are shown as-is while the incorrectly predicted values are shown "
                    "in the format 'prediction/correct':\n"
                )
                diff = _array_diff(pred_for_display, truth)
                msg_lines.append(f"\n```\n{diff}\n```\n")

                example_score = float(np.mean(pred_raw == truth))
                example_score = float(
                    np.nan_to_num(example_score, posinf=0.0, neginf=0.0)
                )
                per_example_scores.append(example_score)
                msg_lines.append(
                    f"Output accuracy: {example_score:.2f} (0 is worst, 1 is best).\n"
                )

        if rr["error"]:
            msg_lines.append(
                f"\n\nYour code produced the following error:\n{rr['error']}\n"
            )

        feedback_parts.append("".join(msg_lines))

    full_feedback = "\n\n".join(feedback_parts)
    mean_score = (
        float(np.mean(np.nan_to_num(per_example_scores, posinf=0.0, neginf=0.0)))
        if per_example_scores
        else 0.0
    )
    return full_feedback, mean_score
