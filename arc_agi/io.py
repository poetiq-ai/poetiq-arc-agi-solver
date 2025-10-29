import ast
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, List, Optional

from arc_agi.types import ARCAGIResult
from arc_agi.official_scorer import ARCScorer, ARCTask, BenchmarkedTaskResults, ScoringResult

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def _attempt_dict(
    task_id: str, pair_index: int, answer: Any, raw: Optional[str], idx: int, result_obj: ARCAGIResult
) -> dict:
    if answer is None:
        answer = "NO_PREDICTION"
    return {
        "answer": answer,
        "result_obj": result_obj,
        "metadata": {
            "model": "unknown",
            "provider": "local",
            "start_timestamp": _now_iso(),
            "end_timestamp": _now_iso(),
            "choices": [
                {
                    "index": idx,
                    "message": {"role": "assistant", "content": (raw or "")},
                    "finish_reason": "stop",
                }
            ],
            "reasoning_summary": None,
            "kwargs": {},
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
            "cost": {"total_cost": 0.0, "prompt_cost": 0.0, "completion_cost": 0.0},
            "task_id": task_id,
            "pair_index": pair_index,
            "test_id": None,
        },
    }


def _safe_parse(raw: Optional[str]):
    if raw is None or raw.strip() == "":
        return []
    try:
        v = ast.literal_eval(raw)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def write_top2_submission(
    task_id: str, results: List[ARCAGIResult], submission_dir: str, expected_pairs: int
) -> str:
    os.makedirs(submission_dir, exist_ok=True)

    # collect raw outputs from top-2
    raw_outputs: List[List[Optional[str]]] = []
    for cand in results[:2]:
        raw_outputs.append([r.get("output") for r in cand.get("results", [])])

    num_pairs = expected_pairs

    test_pairs: List[dict] = []
    for i in range(num_pairs):
        pair_obj = {}
        # attempt_1
        raw1 = (
            raw_outputs[0][i]
            if len(raw_outputs) >= 1 and i < len(raw_outputs[0])
            else None
        )
        pair_obj["attempt_1"] = _attempt_dict(
            task_id, i, _safe_parse(raw1), raw1, idx=0, result_obj=results[0]
        )
        # attempt_2
        if len(raw_outputs) >= 2 and i < len(raw_outputs[1]) and raw_outputs[1][i]:
          raw2 = raw_outputs[1][i]
          pair_obj["attempt_2"] = _attempt_dict(
              task_id, i, _safe_parse(raw2), raw2, idx=1, result_obj=results[1]
          )

        test_pairs.append(pair_obj)

    out_path = os.path.join(submission_dir, f"{task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_pairs, f)
    return out_path


def write_top_k_submission(
    task_id: str, results: List[ARCAGIResult], submission_dir: str, expected_pairs: int, max_k: int = 2
) -> str:
    os.makedirs(submission_dir, exist_ok=True)

    # collect raw outputs from top-k
    raw_outputs: List[List[Optional[str]]] = []
    for cand in results[:max_k]:
        raw_outputs.append([r.get("output") for r in cand.get("results", [])])

    num_pairs = expected_pairs

    test_pairs: List[dict] = []

    for i in range(num_pairs):
        k_shot_obj = {}
        for attempt, raw in enumerate(raw_outputs):
            raw_i = raw[i] if i < len(raw) else None
            k_shot_obj[f"attempt_{attempt + 1}"] = _attempt_dict(
                task_id, i, _safe_parse(raw_i), raw_i, idx=attempt, result_obj=results[attempt]
            )

        test_pairs.append(k_shot_obj)

    out_path = os.path.join(submission_dir, f"{task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_pairs, f, indent=4)
    return out_path


def get_all_scoring(arcagi_scorer: ARCScorer, task_id: str, submission_path: Path, max_k: int = 2) -> list[ScoringResult]:
    """
    Scores a single task submission against the solutions.
    Returns a list of dictionaries containing task_score, task_cost, num_attempts, one for each of up to `max_k` attempts.
    """
    with submission_path.open() as f:
        full_json_data = json.load(f)

    task = ARCTask.from_dict(arcagi_scorer.solutions[task_id])  # type: ignore

    scores = []
    for attempt in range(1, max_k + 1):
        json_data = []
        for data in full_json_data:
            json_data.append({"attempt_1": data.get(f"attempt_{attempt}", {})})
        
        task_submission = BenchmarkedTaskResults(test_pairs=json_data)
        scores.append(arcagi_scorer.score_task(task, task_submission))

    return scores
