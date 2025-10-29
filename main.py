import asyncio
import glob
import json
import os
import resource
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from arc_agi.config import CONFIG
from arc_agi.io import (get_all_scoring, write_top2_submission,
                        write_top_k_submission)
from arc_agi.official_scorer import ARCScorer
from arc_agi.solve import solve

load_dotenv()

# "v1" or "v2"
DATASET = "v1"
# split: "training" or "evaluation"
SPLIT = "evaluation"
# number of problems (None = all)
NUM_PROBLEMS = None
# select particular problems
SELECTED_PROBLEMS = [
    # 'b7999b51',
    # 'd931c21c'
]
# turn on DEBUG mode
DEBUG = True

# time the run started, so multiple runs don't collide
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# path to data files
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", DATASET, SPLIT)
# where to write predictions
SUBMISSIONS_DIR = os.path.join(os.path.dirname(__file__), "submissions", DATASET, SPLIT, f"run_{TIMESTAMP}")
# where official scorer will write results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", DATASET, SPLIT, f"run_{TIMESTAMP}")

arc_scorer = ARCScorer(
    task_dir=DATA_ROOT,
    submission_dir=SUBMISSIONS_DIR,
    print_logs=False,
    results_dir=RESULTS_DIR,
)


async def eval_file(path: str) -> tuple[str, float, Optional[str], float, list]:
    """
    Returns: (name, score, error, elapsed_time)
    """
    task_id = os.path.splitext(os.path.basename(path))[0]
    start_time = time.time()
    try:
        with open(path, "r", encoding="utf-8") as f:
            task = json.load(f)

        train = task.get("train", [])
        test = task.get("test", [])
        train_in = [ex["input"] for ex in train]
        train_out = [ex["output"] for ex in train]
        test_in = [ex["input"] for ex in test]

        results = await solve(train_in, train_out, test_in, problem_id=task_id)

        if DEBUG:
            submission_path = write_top_k_submission(
                task_id=task_id, results=results, submission_dir=SUBMISSIONS_DIR, expected_pairs=len(test_in), max_k=CONFIG["num_experts"]
            )
            all_scores = [res.score for res in get_all_scoring(arc_scorer, task_id, Path(submission_path), max_k=CONFIG["num_experts"])]
            score = max(all_scores[:2]) if all_scores else 0.0
        else:
            submission_path = write_top2_submission(
                task_id=task_id, results=results, submission_dir=SUBMISSIONS_DIR, expected_pairs=len(test_in)
            )
            all_scores = []
            score = arc_scorer.score_task_from_file(task_id, Path(submission_path)).score

        elapsed_time = time.time() - start_time
        return task_id, score, None, elapsed_time, all_scores
    except Exception as e:
        print(f"File eval failed due to exception: {task_id}")
        print(str(e))
        elapsed_time = time.time() - start_time
        return task_id, 0, traceback.format_exc(), elapsed_time, []


async def main():
    start_time = time.time()
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.json")))

    # Ensure we don't run out of file handles
    # Get current soft and hard limits
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set a new soft limit (cannot exceed hard limit)
    new_soft = 65536
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

    if SELECTED_PROBLEMS:
        files = [f for f in files if Path(f).stem in SELECTED_PROBLEMS]

    if NUM_PROBLEMS is not None:
        files = files[:NUM_PROBLEMS]

    print("Writing config.json to submission directory...")
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    # All-caps filename prevents the arcagi official scorer from trying to treat it like
    # a submission, without requiring changing their code.
    with open(os.path.join(SUBMISSIONS_DIR, "CONFIG.JSON"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4)

    print(f"Running {len(files)} problems from {DATASET}/{SPLIT} split...")

    total = 0
    correct = 0
    incorrect = 0
    all_scores = []

    tasks = [asyncio.create_task(eval_file(p)) for p in files]
    for coro in asyncio.as_completed(tasks):
        task_id, score, err, elapsed, scores = await coro
        all_scores.append(scores)
        total += 1
        if err is not None:
            incorrect += 1
            print(
                f"! {task_id} (error in {round(elapsed)}s) [{correct}/{total}]\n{err}"
            )
        else:
            correct += score
            incorrect += 1 - score

            if score == 1.0:
                print(f"✓ {task_id} ({round(elapsed)}s) [{correct}/{total}] scores: {scores}")
            else:
                print(f"✗ {task_id} ({round(elapsed)}s) [{correct}/{total}] scores: {scores}")

    acc = (correct / total) if total else 0.0
    total_time = time.time() - start_time

    if DEBUG:
        print(f"\nAll Scores:\n{all_scores}\n")

        max_num_scores = max([len(scores) for scores in all_scores])
        scores = np.zeros((len(all_scores), max_num_scores))
        for i, scores_list in enumerate(all_scores):
            scores[i, :len(scores_list)] = np.array(scores_list)

        # Compute top-K scores
        kshot_scores = [float(np.max(scores[:, :i + 1], axis=1).mean()) * 100.0 for i in range(scores.shape[-1])]
        print(f"\nK-shot Scores:\n{kshot_scores}\n")

    print("=== Summary ===")
    print(f"Split: {SPLIT}")
    print(f"Problems: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {acc * 100:.3f}")
    print(f"Total time: {round(total_time)}s")

    print("\n=== Official Scoring ===", end='')
    _total_score, _total_tasks = arc_scorer.score_submission()

    if DEBUG:
        print("\nWARNING: Official scoring in DEBUG mode reports final k-shot result, rather than the 2-shot result."
              "\nSee second result in 'K-shot Scores' above for accurate official scoring.")


if __name__ == "__main__":
    asyncio.run(main())
