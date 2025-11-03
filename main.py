import asyncio
import json
import os
import resource
import time
import traceback
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from arc_agi.config import CONFIG_LIST
from arc_agi.io import build_kaggle_two_attempts
from arc_agi.scoring import score_task
from arc_agi.solve import solve

load_dotenv()


# time the run started, so multiple runs don't collide
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# challenge input file
DATA_CHALLENGES = os.path.join(os.path.dirname(__file__), "data", "arc-prize-2024", "arc-agi_evaluation_challenges.json")
# optional challenge solution file
DATA_SOLUTIONS = os.path.join(os.path.dirname(__file__), "data", "arc-prize-2024", "arc-agi_evaluation_solutions.json")
# where to write outputs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT = os.path.join(OUTPUT_DIR, f"submission_{TIMESTAMP}.json")

# number of problems (None = all)
NUM_PROBLEMS = None
# select particular problems
SELECTED_PROBLEMS = [] # e.g. ['b7999b51']


async def _eval_task_data(task_id: str, task: dict) -> tuple[str, Optional[list[dict]], Optional[str], float]:
    """
    Returns: (task_id, kaggle_preds | None on error, error, elapsed_seconds)
    """
    start = time.time()
    try:
        train = task.get("train", [])
        test = task.get("test", [])
        train_in = [ex["input"] for ex in train]
        train_out = [ex["output"] for ex in train]
        test_in = [ex["input"] for ex in test]

        results = await solve(train_in, train_out, test_in, problem_id=task_id)
        kaggle_preds = build_kaggle_two_attempts(results, test_in)

        return task_id, kaggle_preds, None, time.time() - start
    except Exception:
        return task_id, None, traceback.format_exc(), time.time() - start


async def main():
    # Ensure we don't run out of file handles
    # Get current soft and hard limits
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set a new soft limit (cannot exceed hard limit)
    new_soft = 65536
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print(f"Writing config_{TIMESTAMP}.json to output directory...")
    with open(os.path.join(OUTPUT_DIR, f"config_{TIMESTAMP}.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG_LIST, f, indent=4)

    # Load challenges
    with open(DATA_CHALLENGES, "r", encoding="utf-8") as f:
        challenges_blob: dict[str, dict] = json.load(f)

    # Load solutions if present; disable scoring if missing/unreadable
    solutions_blob: Optional[dict[str, list]] = None
    if DATA_SOLUTIONS and os.path.exists(DATA_SOLUTIONS):
        try:
            with open(DATA_SOLUTIONS, "r", encoding="utf-8") as f:
                solutions_blob = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load solutions file '{DATA_SOLUTIONS}': {e}\nScoring will be disabled.")

    items = list(challenges_blob.items())
    if SELECTED_PROBLEMS:
        sel = set(SELECTED_PROBLEMS)
        items = [it for it in items if it[0] in sel]
    if NUM_PROBLEMS is not None:
        items = items[:NUM_PROBLEMS]


    print(f"Running {len(items)} problems from {DATA_CHALLENGES}...")
    print("Scoring:", "enabled" if solutions_blob is not None else "disabled (no solutions)")

    start = time.time()

    submission: dict[str, list[dict]] = {}

    # running scores only if solutions available
    per_task_scores: dict[str, float] = {}
    total = 0
    correct = 0.0
    incorrect = 0.0

    tasks = [asyncio.create_task(_eval_task_data(task_id, task)) for task_id, task in items]

    for coro in asyncio.as_completed(tasks):
        task_id, preds, err, elapsed = await coro

        if err is not None or preds is None:
            print(f"! {task_id} (error in {round(elapsed)}s)\n{err}")
            submission[task_id] = []
        else:
            submission[task_id] = preds

            # running scores if solutions available
            if solutions_blob is not None and task_id in solutions_blob:
                gt_outputs = solutions_blob[task_id]
                task_score = score_task(preds, gt_outputs)
                per_task_scores[task_id] = task_score
                total += 1
                correct += task_score
                incorrect += 1 - task_score
                mark = "✓" if task_score == 1.0 else "✗"
                print(f"{mark} {task_id} ({round(elapsed)}s) [{correct}/{total}]")
            else:
                print(f"· {task_id} ({round(elapsed)}s)")

        # write cumulative Kaggle output after each task
        try:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(submission, f)
        except Exception as e:
            print(f"WARNING: Failed to write partial output to {OUTPUT}: {e}")

    total_time = time.time() - start

    print("\n=== Summary ===")
    print(f"Data file: {DATA_CHALLENGES}")
    print(f"Problems: {len(items)}")
    if solutions_blob is not None and per_task_scores:
        acc = correct / total
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Accuracy: {acc * 100:.3f}")
    else:
        print("Scoring: disabled or no tasks matched in solutions.")
    print(f"Total time: {round(total_time)}s")

    # final write just in case
    try:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(submission, f)
        print(f"\nWrote Kaggle submission to: {OUTPUT}")
    except Exception as e:
        print(f"ERROR: Final write to {OUTPUT} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
