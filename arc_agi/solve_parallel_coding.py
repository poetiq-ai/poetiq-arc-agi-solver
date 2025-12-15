import asyncio

from arc_agi.solve_coding import solve_coding
from arc_agi.types import ARCAGIResult, ExpertConfig
from arc_agi.vote import rank_results


async def solve_parallel_coding(
    *,
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    expert_configs: list[ExpertConfig],
    problem_id: str | None = None,
    logging_dir: str | None = None,
) -> list[ARCAGIResult]:
    """
    Run multiple coding experts in parallel, group by identical test outputs, then rank.
    """
    for it, cfg in enumerate(expert_configs):
        cfg["seed"] += it * cfg["max_iterations"]

    # Solve concurrently
    tasks = [
        asyncio.create_task(
            solve_coding(
                train_in=train_in,
                train_out=train_out,
                test_in=test_in,
                config=cfg,
                problem_id=problem_id,
                expert_index=i,
                logging_dir=logging_dir
            )
        )
        for i, cfg in enumerate(expert_configs)
    ]
    results: list[ARCAGIResult] = await asyncio.gather(*tasks)

    ordered = rank_results(results, expert_configs[0])
    return ordered
