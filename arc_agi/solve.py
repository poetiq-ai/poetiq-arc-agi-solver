from arc_agi.config import CONFIG
from arc_agi.solve_parallel_coding import solve_parallel_coding
from arc_agi.types import ARCAGIResult


async def solve(
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    problem_id: str | None = None,
) -> list[ARCAGIResult]:
    result = await solve_parallel_coding(
        train_in=train_in,
        train_out=train_out,
        test_in=test_in,
        expert_configs=[CONFIG.copy() for _ in range(CONFIG["num_experts"])],
        problem_id=problem_id,
    )

    return result
