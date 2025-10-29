from typing import Literal, Optional, TypedDict

Models = Literal["groq/openai/gpt-oss-120b", "openai/gpt-5", "xai/grok-4-fast"]


class ExpertConfig(TypedDict):
    use_new_voting: bool
    count_failed_matches: bool
    iters_tiebreak: bool
    low_to_high_iters: bool
    solver_prompt: str
    feedback_prompt: str
    llm_id: Models
    max_iterations: int
    solver_temperature: float
    max_solutions: int
    selection_probability: float
    seed: int
    shuffle_examples: bool
    improving_order: bool
    return_best_result: bool
    request_timeout: Optional[int]
    max_total_timeouts: Optional[int]
    max_total_time: Optional[int]
    num_experts: int
    per_iteration_retries: int


MessageRole = Literal["user", "assistant", "system"]


class Message(TypedDict):
    role: MessageRole
    content: str


class RunResult(TypedDict):
    success: bool
    output: str
    soft_score: float
    error: Optional[str]
    code: str


class ARCAGIResult(TypedDict):
    train_results: list[RunResult]
    results: list[RunResult]  # test results
    iteration: int


class ARCAGISolution(TypedDict):
    code: str
    feedback: str
    score: float
