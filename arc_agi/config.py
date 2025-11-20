from arc_agi.prompts import FEEDBACK_PROMPT, SOLVER_PROMPT_1, SOLVER_PROMPT_2, SOLVER_PROMPT_3
from arc_agi.types import ExpertConfig

# To run Poetiq(Gemini-3-a):
NUM_EXPERTS = 1
# To run Poetiq(Gemini-3-b):
# NUM_EXPERTS = 2
# To run Poetiq(Gemini-3-c):
# NUM_EXPERTS = 8

CONFIG_LIST: list[ExpertConfig] = [
  {
    # Prompts
    'solver_prompt': SOLVER_PROMPT_1,
    'feedback_prompt': FEEDBACK_PROMPT,
    # LLM parameters
    'llm_id': 'gemini/gemini-3-pro-preview',
    'solver_temperature': 1.0,
    'request_timeout': 60 * 60, # in seconds
    'max_total_timeouts': 15, # per problem per solver
    'max_total_time': None, # per problem per solver
    'per_iteration_retries': 2,
    # Solver parameters
    'num_experts': 1,
    'max_iterations': 10,
    'max_solutions': 5,
    'selection_probability': 1.0,
    'seed': 0,
    'shuffle_examples': True,
    'improving_order': True,
    'return_best_result': True,
    # Voting parameters
    'use_new_voting': True,
    'count_failed_matches': True,
    'iters_tiebreak': False,
    'low_to_high_iters': False,
  },
] * NUM_EXPERTS
