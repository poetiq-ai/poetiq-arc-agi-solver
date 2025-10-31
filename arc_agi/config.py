from arc_agi.prompts import FEEDBACK_PROMPT, SOLVER_PROMPT
from arc_agi.types import ExpertConfig

CONFIG: ExpertConfig = {
  # Prompts
  'solver_prompt': SOLVER_PROMPT,
  'feedback_prompt': FEEDBACK_PROMPT,
  # LLM parameters
  'llm_id': 'openai/gpt-5',
  'solver_temperature': 1.0,
  'request_timeout': 20 * 60, # in seconds
  'max_total_timeouts': 15, # per problem per solver
  'max_total_time': None, # per problem per solver
  'per_iteration_retries': 5,
  # Solver parameters
  'num_experts': 2,
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
}
