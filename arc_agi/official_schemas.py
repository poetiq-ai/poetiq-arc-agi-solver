# https://github.com/arcprize/arc-agi-benchmarking/blob/main/src/arc_agi_benchmarking/schemas.py

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator, model_validator


class APIType:
    """
    Enum for the different API types that can be used with the OpenAI API.
    """
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"

class ARCTaskOutput(BaseModel):
    output: List[List[int]]

class ARCPair(BaseModel):
    input: List[List[int]]
    output: Optional[List[List[int]]] = None

class ARCTask(BaseModel):
    train: List[ARCPair]
    test: List[ARCPair]

    def __eq__(self, other: 'ARCTask') -> bool:
        if not isinstance(other, ARCTask):
            return False
        return (len(self.train) == len(other.train) and
                len(self.test) == len(other.test) and
                all(a == b for a, b in zip(self.train, other.train)) and
                all(a == b for a, b in zip(self.test, other.test)))

    def __repr__(self) -> str:
        n_train = len(self.train)
        n_test = len(self.test)
        task_hash = self.get_hash()
        return f"ARCTask({n_train} train + {n_test} test, {task_hash})"

    def __str__(self) -> str:
        return self.__repr__()

    def get_hash(self) -> str:
        """Generate a stable hash of the form XXXXX-XXXXX-XXXXX_XXXXX-XXXXX-XXXXX where
        the first set is one hash for each train pair, and the second set is one hash for each test pair"""
        # Generate hash for each train pair
        train_hashes = []
        for pair in self.train:
            pair_json = json.dumps(pair.model_dump(), sort_keys=True)
            pair_hash = hashlib.sha256(pair_json.encode()).hexdigest()[:5]
            train_hashes.append(pair_hash)

        # Generate hash for each test pair
        test_hashes = []
        for pair in self.test:
            pair_json = json.dumps(pair.model_dump(), sort_keys=True)
            pair_hash = hashlib.sha256(pair_json.encode()).hexdigest()[:5]
            test_hashes.append(pair_hash)

        # Format as XXXXX-XXXXX-XXXXX_XXXXX-XXXXX-XXXXX
        train_part = "-".join(train_hashes)
        test_part = "-".join(test_hashes)

        return f"{train_part}_{test_part}"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ARCTask':
        if isinstance(data, str):
            data = json.loads(data)
        return cls.model_validate(data)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ARCTask':
        with open(filepath, 'r') as f:
            return cls.from_dict(json.loads(f.read()))

    def save_to_file(self, filepath: str, indent: int = 2) -> None:
        with open(filepath, 'w') as f:
            f.write(self.to_json(indent=indent))


class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails

class Cost(BaseModel):
    prompt_cost: float
    completion_cost: float # Cost of completion_tokens * output_cost_per_token
    reasoning_cost: Optional[float] = None  # Cost of reasoning_tokens * output_cost_per_token. Optional as not all providers return it.
    total_cost: float      # Sum of prompt_cost, completion_cost, and reasoning_cost

class AttemptMetadata(BaseModel):
    model: str
    provider: str
    start_timestamp: datetime
    end_timestamp: datetime
    choices: List[Choice]
    reasoning_summary: Optional[str] = None
    kwargs: Dict[str, Any]
    usage: Usage
    cost: Cost
    task_id: Optional[str] = None
    pair_index: Optional[int] = 0
    test_id: Optional[str] = None

    model_config = {
        'json_encoders': {
            datetime: lambda v: v.isoformat()
        }
    }

    def __str__(self):
        """
        Customize string representation for prettier output
        """
        return json.dumps(self.model_dump(), indent=2, default=str)

    __repr__ = __str__

class Attempt(BaseModel):
    answer: Union[str, List[List[int]]]
    metadata: AttemptMetadata
    correct: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def check_answer_present(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the required 'answer' field exists."""
        if isinstance(values, dict) and "answer" not in values:
            raise KeyError("answer")
        return values

    @field_validator('answer', mode='before')
    @classmethod
    def parse_answer(cls, v: Union[str, List[List[int]]]):
        """Parse answer strings using the backscan_json_parser."""
        if isinstance(v, str):
            from arc_agi.official_parsing import parse_and_validate_json

            parsed = parse_and_validate_json(v)
            if parsed is not None:
                return parsed
        return v

class TestPairAttempts(BaseModel):
    attempts: List[Optional[Attempt]]

    @model_validator(mode='before')
    @classmethod
    def validate_attempts(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dictionary of attempts with keys like 'attempt_1', 'attempt_2' to a list of attempts
        (one for each pair) if the attempts field is a dictionary.
        """
        if isinstance(values, list):
            return values

        attempts = values
        if isinstance(attempts, dict):
            # Check if keys follow the pattern 'attempt_X'
            attempt_list = []
            # Sort keys to ensure correct order (attempt_1, attempt_2, etc.)
            for key in sorted(attempts.keys()):
                if key.startswith('attempt_'):
                    attempt_list.append(attempts[key])

            values['attempts'] = attempt_list


        return values

    def __getitem__(self, index):
        """
        Allow subscripting to access attempts directly.
        """
        if isinstance(self.attempts, list):
            return self.attempts[index]
        elif isinstance(self.attempts, dict):
            if isinstance(index, int):
                # Convert to attempt_X format
                key = f"attempt_{index+1}"
                return self.attempts.get(key)
            return self.attempts.get(index)

    def __len__(self):
        """
        Return the number of attempts.
        """
        if isinstance(self.attempts, list):
            return len(self.attempts)
        elif isinstance(self.attempts, dict):
            return len(self.attempts)
        return 0

    def __iter__(self):
        """
        Allow iteration over attempts.
        """
        if isinstance(self.attempts, list):
            return iter(self.attempts)
        elif isinstance(self.attempts, dict):
            # Sort keys to ensure consistent iteration order
            return iter([self.attempts[key] for key in sorted(self.attempts.keys())])

class BenchmarkedTaskResults(BaseModel):
    """
    Top level object for a tested task, consisting of a list of tested test pairs
    """
    test_pairs: List[TestPairAttempts]

    def __len__(self):
        return len(self.test_pairs)

    def __getitem__(self, index):
        return self.test_pairs[index]

    def __iter__(self):
        return iter(self.test_pairs)

class ModelPricing(BaseModel):
    date: str
    input: float
    output: float

class ModelConfig(BaseModel):
    """
    A model configuration used to populate a models kwargs and calculate pricing metadata. Not all fields are required by all providers.
    Points to model.yml
    """
    name: str  # This is now the config_name
    model_name: str  # The actual model name to use with the provider's API
    provider: str
    pricing: ModelPricing
    api_type: Optional[str] = APIType.CHAT_COMPLETIONS # currently only used by openai
    kwargs: Dict[str, Any] = {}

    model_config = {
        'protected_namespaces': (),
        'extra': 'allow'
    }

    @model_validator(mode='before')
    @classmethod
    def extract_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all extra fields into kwargs"""
        if not isinstance(values, dict):
            return values

        kwargs = {}
        known_fields = {'name', 'provider', 'pricing', 'kwargs', 'model_name', 'api_type'}

        for field_name, value in values.items():
            if field_name not in known_fields:
                kwargs[field_name] = value

        # Update the kwargs field with our extracted values
        if kwargs:
            values['kwargs'] = {**kwargs, **values.get('kwargs', {})}

            # Remove the extracted fields from the top level
            for field_name in kwargs:
                if field_name in values:
                    del values[field_name]

        return values

class ScoringResult(BaseModel):
    """
    Result of scoring a task, containing the score (accuracy), cost, and number of attempts.
    """
    score: float  # Score between 0.0 and 1.0 representing accuracy
    total_cost: float   # Total cost of all attempts
    attempts: int # Total number of attempts made
    output_tokens: int # Total number of output tokens used
    duration: float # Total duration of all attempts in seconds
    num_attempts_with_empty_list: int # Number of attempts that returned an empty list