# https://github.com/arcprize/arc-agi-benchmarking/blob/main/src/arc_agi_benchmarking/utils/parsing.py

import json
import re
from typing import Callable, List, Optional

# Type hint for the provider's JSON extraction function (can be refined later)
ProviderJsonExtractor = Callable[[str], Optional[List[List[int]]]]

# --- Helper Parsing Functions: Can be added to as needed ---

def backscan_json_parser(log_str: str) -> Optional[List[List[int]]]:
    """
    Extract the last valid JSON substring that matches the List[List] structure
    from the given log string by scanning backwards from the end.

    Parameters:
        log_str (str): The full log output text.

    Returns:
        The parsed List[List] object if found and valid, otherwise None.
    """
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(log_str) - 1, -1, -1):
        char = log_str[i]
        if char in (']', '}'):
            last_bracket_idx = i
            closing_bracket = char
            break

    if last_bracket_idx == -1:
        return None

    opening_bracket = '[' if closing_bracket == ']' else '{'

    bracket_counter = 1 # Start at 1 to account for the found closing bracket
    start_idx = -1

    for i in range(last_bracket_idx - 1, -1, -1):
        char = log_str[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break

    if start_idx == -1:
        return None

    json_candidate = log_str[start_idx:last_bracket_idx+1]

    try:
        parsed_json = json.loads(json_candidate)

        # Validate the structure: must be a non-empty list of lists.
        if isinstance(parsed_json, list) and parsed_json and all(isinstance(row, list) for row in parsed_json):
            return parsed_json
        else:
            return None

    except json.JSONDecodeError:
        return None

def extract_from_boxed(log_str: str) -> Optional[List[List[int]]]:
    """
    Extracts JSON from a LaTeX-style \\boxed{} command in a string.
    """
    match = re.search(r"\\boxed\{(.*?)\}", log_str, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            # The content inside boxed is often a list of lists
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all(isinstance(i, list) for i in parsed_json):
                return parsed_json
        except json.JSONDecodeError:
            # If json.loads fails, it's not the JSON we're looking for.
            pass
    return None


# --- Main Parsing Orchestrator ---

def parse_and_validate_json(response: str) -> Optional[List[List[int]]]:
    """
    Orchestrates parsing by trying the provider extractor.
    Returns the parsed List[List[int]] or raises ValueError if validation fails.
    """
    parsing_attempts = [
        extract_from_boxed,
        backscan_json_parser
    ]

    for parser in parsing_attempts:
        result = parser(response)
        if result is not None:
            # Validate the structure: must be list of lists
            if isinstance(result, list) and all(isinstance(row, list) for row in result):
                return result # Return immediately on first success and validation
            else:
                # Raise error if structure is invalid, triggering retry in main loop
                raise ValueError(f"Parser {parser.__name__} produced invalid structure: {result!r}")

    # Raise an error here if no method works, triggering retry in main loop
    raise ValueError(f"Failed to parse response after all attempts: {response!r}")

if __name__ == "__main__":
    response = """### Summary and Final Answer:\nThe pattern transforms the input, which consists of two lists, into six lists in the output. Each of the first four output lists is constructed by repeating the original or reversed input lists three times. The fifth and sixth output lists are repetitions of the first and second output lists, respectively.\n\nFor the test input `[[3, 2], [7, 8]]`:\n1. The first input list `[3, 2]` is repeated three times to form the first output list.\n2. The second input list `[7, 8]` is repeated three times to form the second output list.\n3. The reversed first input list `[2, 3]` is repeated three times to form the third output list.\n4. The reversed second input list `[8, 7]` is repeated three times to form the fourth output list.\n5. The first output list is repeated as the fifth output list.\n6. The second output list is repeated as the sixth output list.\n\nThus, the output is:\n\n```python\n[[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]\n```\n\nFinal answer (formatted to match the training output style):\n\n```python\n[[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]\n```\n\n\\boxed{[[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]}'
"""
    print(parse_and_validate_json(response))