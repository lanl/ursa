import json
import re


def extract_json(text: str) -> list[dict]:
    """
    Extract a JSON object or array from text that might contain markdown or other content.

    The function attempts three strategies:
        1. Extract JSON from a markdown code block labeled as JSON.
        2. Extract JSON from any markdown code block.
        3. Use bracket matching to extract a JSON substring starting with '{' or '['.

    Returns:
        A Python object parsed from the JSON string (dict or list).

    Raises:
        ValueError: If no valid JSON is found.
    """
    # Approach 1: Look for a markdown code block specifically labeled as JSON.
    labeled_block = re.search(
        r"```json\s*([\[{].*?[\]}])\s*```", text, re.DOTALL
    )
    if labeled_block:
        json_str = labeled_block.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fall back to the next approach if parsing fails.
            pass

    # Approach 2: Look for any code block delimited by triple backticks.
    generic_block = re.search(r"```(.*?)```", text, re.DOTALL)
    if generic_block:
        json_str = generic_block.group(1).strip()
        if json_str.startswith("{") or json_str.startswith("["):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # Approach 3: Attempt to extract JSON using bracket matching.
    # Find the first occurrence of either '{' or '['.
    first_obj = text.find("{")
    first_arr = text.find("[")
    if first_obj == -1 and first_arr == -1:
        raise ValueError("No JSON object or array found in the text.")

    # Determine which bracket comes first.
    if first_obj == -1:
        start = first_arr
        open_bracket = "["
        close_bracket = "]"
    elif first_arr == -1:
        start = first_obj
        open_bracket = "{"
        close_bracket = "}"
    else:
        if first_obj < first_arr:
            start = first_obj
            open_bracket = "{"
            close_bracket = "}"
        else:
            start = first_arr
            open_bracket = "["
            close_bracket = "]"

    # Bracket matching: find the matching closing bracket.
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == open_bracket:
            depth += 1
        elif text[i] == close_bracket:
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError(
            "Could not find matching closing bracket for JSON content."
        )

    json_str = text[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError("Extracted content is not valid JSON.") from e
