import keyword
import re


# JSON type mapper
def translate_type(json_type: str):
    return {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }.get(json_type, str)


def sanitize_name(name: str) -> str:
    """
    Convert a string to a valid Python identifier by:
    1. Replacing all non-alphanumeric chars with underscores.
    2. Ensuring it doesn't start with a digit.
    3. Avoiding Python keywords/reserved names.
    4. Ensuring uniqueness (if needed, caller should handle collisions).
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected string, got {type(name).__name__}")

    # Replaces all non-alphanumeric chars (except underscores) with '_'
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name)

    # Prepends '_' if the name starts with a digit
    if safe and safe[0].isdigit():
        safe = "_" + safe

    # Avoids Python keywords/reserved names
    if keyword.iskeyword(safe):
        safe += "_"

    # Ensures the name isn't empty after sanitization
    if not safe:
        safe = "_empty"

    return safe
