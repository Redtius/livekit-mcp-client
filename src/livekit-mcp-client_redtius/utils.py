import keyword
import re

# JSON type mapper
def map_json_type_to_py(json_type: str):
    return {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list
    }.get(json_type, str)

def sanitize_name(name):
    # replace hyphens with underscores and prepend underscore if needed
    safe = re.sub(r'\W|^(?=\d)', '_', name)
    if keyword.iskeyword(safe):
        safe += '_'
    return safe