import re
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_model(raw: str, model: type[M]) -> M:
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return model.model_validate_json(text[start : end + 1])
