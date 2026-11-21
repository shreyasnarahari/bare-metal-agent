from __future__ import annotations

import json
import math
from typing import Any


class StrictJSONError(ValueError):
    pass


def strict_json_loads(text: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise StrictJSONError(f"Invalid JSON numeric constant: {constant}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise StrictJSONError("JSON numbers must be finite")
        return number

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise StrictJSONError(f"Duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise StrictJSONError(str(exc)) from exc


__all__ = ["StrictJSONError", "strict_json_loads"]
