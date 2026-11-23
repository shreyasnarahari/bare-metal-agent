"""A BFCL model handler for Toolsmith results, so BFCL's official checker can score them.

The handler is registered under a name per model and mode; BFCL then reads the
result files Toolsmith wrote and scores them with its own AST checker.
"""

from __future__ import annotations

import json
from typing import Any


def decode_standardized_result(result: Any) -> list[dict[str, Any]]:
    """``[{name: arguments}]`` with JSON-string arguments decoded, as BFCL's checker expects."""
    if not isinstance(result, list):
        raise TypeError("A function-calling result must be a list")
    decoded = []
    for invoked_function in result:
        if not isinstance(invoked_function, dict) or len(invoked_function) != 1:
            raise ValueError("Each call must contain exactly one function name")
        name = next(iter(invoked_function))
        arguments = invoked_function[name]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise TypeError("Function arguments must be a JSON object")
        decoded.append({name: arguments})
    return decoded


def handler_type() -> type:
    """The handler class, built lazily so bfcl-eval is needed only when scoring."""
    try:
        from bfcl_eval.model_handler.base_handler import BaseHandler
        from bfcl_eval.model_handler.utils import convert_to_function_call
    except ImportError as exc:
        raise ImportError("Install the pinned bfcl-eval package for BFCL scoring") from exc

    class ToolsmithBFCLHandler(BaseHandler):
        def decode_ast(self, result, language, has_tool_call_tag):
            if self.is_fc_model:
                return decode_standardized_result(result)
            raise ValueError("Toolsmith BFCL handlers require FC mode")

        def decode_execute(self, result, has_tool_call_tag):
            if self.is_fc_model:
                return convert_to_function_call(decode_standardized_result(result))
            raise ValueError("Toolsmith BFCL handlers require FC mode")

    return ToolsmithBFCLHandler


def register_handler(registry_name: str, *, model_name: str) -> Any:
    """Add the handler to BFCL's model registry under ``registry_name`` (in this process only)."""
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    except ImportError as exc:
        raise ImportError("Install the pinned bfcl-eval package for BFCL scoring") from exc
    config = ModelConfig(
        model_name=model_name,
        display_name=registry_name,
        url="https://github.com/shreyasnarahari/bare-metal-agent",
        org="Toolsmith",
        license="Apache-2.0",
        model_handler=handler_type(),
        is_fc_model=True,
        underscore_to_dot=True,
    )
    MODEL_CONFIG_MAPPING[registry_name] = config
    return config


__all__ = ["decode_standardized_result", "handler_type", "register_handler"]
