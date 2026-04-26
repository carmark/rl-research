"""Calculator tool for the Agentic RL environment."""

from __future__ import annotations

import math
import logging
from typing import Any

from .base import BaseTool

logger = logging.getLogger(__name__)

# Restricted set of safe names for eval
_SAFE_MATH_NAMES = {
    k: getattr(math, k)
    for k in [
        "sqrt", "pow", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "ceil", "floor", "fabs", "factorial", "gcd",
        "pi", "e", "inf",
    ]
}
_SAFE_MATH_NAMES.update({
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
})


class CalculatorTool(BaseTool):
    """Safe mathematical expression evaluator.

    Supports standard arithmetic, math functions, and common constants.
    Uses a restricted eval environment to prevent code injection.
    """

    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Supports arithmetic operators "
        "(+, -, *, /, **, %), math functions (sqrt, log, sin, cos, etc.), "
        "and constants (pi, e)."
    )
    parameters = {
        "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate, e.g. 'sqrt(2) + 3 * pi'",
        },
    }

    def execute(self, expression: str, **kwargs: Any) -> float | int:
        # Sanitize: reject dangerous patterns
        forbidden = ["import", "__", "exec", "eval", "open", "os.", "sys."]
        expr_lower = expression.lower()
        for pat in forbidden:
            if pat in expr_lower:
                raise ValueError(f"Expression contains forbidden pattern: {pat}")

        try:
            result = eval(expression, {"__builtins__": {}}, _SAFE_MATH_NAMES)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")

        if isinstance(result, (int, float)):
            return result
        raise ValueError(f"Expression did not return a number: {type(result)}")
