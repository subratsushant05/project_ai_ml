"""A tiny deterministic toolset for the bundled example agents.

Three tools -- ``calculator``, ``weather``, ``web_search`` -- all backed by
in-repo fixtures so demos and tests are fast, offline, and reproducible.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

WEATHER_FIXTURES: dict[str, dict[str, Any]] = {
    "paris": {"temp_c": 18, "conditions": "partly cloudy"},
    "tokyo": {"temp_c": 24, "conditions": "sunny"},
    "berlin": {"temp_c": 11, "conditions": "rainy"},
    "london": {"temp_c": 14, "conditions": "overcast"},
}

SEARCH_CORPUS: list[dict[str, str]] = [
    {
        "title": "Acme Corporation",
        "text": (
            "Acme Corporation is an industrial supplier. Acme Corporation was "
            "founded by Jane Smith in 1998 in Portland."
        ),
        "answer": "Acme Corporation was founded by Jane Smith in 1998.",
    },
    {
        "title": "Voyager 1",
        "text": (
            "Voyager 1 is a space probe launched by NASA in 1977. The Voyager 1 "
            "probe studies the outer heliosphere and interstellar space."
        ),
        "answer": "The Voyager 1 probe studies the outer heliosphere and interstellar space.",
    },
    {
        "title": "Zephyr framework",
        "text": (
            "Zephyr is an open-source web framework. The Zephyr framework is "
            "written in Rust and focuses on low-latency services."
        ),
        "answer": "The Zephyr framework is written in Rust.",
    },
    {
        "title": "Portland",
        "text": "Portland is a city in Oregon known for its bridges and coffee.",
        "answer": "Portland is a city in Oregon.",
    },
]

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression.

    Args:
        expression: Arithmetic over ``+ - * /`` with parentheses, e.g.
            ``"(5 + 3) * 2"``.

    Returns:
        The result as a string; integral results render without a decimal.

    Raises:
        ValueError: If the expression contains anything but basic arithmetic.
    """
    result = _eval_node(ast.parse(expression, mode="eval"))
    return str(int(result)) if result == int(result) else str(round(result, 6))


def weather(city: str) -> str:
    """Look up weather for a city from bundled fixture data.

    Args:
        city: City name (case-insensitive).

    Returns:
        A one-line report like ``"18 C, partly cloudy"``.

    Raises:
        KeyError: If the city is not in the fixtures.
    """
    record = WEATHER_FIXTURES[city.lower()]
    return f"{record['temp_c']} C, {record['conditions']}"


def best_search_doc(query: str) -> dict[str, str]:
    """Return the corpus document with the highest keyword overlap.

    Args:
        query: Free-text search query.

    Returns:
        The best-matching document (ties broken by corpus order).
    """
    query_tokens = {tok for tok in query.lower().split() if len(tok) > 2}

    def overlap(doc: dict[str, str]) -> int:
        doc_tokens = set((doc["title"] + " " + doc["text"]).lower().split())
        return len(query_tokens & doc_tokens)

    return max(SEARCH_CORPUS, key=overlap)


def web_search(query: str) -> str:
    """Search the bundled fixture corpus.

    Args:
        query: Free-text search query.

    Returns:
        The text of the best-matching document.
    """
    return best_search_doc(query)["text"]


TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "weather": weather,
    "web_search": web_search,
}
