"""Core record type shared across the data layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Example:
    """A single instruction-tuning example (helpdesk ticket -> resolution).

    Attributes:
        id: Stable unique identifier, e.g. ``"TCK-00042"``.
        category: Ticket category used for stratified splitting.
        instruction: The user-facing ticket text (model input).
        response: The agent resolution text (model target).
        meta: Optional free-form metadata (never fed to the model).
    """

    id: str
    category: str
    instruction: str
    response: str
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSONL/CSV output."""
        d = asdict(self)
        if not d["meta"]:
            d.pop("meta")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        """Build an :class:`Example` from a plain dict, ignoring unknown keys.

        Args:
            d: Mapping with at least ``id``, ``category``, ``instruction``
                and ``response`` keys.

        Returns:
            A new :class:`Example`.

        Raises:
            KeyError: If a required key is missing.
        """
        return cls(
            id=str(d["id"]),
            category=str(d["category"]),
            instruction=str(d["instruction"]),
            response=str(d["response"]),
            meta=dict(d.get("meta", {})),
        )
