"""Shared value objects used by the reproduction pipeline."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence


CONDITION_ALIASES = {
    "unconditional": "unconditional",
    "pattern": "pattern",
    "relationnumber": "relation_number",
    "relation_number": "relation_number",
    "entitynumber": "entity_number",
    "entity_number": "entity_number",
    "relation": "specific_relation",
    "specific_relation": "specific_relation",
    "entity": "specific_entity",
    "specific_entity": "specific_entity",
}


def normalize_condition(value: str) -> str:
    try:
        return CONDITION_ALIASES[value]
    except KeyError as exc:
        supported = ", ".join(sorted(CONDITION_ALIASES))
        raise ValueError(f"Unsupported condition {value!r}; expected one of: {supported}") from exc


@dataclass(frozen=True)
class ConditionSpec:
    kind: str
    value: str

    def __post_init__(self) -> None:
        normalized = normalize_condition(self.kind)
        if normalized == "unconditional":
            raise ValueError("Unconditional batches do not carry a ConditionSpec")
        object.__setattr__(self, "kind", normalized)


@dataclass
class PreparedBatch:
    source: Sequence[str]
    target: Sequence[str]
    pattern_id: Any
    input_ids: Any
    attention_mask: Any
    labels: Any
    source_attention_mask: Any
    conditions: Optional[Sequence[ConditionSpec]] = None
