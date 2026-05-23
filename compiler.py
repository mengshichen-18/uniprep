from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CompiledCondition:
    feature: str
    operator: str
    value: float | List[float]


@dataclass(frozen=True)
class CompiledRule:
    rule_id: str
    intent: str
    delta: float
    conditions: List[CompiledCondition]


@dataclass(frozen=True)
class CompiledPolicy:
    task: str
    version: int
    selected_features: List[str]
    rules: List[CompiledRule]


def compile_policy(doc: Dict) -> CompiledPolicy:
    rules: List[CompiledRule] = []
    for rule in doc.get("rules", []):
        conditions = [
            CompiledCondition(
                feature=str(cond["feature"]),
                operator=str(cond["operator"]),
                value=cond["value"],
            )
            for cond in rule.get("conditions", [])
        ]
        rules.append(
            CompiledRule(
                rule_id=str(rule["id"]),
                intent=str(rule.get("intent", "")),
                delta=float(rule["delta"]),
                conditions=conditions,
            )
        )
    return CompiledPolicy(
        task=str(doc["task"]),
        version=int(doc["version"]),
        selected_features=[str(name) for name in doc.get("selected_features", [])],
        rules=rules,
    )
