from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence


SUPPORTED_OPERATORS = {">", ">=", "<", "<=", "between", "in_top_band"}


def load_policy_document(path: str | Path) -> Dict:
    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_policy_document(
    doc: Dict,
    *,
    expected_task: str,
    allowed_features: Sequence[str],
) -> Dict:
    if not isinstance(doc, dict):
        raise TypeError("Policy document must be a JSON object.")

    task = str(doc.get("task", "")).strip()
    if task != expected_task:
        raise ValueError(f"Policy task mismatch: expected {expected_task}, got {task!r}")

    version = int(doc.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported policy version={version}; expected version=1")

    allowed = set(str(name) for name in allowed_features)
    selected_features = doc.get("selected_features", [])
    if selected_features is None:
        selected_features = []
    if not isinstance(selected_features, list):
        raise TypeError("selected_features must be a list of feature names.")
    for feature_name in selected_features:
        feature_name = str(feature_name)
        if feature_name not in allowed:
            raise ValueError(f"Unknown selected feature {feature_name!r}; allowed={sorted(allowed)}")

    rules_in = doc.get("rules", [])
    if not isinstance(rules_in, list):
        raise TypeError("rules must be a list.")

    normalized_rules: List[Dict] = []
    for idx, rule in enumerate(rules_in):
        if not isinstance(rule, dict):
            raise TypeError(f"Rule at index {idx} must be an object.")
        rule_id = str(rule.get("id", f"rule_{idx}"))
        conditions = rule.get("if")
        normalized_conditions_input = False
        if conditions is None:
            conditions = rule.get("conditions", [])
            normalized_conditions_input = True
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(f"Rule {rule_id} must define a non-empty 'if' list.")
        normalized_conditions: List[Dict] = []
        for cond_idx, cond in enumerate(conditions):
            if normalized_conditions_input:
                if not isinstance(cond, dict):
                    raise ValueError(f"Rule {rule_id} normalized condition #{cond_idx} must be an object.")
                feature_name = str(cond.get("feature", ""))
                operator = str(cond.get("operator", ""))
                value = cond.get("value")
            else:
                if not isinstance(cond, list) or len(cond) != 3:
                    raise ValueError(f"Rule {rule_id} condition #{cond_idx} must be [feature, op, value].")
                feature_name = str(cond[0])
                operator = str(cond[1])
                value = cond[2]
            if feature_name not in allowed:
                raise ValueError(
                    f"Rule {rule_id} references unknown feature {feature_name!r}; allowed={sorted(allowed)}"
                )
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Rule {rule_id} uses unsupported operator {operator!r}; supported={sorted(SUPPORTED_OPERATORS)}"
                )
            if operator == "between":
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(f"Rule {rule_id} between operator expects [lo, hi].")
                value = [float(value[0]), float(value[1])]
            elif operator == "in_top_band":
                value = float(value)
                if value <= 0.0 or value > 1.0:
                    raise ValueError(f"Rule {rule_id} in_top_band value must be in (0, 1].")
            else:
                value = float(value)
            normalized_conditions.append(
                {
                    "feature": feature_name,
                    "operator": operator,
                    "value": value,
                }
            )

        if "then" in rule:
            then_payload = rule.get("then", {})
            if not isinstance(then_payload, dict) or "delta" not in then_payload:
                raise ValueError(f"Rule {rule_id} must provide then.delta.")
            delta = float(then_payload["delta"])
        elif "delta" in rule:
            delta = float(rule["delta"])
        else:
            raise ValueError(f"Rule {rule_id} must provide delta.")
        normalized_rules.append(
            {
                "id": rule_id,
                "intent": str(rule.get("intent", "")).strip(),
                "conditions": normalized_conditions,
                "delta": delta,
            }
        )

    return {
        "task": expected_task,
        "version": 1,
        "selected_features": [str(name) for name in selected_features],
        "rules": normalized_rules,
    }
