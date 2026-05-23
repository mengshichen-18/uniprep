from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Dict, Sequence

import torch

from compiler import CompiledPolicy, compile_policy
from feature_schema import allowed_feature_names, build_evidence
from rule_schema import load_policy_document, validate_policy_document


logger = logging.getLogger(__name__)


class ResidualReranker:
    def __init__(
        self,
        *,
        task: str,
        pair_feature_order: Sequence[str],
        policy_path: str,
        policy_doc_override: Dict | None = None,
        match_mode: str = "partial",
        hard_guard_patterns: Sequence[str] | None = None,
        min_match_ratio: float = 1.0,
        min_match_ratio_pos: float | None = None,
        min_match_ratio_neg: float | None = None,
        scale_delta_by_match: bool = False,
        shift_s_prior_by_threshold: bool = False,
        policy_relax: float = 0.0,
        rule_combination_mode: str = "accumulate",
        hard_override_eps: float = 1e-3,
        positive_boundary_only: bool = False,
        positive_boundary_margin: float = 0.2,
        negative_boundary_margin: float = 1.0,
    ) -> None:
        self.task = str(task)
        self.policy_path = str(policy_path)
        self.policy_doc_override = copy.deepcopy(policy_doc_override) if policy_doc_override is not None else None
        self.pair_feature_order = [str(name) for name in pair_feature_order]
        self.match_mode = str(match_mode).strip().lower()
        if self.match_mode not in {"strict", "partial", "guarded_partial"}:
            raise ValueError(
                f"Unknown match_mode={match_mode!r}; expected one of: strict,partial,guarded_partial"
            )
        self.hard_guard_patterns = self._normalize_guard_patterns(hard_guard_patterns)
        self.min_match_ratio = float(min_match_ratio)
        self.min_match_ratio_pos = None if min_match_ratio_pos is None else float(min_match_ratio_pos)
        self.min_match_ratio_neg = None if min_match_ratio_neg is None else float(min_match_ratio_neg)
        self.scale_delta_by_match = bool(scale_delta_by_match)
        self.shift_s_prior_by_threshold = bool(shift_s_prior_by_threshold)
        self.policy_relax = float(policy_relax)
        self.rule_combination_mode = str(rule_combination_mode).strip().lower()
        if self.rule_combination_mode not in {"accumulate", "hard_override"}:
            raise ValueError(
                f"Unknown rule_combination_mode={rule_combination_mode!r}; expected one of: accumulate,hard_override"
            )
        self.hard_override_eps = max(0.0, float(hard_override_eps))
        self.positive_boundary_only = bool(positive_boundary_only)
        self.positive_boundary_margin = max(0.0, float(positive_boundary_margin))
        self.negative_boundary_margin = max(0.0, float(negative_boundary_margin))
        self.allowed_features = allowed_feature_names(task=self.task, pair_feature_order=self.pair_feature_order)
        self.policy = self._load_compiled_policy(self.policy_path, policy_doc_override=self.policy_doc_override)
        self.selected_features = (
            list(self.policy.selected_features) if self.policy.selected_features else list(self.allowed_features)
        )

    @staticmethod
    def _normalize_guard_patterns(raw: Sequence[str] | None) -> list[str]:
        patterns: list[str] = []
        for item in raw or []:
            token = str(item).strip()
            if not token or token == "none":
                continue
            patterns.append(token)
        return patterns

    def _is_guard_feature(self, feature_name: str) -> bool:
        name = str(feature_name)
        for pattern in self.hard_guard_patterns:
            if name == pattern or name.startswith(pattern):
                return True
        return False

    def _load_compiled_policy(self, policy_path: str, *, policy_doc_override: Dict | None = None) -> CompiledPolicy:
        doc = copy.deepcopy(policy_doc_override) if policy_doc_override is not None else load_policy_document(policy_path)
        normalized = validate_policy_document(
            doc,
            expected_task=self.task,
            allowed_features=self.allowed_features,
        )
        normalized = self._relax_policy_document(normalized, relax=self.policy_relax)
        policy = compile_policy(normalized)
        logger.info(
            "[Rerank] loaded task=%s policy=%s rules=%d selected_features=%d match_mode=%s guards=%s relax=%.3f",
            self.task,
            "<policy_doc_override>" if policy_doc_override is not None else policy_path,
            len(policy.rules),
            len(policy.selected_features),
            self.match_mode,
            self.hard_guard_patterns,
            float(self.policy_relax),
        )
        return policy

    @property
    def enabled(self) -> bool:
        return len(self.policy.rules) > 0

    @staticmethod
    def _relax_policy_document(doc: Dict, *, relax: float) -> Dict:
        if abs(float(relax)) < 1e-12:
            return doc
        out = copy.deepcopy(doc)
        for rule in out.get("rules", []):
            for cond in rule.get("conditions", []):
                op = str(cond["operator"])
                if op in (">", ">="):
                    cond["value"] = float(cond["value"]) - float(relax)
                elif op in ("<", "<="):
                    cond["value"] = float(cond["value"]) + float(relax)
                elif op == "between":
                    lo = float(cond["value"][0]) - float(relax)
                    hi = float(cond["value"][1]) + float(relax)
                    cond["value"] = [lo, hi]
                elif op == "in_top_band":
                    cond["value"] = min(1.0, float(cond["value"]) + float(relax))
        return out

    def compute_raw_delta(
        self,
        *,
        s_prior: torch.Tensor,
        pair_features: torch.Tensor | None,
        src_degree: torch.Tensor,
        dst_degree: torch.Tensor,
        reference_threshold: float = 0.5,
    ) -> torch.Tensor:
        evidence = build_evidence(
            task=self.task,
            s_prior=s_prior,
            pair_feature_order=self.pair_feature_order,
            pair_features=pair_features,
            src_degree=src_degree,
            dst_degree=dst_degree,
        )
        return self._apply_policy(evidence, reference_threshold=float(reference_threshold))

    def _apply_policy(self, evidence: Dict[str, torch.Tensor], *, reference_threshold: float) -> torch.Tensor:
        if not self.policy.rules:
            sample = next(iter(evidence.values()))
            return torch.zeros_like(sample)

        sample = next(iter(evidence.values()))
        s_prior = evidence.get("s_prior", sample)
        if self.rule_combination_mode == "hard_override":
            final_scores = s_prior.clone()
            for rule in self.policy.rules:
                if not rule.conditions:
                    continue
                condition_masks: list[torch.Tensor] = []
                for condition in rule.conditions:
                    condition_masks.append(
                        self._evaluate_condition(
                            evidence,
                            condition.feature,
                            condition.operator,
                            condition.value,
                            reference_threshold=reference_threshold,
                        )
                    )
                if not condition_masks:
                    continue
                match_count = torch.zeros_like(sample, dtype=torch.int32)
                for cond_mask in condition_masks:
                    match_count = match_count + cond_mask.to(dtype=torch.int32)
                required = self._required_match_count(rule_delta=float(rule.delta), num_conditions=len(rule.conditions))
                mask = self._build_rule_mask(
                    rule=rule,
                    condition_masks=condition_masks,
                    match_count=match_count,
                    required=required,
                    sample=sample,
                )
                mask = self._apply_sign_specific_boundary(
                    mask=mask,
                    rule_delta=float(rule.delta),
                    s_prior=s_prior,
                    reference_threshold=float(reference_threshold),
                )
                if bool(mask.any()):
                    if float(rule.delta) >= 0.0:
                        target = torch.full_like(final_scores, min(1.0, float(reference_threshold) + self.hard_override_eps))
                    else:
                        target = torch.full_like(final_scores, max(0.0, float(reference_threshold) - self.hard_override_eps))
                    final_scores = torch.where(mask, target, final_scores)
            return final_scores - s_prior

        deltas = torch.zeros_like(sample)
        for rule in self.policy.rules:
            if not rule.conditions:
                continue
            condition_masks: list[torch.Tensor] = []
            for condition in rule.conditions:
                condition_masks.append(
                    self._evaluate_condition(
                        evidence,
                        condition.feature,
                        condition.operator,
                        condition.value,
                        reference_threshold=reference_threshold,
                    )
                )
            if not condition_masks:
                continue
            match_count = torch.zeros_like(sample, dtype=torch.int32)
            for cond_mask in condition_masks:
                match_count = match_count + cond_mask.to(dtype=torch.int32)
            required = self._required_match_count(rule_delta=float(rule.delta), num_conditions=len(rule.conditions))
            mask = self._build_rule_mask(
                rule=rule,
                condition_masks=condition_masks,
                match_count=match_count,
                required=required,
                sample=sample,
            )
            mask = self._apply_sign_specific_boundary(
                mask=mask,
                rule_delta=float(rule.delta),
                s_prior=s_prior,
                reference_threshold=float(reference_threshold),
            )
            if bool(mask.any()):
                delta = torch.full_like(sample, float(rule.delta))
                if self.scale_delta_by_match:
                    delta = delta * (match_count.to(dtype=sample.dtype) / float(len(rule.conditions)))
                deltas = deltas + mask.to(dtype=deltas.dtype) * delta
        return deltas

    def _apply_sign_specific_boundary(
        self,
        *,
        mask: torch.Tensor,
        rule_delta: float,
        s_prior: torch.Tensor,
        reference_threshold: float,
    ) -> torch.Tensor:
        if not bool(mask.any()):
            return mask
        if rule_delta >= 0.0:
            if not self.positive_boundary_only:
                return mask
            if self.positive_boundary_margin >= 1.0:
                return mask
            return mask & (torch.abs(s_prior - float(reference_threshold)) <= float(self.positive_boundary_margin))
        if self.negative_boundary_margin >= 1.0:
            return mask
        return mask & (torch.abs(s_prior - float(reference_threshold)) <= float(self.negative_boundary_margin))

    def _build_rule_mask(
        self,
        *,
        rule,
        condition_masks: Sequence[torch.Tensor],
        match_count: torch.Tensor,
        required: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self.match_mode == "strict":
            return match_count >= int(len(rule.conditions))
        if self.match_mode == "partial":
            return match_count >= int(required)
        if self.match_mode != "guarded_partial":
            raise ValueError(f"Unsupported match_mode={self.match_mode!r}")

        guard_indices = [idx for idx, cond in enumerate(rule.conditions) if self._is_guard_feature(cond.feature)]
        if not guard_indices:
            return match_count >= int(required)

        guard_mask = torch.ones_like(sample, dtype=torch.bool)
        for idx in guard_indices:
            guard_mask = guard_mask & condition_masks[idx]

        num_guards = len(guard_indices)
        if num_guards >= len(rule.conditions):
            return guard_mask

        soft_match_count = torch.zeros_like(sample, dtype=torch.int32)
        for idx, cond_mask in enumerate(condition_masks):
            if idx in guard_indices:
                continue
            soft_match_count = soft_match_count + cond_mask.to(dtype=torch.int32)
        soft_required = max(0, int(required) - int(num_guards))
        if soft_required <= 0:
            return guard_mask
        return guard_mask & (soft_match_count >= int(soft_required))

    def _required_match_count(self, *, rule_delta: float, num_conditions: int) -> int:
        ratio = float(self.min_match_ratio)
        if rule_delta >= 0.0 and self.min_match_ratio_pos is not None:
            ratio = float(self.min_match_ratio_pos)
        elif rule_delta < 0.0 and self.min_match_ratio_neg is not None:
            ratio = float(self.min_match_ratio_neg)
        scaled = max(0.0, min(1.0, ratio)) * float(num_conditions)
        required = int(math.ceil(scaled))
        return max(1, min(required, int(num_conditions)))

    def _evaluate_condition(
        self,
        evidence: Dict[str, torch.Tensor],
        feature_name: str,
        operator: str,
        value,
        *,
        reference_threshold: float,
    ) -> torch.Tensor:
        if feature_name not in evidence:
            raise KeyError(f"Missing evidence feature {feature_name!r} for task {self.task}")
        feats = evidence[feature_name]
        value = self._normalize_condition_value(
            feature_name=feature_name,
            operator=operator,
            value=value,
            reference_threshold=reference_threshold,
        )
        if operator == ">":
            return feats > float(value)
        if operator == ">=":
            return feats >= float(value)
        if operator == "<":
            return feats < float(value)
        if operator == "<=":
            return feats <= float(value)
        if operator == "between":
            lo = float(value[0])
            hi = float(value[1])
            return (feats >= lo) & (feats <= hi)
        if operator == "in_top_band":
            frac = float(value)
            if feats.numel() == 0:
                return torch.zeros_like(feats, dtype=torch.bool)
            cutoff = torch.quantile(feats.detach().float(), max(0.0, 1.0 - frac))
            return feats >= cutoff.to(device=feats.device, dtype=feats.dtype)
        raise ValueError(f"Unsupported operator {operator!r}")

    def _normalize_condition_value(
        self,
        *,
        feature_name: str,
        operator: str,
        value,
        reference_threshold: float,
    ):
        if not self.shift_s_prior_by_threshold or feature_name != "s_prior" or operator == "in_top_band":
            return value
        offset = float(reference_threshold) - 0.5
        if abs(offset) < 1e-12:
            return value
        if operator == "between":
            lo = max(0.0, min(1.0, float(value[0]) + offset))
            hi = max(0.0, min(1.0, float(value[1]) + offset))
            return [min(lo, hi), max(lo, hi)]
        return max(0.0, min(1.0, float(value) + offset))


def default_policy_path(*, repo_root: str, task: str) -> str:
    mapping = {
        "entity_matching": "entity_matching_oracle_v1.json",
        "joinable_table_search": "joinable_table_search_oracle_v1.json",
        "union_table_search": "union_table_search_oracle_v1.json",
        "schema_matching": "schema_matching_oracle_v1.json",
    }
    if task not in mapping:
        raise ValueError(f"Unknown task={task}")
    return str(Path(repo_root) / "0325_policy" / "policy_batches" / "policy_oracle_v1" / mapping[task])
