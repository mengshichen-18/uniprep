from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch


_BASE_FEATURES: Sequence[str] = (
    "s_prior",
    "uncertainty",
    "src_degree",
    "dst_degree",
    "degree_ratio",
)


_TASK_DERIVED_FEATURES = {
    "union_table_search": {
        "overlap_mean_max": ("col_overlap_a2b_mean", "col_overlap_b2a_mean"),
        "overlap_cov_max": ("col_overlap_a2b_cov", "col_overlap_b2a_cov"),
        "overlap_direction_gap": ("col_overlap_a2b_mean", "col_overlap_b2a_mean"),
        "size_balance": ("col_count_ratio", "row_count_ratio"),
    },
    "joinable_table_search": {
        "coverage_max": ("coverage_a", "coverage_b"),
    },
}


def allowed_feature_names(task: str, pair_feature_order: Sequence[str]) -> List[str]:
    names: List[str] = list(_BASE_FEATURES)
    for name in pair_feature_order:
        if name not in names:
            names.append(str(name))
    for derived_name, required in _TASK_DERIVED_FEATURES.get(task, {}).items():
        if all(req in names for req in required):
            names.append(derived_name)
    return names


def build_evidence(
    *,
    task: str,
    s_prior: torch.Tensor,
    pair_feature_order: Sequence[str],
    pair_features: torch.Tensor | None,
    src_degree: torch.Tensor,
    dst_degree: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if s_prior.dim() != 1:
        s_prior = s_prior.view(-1)
    dtype = s_prior.dtype
    device = s_prior.device
    batch_size = int(s_prior.numel())

    evidence: Dict[str, torch.Tensor] = {
        "s_prior": s_prior,
        "uncertainty": 1.0 - torch.abs(s_prior - 0.5) * 2.0,
        "src_degree": src_degree.to(device=device, dtype=dtype).view(-1),
        "dst_degree": dst_degree.to(device=device, dtype=dtype).view(-1),
    }

    degree_den = torch.maximum(evidence["src_degree"], evidence["dst_degree"])
    degree_num = torch.minimum(evidence["src_degree"], evidence["dst_degree"])
    evidence["degree_ratio"] = torch.where(
        degree_den > 0,
        degree_num / degree_den,
        torch.zeros((batch_size,), dtype=dtype, device=device),
    )

    if pair_features is None:
        pair_features = torch.zeros((batch_size, len(pair_feature_order)), dtype=dtype, device=device)
    else:
        pair_features = pair_features.to(device=device, dtype=dtype)

    for idx, name in enumerate(pair_feature_order):
        if idx >= pair_features.size(1):
            evidence[str(name)] = torch.zeros((batch_size,), dtype=dtype, device=device)
        else:
            evidence[str(name)] = pair_features[:, idx].view(-1)

    _add_task_derived_features(task=task, evidence=evidence)
    return evidence


def _add_task_derived_features(*, task: str, evidence: Dict[str, torch.Tensor]) -> None:
    if task == "union_table_search":
        _add_if_present(evidence, "overlap_mean_max", _max_two, "col_overlap_a2b_mean", "col_overlap_b2a_mean")
        _add_if_present(evidence, "overlap_cov_max", _max_two, "col_overlap_a2b_cov", "col_overlap_b2a_cov")
        _add_if_present(
            evidence,
            "overlap_direction_gap",
            lambda a, b: torch.abs(a - b),
            "col_overlap_a2b_mean",
            "col_overlap_b2a_mean",
        )
        _add_if_present(evidence, "size_balance", _min_two, "col_count_ratio", "row_count_ratio")
    elif task == "joinable_table_search":
        _add_if_present(evidence, "coverage_max", _max_two, "coverage_a", "coverage_b")


def _add_if_present(
    evidence: Dict[str, torch.Tensor],
    out_name: str,
    fn,
    lhs_name: str,
    rhs_name: str,
) -> None:
    if lhs_name not in evidence or rhs_name not in evidence:
        return
    evidence[out_name] = fn(evidence[lhs_name], evidence[rhs_name])


def _max_two(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.maximum(lhs, rhs)


def _min_two(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.minimum(lhs, rhs)


def feature_names_for_policy(
    *,
    task: str,
    pair_feature_order: Sequence[str],
    selected_features: Iterable[str] | None = None,
) -> List[str]:
    allowed = allowed_feature_names(task=task, pair_feature_order=pair_feature_order)
    if selected_features is None:
        return allowed
    keep = []
    selected = {str(name) for name in selected_features}
    for name in allowed:
        if name in selected:
            keep.append(name)
    return keep
