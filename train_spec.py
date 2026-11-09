from __future__ import annotations

import copy
from typing import Dict


TASK_ORDER = ["entity_matching", "joinable_table_search", "union_table_search", "schema_matching"]

TASK_PERMUTATIONS = {
    0: TASK_ORDER,
    1: ["joinable_table_search", "schema_matching", "union_table_search", "entity_matching"],
    2: ["schema_matching", "union_table_search", "entity_matching", "joinable_table_search"],
    3: ["union_table_search", "entity_matching", "joinable_table_search", "schema_matching"],
}

PAIR_FEATURE_DEFAULTS = {
    "entity_matching": ["embedding_similarity", "row_value_overlap", "row_profile"],
    "joinable_table_search": [
        "jaccard_containment",
        "value_distribution",
        "overlap_coverage",
        "value_profile",
        "header_similarity",
    ],
    "union_table_search": ["column_overlap", "header_jaccard", "table_size_ratio"],
    "schema_matching": ["header_similarity", "value_stats", "value_overlap"],
}

# Frozen online path defaults: rerank/policy switches are removed from runtime semantics.
GLOBAL_V3C_CONFIG: Dict[str, object] = {
    "eval_threshold_mode": "val_best",
    "eval_fixed_threshold": 0.5,
    "training_online_rerank_enabled": False,
}

# Keep per-task entry points for future threshold overrides only.
TASK_V3C_OVERRIDES: Dict[str, Dict[str, object]] = {
    "entity_matching": {},
    "joinable_table_search": {},
    "union_table_search": {},
    "schema_matching": {},
}


def task_v3c_config(task: str) -> Dict[str, object]:
    if task not in TASK_V3C_OVERRIDES:
        raise ValueError(f"Unknown task={task}")
    merged = copy.deepcopy(GLOBAL_V3C_CONFIG)
    merged.update(copy.deepcopy(TASK_V3C_OVERRIDES[task]))
    return merged
