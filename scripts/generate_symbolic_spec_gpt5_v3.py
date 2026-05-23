#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from symbolic_feature import (  # noqa: E402
    allowed_symbolic_function_names,
    extract_json_object,
    save_symbolic_feature_spec,
    validate_symbolic_feature_spec,
)


TASK_HINTS = {
    "entity_matching": (
        "Use compact but minimally lossy channels. Keep major evidence mechanisms distinguishable "
        "(serialization, overlap/containment, semantic similarity, numeric consistency, profile). "
        "Prefer single-atom or light same-family channels by default, and avoid over-fusing unrelated "
        "families into one channel. Use symbolic transforms when they improve stability or calibration, "
        "but keep expressions short and interpretable."
    ),
    "joinable_table_search": "Column-pair joinability: containment/coverage/value consistency are important.",
    "union_table_search": "Table-pair unionability: bidirectional overlap and size compatibility matter.",
    "schema_matching": "Column semantic alignment: header/value consistency and stability are important.",
}

TASK_GROUP_TOKEN_CATALOG = {
    "entity_matching": [
        "embedding_similarity",
        "row_value_overlap",
        "row_profile",
        "serial_value_alignment",
        "serial_lexical_plus",
        "header_alignment",
    ],
    "joinable_table_search": [
        "jaccard_containment",
        "value_distribution",
        "overlap_coverage",
        "value_profile",
        "header_similarity",
    ],
    "union_table_search": [
        "column_overlap",
        "header_jaccard",
        "table_size_ratio",
    ],
    "schema_matching": [
        "header_similarity",
        "value_stats",
        "value_overlap",
    ],
}

DEFAULT_ATOM_DICTIONARY_PATHS: list = []
DEFAULT_API_KEY_FILE = _ROOT.parent / "0325_policy_pro" / "LIGHTNING_API_KEY.md"

TASK_ROLE_EXAMPLES = {
    "entity_matching": [
        "serial_lexical",
        "value_overlap",
        "profile_consistency",
        "numeric_alignment",
        "robustness",
        "boundary_guard",
        "calibration",
        "hard_negative_guard",
    ],
    "joinable_table_search": [
        "containment_core",
        "coverage_balance",
        "coverage_direction",
        "distribution_profile",
        "header_semantic",
        "robustness",
        "noise_guard",
        "calibration",
    ],
    "union_table_search": [
        "overlap_a2b",
        "overlap_b2a",
        "overlap_coverage",
        "header_alignment",
        "size_compatibility",
        "robustness",
        "noise_guard",
        "calibration",
    ],
    "schema_matching": [
        "header_semantic",
        "value_overlap",
        "value_containment",
        "profile_stats",
        "missingness_guard",
        "numeric_consistency",
        "robustness",
        "calibration",
    ],
}

GROUP_TOKEN_EXPANSIONS = {
    "embedding_similarity": ["row_emb_cosine", "row_emb_l1_sim"],
    "row_value_overlap": ["row_value_jaccard", "row_value_containment_max", "row_token_jaccard"],
    "row_profile": ["row_nonempty_ratio", "row_numeric_ratio_sim", "row_avg_len_ratio"],
    "serial_value_alignment": ["row_serial_token_jaccard", "row_serial_edit_similarity", "row_numeric_value_overlap"],
    # Backward-compat alias.
    "ditto_proxy": ["row_serial_token_jaccard", "row_serial_edit_similarity", "row_numeric_value_overlap"],
    "serial_lexical_plus": [
        "row_serial_char3_jaccard",
        "row_serial_char4_jaccard",
        "row_token_idf_jaccard",
        "row_numeric_rel_diff_sim",
    ],
    "header_alignment": ["row_header_jaccard", "row_header_value_exact_ratio", "row_header_token_jaccard"],
    "jaccard_containment": ["jaccard", "containment_max"],
    "value_distribution": ["value_distribution"],
    "overlap_coverage": ["coverage_a", "coverage_b", "coverage_max"],
    "value_profile": ["unique_ratio_sim", "numeric_ratio_sim", "avg_len_ratio_sim"],
    "header_similarity": ["header_token_jaccard", "header_edit_similarity"],
    "value_stats": ["unique_ratio_sim", "missing_ratio_sim", "numeric_ratio_sim", "avg_len_ratio_sim"],
    "value_overlap": ["value_jaccard", "value_containment_max"],
    "column_overlap": ["col_overlap_a2b_mean", "col_overlap_b2a_mean", "col_overlap_a2b_cov", "col_overlap_b2a_cov"],
    "header_jaccard": ["header_jaccard"],
    "table_size_ratio": ["col_count_ratio", "row_count_ratio"],
}


def _default_passthrough_ratio() -> float:
    return 0.0


def _normalize_ratio(value: float, *, name: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {ratio}")
    return ratio


def _compute_min_passthrough_channels(*, num_channels: int, ratio: float) -> int:
    if int(num_channels) <= 0:
        return 0
    r = _normalize_ratio(float(ratio), name="passthrough_ratio")
    if r <= 0.0:
        return 0
    return max(1, int(math.ceil(float(num_channels) * r)))

# Optional explicit evidence-family hints by token/atom name.
FAMILY_OVERRIDES = {
    "embedding_similarity": "semantic_embedding",
    "row_emb_cosine": "semantic_embedding",
    "row_emb_l1_sim": "semantic_embedding",
    "row_value_overlap": "set_overlap_coverage",
    "row_value_jaccard": "set_overlap_coverage",
    "row_value_containment_max": "set_overlap_coverage",
    "row_token_jaccard": "set_overlap_coverage",
    "row_profile": "profile_statistics",
    "row_nonempty_ratio": "profile_statistics",
    "row_numeric_ratio_sim": "profile_statistics",
    "row_avg_len_ratio": "profile_statistics",
    "serial_value_alignment": "lexical_serialization",
    "ditto_proxy": "lexical_serialization",
    "row_serial_token_jaccard": "lexical_serialization",
    "row_serial_edit_similarity": "lexical_serialization",
    "row_numeric_value_overlap": "lexical_serialization",
    "serial_lexical_plus": "lexical_serialization",
    "row_serial_char3_jaccard": "lexical_serialization",
    "row_serial_char4_jaccard": "lexical_serialization",
    "row_token_idf_jaccard": "lexical_serialization",
    "row_numeric_rel_diff_sim": "lexical_serialization",
    "header_alignment": "schema_header",
    "row_header_jaccard": "schema_header",
    "row_header_value_exact_ratio": "schema_header",
    "row_header_token_jaccard": "schema_header",
    "jaccard_containment": "set_overlap_coverage",
    "jaccard": "set_overlap_coverage",
    "containment_max": "set_overlap_coverage",
    "overlap_coverage": "set_overlap_coverage",
    "coverage_a": "set_overlap_coverage",
    "coverage_b": "set_overlap_coverage",
    "coverage_max": "set_overlap_coverage",
    "value_distribution": "distributional_profile",
    "value_profile": "distributional_profile",
    "unique_ratio_sim": "distributional_profile",
    "numeric_ratio_sim": "distributional_profile",
    "avg_len_ratio_sim": "distributional_profile",
    "header_similarity": "schema_header",
    "header_token_jaccard": "schema_header",
    "header_edit_similarity": "schema_header",
    "value_stats": "distributional_profile",
    "missing_ratio_sim": "distributional_profile",
    "value_overlap": "set_overlap_coverage",
    "value_jaccard": "set_overlap_coverage",
    "value_containment_max": "set_overlap_coverage",
    "column_overlap": "set_overlap_coverage",
    "col_overlap_a2b_mean": "set_overlap_coverage",
    "col_overlap_b2a_mean": "set_overlap_coverage",
    "col_overlap_a2b_cov": "set_overlap_coverage",
    "col_overlap_b2a_cov": "set_overlap_coverage",
    "header_jaccard": "schema_header",
    "table_size_ratio": "size_scale",
    "col_count_ratio": "size_scale",
    "row_count_ratio": "size_scale",
}


def _default_feature_cards_path() -> str:
    return str((_ROOT / "symbolic_feature_cards.json").resolve())


def _model_candidates(model_name: str) -> List[str]:
    name = str(model_name).strip()
    if not name:
        return ["gpt-5-mini", "openai/gpt-5-mini", "gpt-5", "openai/gpt-5"]
    if "/" in name:
        return [name]
    return [name, f"openai/{name}"]


class LLMCallFailure(RuntimeError):
    def __init__(self, message: str, call_records: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.call_records: List[Dict[str, Any]] = [dict(item) for item in call_records]


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _extract_usage_fields(resp: Any) -> Dict[str, Optional[int]]:
    usage_obj = getattr(resp, "usage", None)
    if usage_obj is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    usage_payload: Dict[str, Any] = {}
    if hasattr(usage_obj, "model_dump"):
        try:
            usage_payload = usage_obj.model_dump()
        except Exception:  # noqa: BLE001
            usage_payload = {}
    elif isinstance(usage_obj, dict):
        usage_payload = dict(usage_obj)

    prompt_tokens = _safe_int(
        usage_payload.get("prompt_tokens", getattr(usage_obj, "prompt_tokens", None))
    )
    completion_tokens = _safe_int(
        usage_payload.get("completion_tokens", getattr(usage_obj, "completion_tokens", None))
    )
    total_tokens = _safe_int(
        usage_payload.get("total_tokens", getattr(usage_obj, "total_tokens", None))
    )

    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = int((prompt_tokens or 0) + (completion_tokens or 0))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _parse_csv(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _discover_api_key_from_text(text: str, *, label: str = "") -> str:
    raw = str(text)
    key_label = str(label).strip()
    if key_label:
        patterns = [
            rf"{re.escape(key_label)}\s*[:=]\s*(\S+)",
            rf"{re.escape(key_label)}\s+(\S+)",
        ]
        for pat in patterns:
            match = re.search(pat, raw)
            if match:
                return str(match.group(1)).strip().strip("'\"")
    generic = re.findall(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{15,}", raw)
    return str(generic[-1]).strip() if generic else ""


def _discover_api_key(path: Path, *, label: str = "") -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _discover_api_key_from_text(text, label=label)


def _task_role_examples(task: str, *, num_channels: int) -> List[str]:
    base = list(TASK_ROLE_EXAMPLES.get(str(task), []))
    if not base:
        base = ["signal_core", "supporting_evidence", "robustness", "calibration"]
    n = max(1, int(num_channels))
    out = list(base[:n])
    idx = 1
    while len(out) < n:
        out.append(f"role_{idx:02d}")
        idx += 1
    return out


def _resolve_channel_roles(
    *,
    task: str,
    num_channels: int,
    provided_roles: Sequence[str],
) -> List[str]:
    roles = [str(x).strip() for x in provided_roles if str(x).strip()]
    if roles:
        if len(roles) != int(num_channels):
            raise ValueError(
                f"len(channel_roles) must equal num_channels ({num_channels}), got {len(roles)}"
            )
        role_set = set()
        for role in roles:
            token = role.lower()
            if token in role_set:
                raise ValueError(f"Duplicate channel role: {role!r}")
            role_set.add(token)
        return roles
    return _task_role_examples(str(task), num_channels=int(num_channels))


def _load_task_hint_strong_atoms(
    *,
    atom_dict_paths: Sequence[str],
    split: str,
    top_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    stats: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {}
    split_key = str(split).strip().lower() or "test"
    if split_key not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported strong-atom split: {split!r}")
    for path in atom_dict_paths:
        p = Path(str(path).strip()).expanduser()
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tasks = doc.get("tasks", {})
        if not isinstance(tasks, Mapping):
            continue
        for task_name, task_doc in tasks.items():
            if not isinstance(task_doc, Mapping):
                continue
            atoms = task_doc.get("atoms", {})
            if not isinstance(atoms, Mapping):
                continue
            task_bucket = stats.setdefault(str(task_name), {})
            for atom_name, atom_doc in atoms.items():
                if not isinstance(atom_doc, Mapping):
                    continue
                split_doc = atom_doc.get(split_key, {})
                if not isinstance(split_doc, Mapping):
                    continue
                auc = split_doc.get("auc", None)
                best_f1 = split_doc.get("best_f1", None)
                missing_rate = split_doc.get("missing_rate", None)
                try:
                    auc_f = float(auc)
                except Exception:
                    continue
                try:
                    f1_f = float(best_f1) if best_f1 is not None else 0.0
                except Exception:
                    f1_f = 0.0
                try:
                    miss_f = float(missing_rate) if missing_rate is not None else 0.0
                except Exception:
                    miss_f = 0.0
                task_bucket.setdefault(str(atom_name), []).append((auc_f, f1_f, miss_f))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for task_name, atom_bucket in stats.items():
        rows: List[Dict[str, Any]] = []
        for atom_name, vals in atom_bucket.items():
            if not vals:
                continue
            auc_mean = float(sum(v[0] for v in vals) / len(vals))
            f1_mean = float(sum(v[1] for v in vals) / len(vals))
            miss_mean = float(sum(v[2] for v in vals) / len(vals))
            rows.append(
                {
                    "atom": str(atom_name),
                    "mean_auc": auc_mean,
                    "mean_best_f1": f1_mean,
                    "mean_missing_rate": miss_mean,
                    "n_datasets": int(len(vals)),
                }
            )
        rows.sort(
            key=lambda x: (
                -float(x.get("mean_auc", -1.0)),
                -float(x.get("mean_best_f1", -1.0)),
                float(x.get("mean_missing_rate", 0.0)),
                str(x.get("atom", "")),
            )
        )
        out[str(task_name)] = rows[: max(0, int(top_k))]
    return out


def _soft_prior_summary_from_rows(
    *,
    strong_atom_rows: Sequence[Mapping[str, Any]],
    max_atoms: int = 6,
    max_families: int = 4,
) -> Dict[str, List[str]]:
    rows = list(strong_atom_rows or [])
    atoms: List[str] = []
    atom_seen: Set[str] = set()
    families: List[str] = []
    fam_seen: Set[str] = set()
    for item in rows:
        atom = str(item.get("atom", "")).strip()
        if atom and atom not in atom_seen:
            atom_seen.add(atom)
            atoms.append(atom)
        key = atom.lower()
        family = "general_similarity"
        if any(t in key for t in ("serial", "token", "edit", "lexical")):
            family = "lexical_serialization"
        elif any(t in key for t in ("jaccard", "containment", "overlap", "coverage")):
            family = "set_overlap_coverage"
        elif any(t in key for t in ("emb", "embedding")):
            family = "semantic_embedding"
        elif any(t in key for t in ("profile", "distribution", "ratio", "stats")):
            family = "distributional_profile"
        if family not in fam_seen:
            fam_seen.add(family)
            families.append(family)
        if len(atoms) >= int(max_atoms) and len(families) >= int(max_families):
            break
    return {
        "atoms": atoms[: max(0, int(max_atoms))],
        "families": families[: max(0, int(max_families))],
    }


def _render_task_hint(
    *,
    task: str,
    strong_atom_rows: Sequence[Mapping[str, Any]],
) -> str:
    base = TASK_HINTS.get(task, "General binary matching task.")
    if str(task).strip() == "entity_matching":
        return base
    prior = _soft_prior_summary_from_rows(strong_atom_rows=strong_atom_rows)
    atoms = list(prior.get("atoms", []))
    families = list(prior.get("families", []))
    if not atoms and not families:
        return base
    parts: List[str] = [base]
    if families:
        parts.append(
            "Prioritize robust evidence diversity across families: "
            + ", ".join(families)
            + "."
        )
    # For EM keep prior guidance implicit (family-level only), avoid explicit atom list in task hint.
    if atoms and str(task).strip() != "entity_matching":
        parts.append(
            "Representative high-value atoms (soft prior, no hard coverage): "
            + ", ".join(atoms)
            + "."
        )
    return " ".join(parts)


def _load_feature_cards(path: str) -> Dict[str, Dict[str, str]]:
    cards_path = Path(path).expanduser()
    if not cards_path.exists():
        return {}
    raw = json.loads(cards_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        return {}
    features = raw.get("features", {})
    if not isinstance(features, Mapping):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for name, payload in features.items():
        if not isinstance(payload, Mapping):
            continue
        key = str(name).strip()
        if not key:
            continue
        out[key] = {
            "definition": str(payload.get("definition", "")).strip(),
            "formula": str(payload.get("formula", "")).strip(),
            "range": str(payload.get("range", "")).strip(),
            "caution": str(payload.get("caution", "")).strip(),
            "family": str(payload.get("family", "")).strip(),
        }
    return out


def _infer_feature_family(name: str, meta: Mapping[str, str]) -> str:
    explicit = str(meta.get("family", "")).strip().lower()
    if explicit:
        return explicit
    key = str(name).strip().lower()
    if key in FAMILY_OVERRIDES:
        return str(FAMILY_OVERRIDES[key])
    if "header" in key:
        return "schema_header"
    if any(t in key for t in ("jaccard", "containment", "coverage", "overlap")):
        return "set_overlap_coverage"
    if any(t in key for t in ("emb", "embedding")):
        return "semantic_embedding"
    if any(t in key for t in ("profile", "distribution", "ratio", "stats")):
        return "distributional_profile"
    if any(t in key for t in ("serial", "token", "edit", "lexical")):
        return "lexical_serialization"
    if any(t in key for t in ("count", "size")):
        return "size_scale"
    return "general_similarity"


def _is_group_card(meta: Mapping[str, str]) -> bool:
    range_hint = str(meta.get("range", "")).strip().lower()
    formula = str(meta.get("formula", "")).strip().lower()
    if "group token" in range_hint:
        return True
    if formula.startswith("expands_to="):
        return True
    return False


def _extract_expands_to(meta: Mapping[str, str]) -> List[str]:
    formula = str(meta.get("formula", "")).strip()
    m = re.match(r"^expands_to=\[(.*)\]$", formula)
    if not m:
        return []
    body = str(m.group(1)).strip()
    if not body:
        return []
    return [x.strip() for x in body.split(",") if x.strip()]


def _detect_group_tokens(
    *,
    feature_pool: Sequence[str],
    cards: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for token in feature_pool:
        if token in GROUP_TOKEN_EXPANSIONS:
            expands = [str(x).strip() for x in list(GROUP_TOKEN_EXPANSIONS.get(token, [])) if str(x).strip()]
            # Treat self-expanding aliases as atom-equivalent (e.g., header_jaccard -> [header_jaccard]).
            if len(expands) == 1 and expands[0] == str(token):
                continue
            out.append(
                {
                    "token": str(token),
                    "expands_to": expands,
                }
            )
            continue
        meta = cards.get(str(token), {})
        if not meta:
            continue
        if not _is_group_card(meta):
            continue
        out.append(
            {
                "token": str(token),
                "expands_to": _extract_expands_to(meta),
            }
        )
    return out


def _build_group_token_block(
    *,
    group_tokens: Sequence[Mapping[str, Any]],
    max_items: int,
) -> str:
    if int(max_items) <= 0:
        return ""
    if not group_tokens:
        return ""
    lines: List[str] = []
    lines.append("Group-token hints:")
    lines.append(
        "- The following features are group tokens (shorthand bundles), not single atom features:"
    )
    for item in list(group_tokens)[: int(max_items)]:
        token = str(item.get("token", "")).strip()
        if not token:
            continue
        expands = item.get("expands_to", [])
        if isinstance(expands, list) and expands:
            lines.append(f"- {token}: expands_to=[{', '.join(str(x) for x in expands)}]")
        else:
            lines.append(f"- {token}: group token (expansion not provided in cards)")
    lines.append("- Prefer explicit atom features when fine-grained control is needed.")
    lines.append("- Atoms expanded from the same group are usually correlated/redundant evidence.")
    lines.append("- Avoid counting many same-group atoms as independent strong signals unless justified.")
    return "\n".join(lines) + "\n"


def _build_task_group_catalog_block(
    *,
    task: str,
    feature_pool: Sequence[str],
    cards: Mapping[str, Mapping[str, str]],
    max_items: int,
) -> str:
    tokens = list(TASK_GROUP_TOKEN_CATALOG.get(str(task), []))
    if not tokens:
        return ""
    if int(max_items) <= 0:
        return ""
    lines: List[str] = []
    lines.append(f"Task-level group catalog ({task}):")
    lines.append("- Tokens below are group families; atoms in the same family are correlated/redundant.")
    lines.append("- Status tag: [in_pool] means token is in current Feature Pool; [catalog_only] is reference only.")
    pool = set(str(x) for x in feature_pool)
    shown = 0
    for token in tokens:
        if shown >= int(max_items):
            break
        expands = list(GROUP_TOKEN_EXPANSIONS.get(str(token), []))
        if not expands:
            meta = cards.get(str(token), {})
            expands = _extract_expands_to(meta) if meta else []
        status = "in_pool" if token in pool else "catalog_only"
        if expands:
            lines.append(f"- [{status}] {token}: expands_to=[{', '.join(expands)}]")
        else:
            lines.append(f"- [{status}] {token}: group token")
        shown += 1
    lines.append("- Prefer cross-group diversity; do not over-count same-group atoms as independent evidence.")
    return "\n".join(lines) + "\n"


def _build_feature_cards_block(
    *,
    feature_pool: Sequence[str],
    cards: Mapping[str, Mapping[str, str]],
    max_cards: int,
    compact: bool = False,
) -> str:
    lines: List[str] = []
    if int(max_cards) <= 0:
        return ""
    for name in feature_pool[: int(max_cards)]:
        key = str(name)
        meta = cards.get(key, {})
        if meta and _is_group_card(meta):
            # Skip group tokens in atom-card section; they are handled in a dedicated block.
            continue
        definition = str(meta.get("definition", "")).strip() or "No curated definition provided."
        formula = str(meta.get("formula", "")).strip() or "Use as provided in feature pool."
        range_hint = str(meta.get("range", "")).strip() or "Unknown."
        caution = str(meta.get("caution", "")).strip() or "Treat uncertain behavior conservatively."
        family = _infer_feature_family(key, meta)
        if compact:
            lines.append(f"- {key}: family={family}; formula={formula}; range={range_hint}")
        else:
            lines.append(f"- {key}")
            lines.append(f"  family: {family}")
            lines.append(f"  definition: {definition}")
            lines.append(f"  formula: {formula}")
            lines.append(f"  range: {range_hint}")
            lines.append(f"  caution: {caution}")
    if not lines:
        return ""
    return "Feature cards:\n" + "\n".join(lines) + "\n"


def _load_feature_pool(args: argparse.Namespace) -> List[str]:
    names: List[str] = []

    if str(args.feature_pool).strip():
        names.extend(_parse_csv(args.feature_pool))

    if str(args.feature_pool_file).strip():
        fp = Path(args.feature_pool_file)
        if not fp.exists():
            raise FileNotFoundError(f"feature pool file not found: {fp}")
        raw = fp.read_text(encoding="utf-8").strip()
        if fp.suffix.lower() == ".json":
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                values = parsed.get("feature_pool", parsed.get("features", []))
            else:
                values = parsed
            if not isinstance(values, list):
                raise ValueError("feature_pool_file json must contain a list or {feature_pool:[...]}.")
            names.extend([str(x).strip() for x in values if str(x).strip()])
        else:
            names.extend(_parse_csv(raw.replace("\n", ",")))

    if str(args.replay_npz).strip():
        z = np.load(args.replay_npz, allow_pickle=True)
        if "pair_feature_order" not in z:
            raise KeyError(f"replay npz missing pair_feature_order: {args.replay_npz}")
        names.extend([str(x) for x in np.asarray(z["pair_feature_order"]).tolist()])

        include_extra = str(args.include_extra_features).strip().lower()
        if include_extra in {"degree", "all"}:
            names.extend(["src_degree", "dst_degree", "degree_ratio"])
        if include_extra in {"gnn", "all"}:
            names.extend(["gnn_score", "uncertainty"])

    dedup: List[str] = []
    seen = set()
    for name in names:
        key = str(name).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    return dedup


def _load_dataset_context(path: str) -> Dict[str, Any]:
    token = str(path).strip()
    if not token:
        return {}
    context_path = Path(token).expanduser()
    if not context_path.exists():
        raise FileNotFoundError(f"dataset context file not found: {context_path}")
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dataset context file must contain a JSON object")
    return dict(payload)


def _build_dataset_context_block(context: Mapping[str, Any], *, max_atoms: int) -> str:
    if not context:
        return ""
    dataset = str(context.get("dataset_name", "")).strip()
    task = str(context.get("task", "")).strip()
    split = str(context.get("split", "train")).strip() or "train"
    source_file = str(context.get("source_file", "")).strip()
    atoms = context.get("selected_atoms", [])
    if not isinstance(atoms, list):
        atoms = []

    lines: List[str] = []
    lines.append("Dataset-aware context (use as guidance only):")
    if dataset:
        lines.append(f"- dataset: {dataset}")
    if task:
        lines.append(f"- task: {task}")
    lines.append(f"- split: {split} (train-only stats)")
    if source_file:
        lines.append(f"- source: {source_file}")

    used = 0
    for item in atoms:
        if used >= int(max_atoms):
            break
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        auc = item.get("auc", None)
        best_f1 = item.get("best_f1", None)
        best_thr = item.get("best_f1_threshold", None)
        miss = item.get("missing_rate", None)
        p50 = item.get("p50", None)
        p95 = item.get("p95", None)
        lines.append(
            "- "
            f"{name}: auc={auc}, best_f1={best_f1}, best_thr={best_thr}, "
            f"missing={miss}, p50={p50}, p95={p95}"
        )
        used += 1

    if used <= 0:
        lines.append("- no usable atom stats in context payload")
    return "\n".join(lines) + "\n"


def _build_prompt(
    *,
    task: str,
    feature_pool: Sequence[str],
    objective: str,
    operator_examples: Sequence[str],
    feature_cards_block: str,
    task_group_catalog_block: str,
    group_token_block: str,
    dataset_context_block: str,
    num_channels_exact: int,
    min_num_channels: int,
    channel_roles: Sequence[str],
    role_examples: Sequence[str],
    rendered_task_hint: str,
    enable_coverage_constraint: bool,
    enable_single_atom_hint: bool,
    min_passthrough_channels: int,
) -> Dict[str, str]:
    task_hint = str(rendered_task_hint).strip() or TASK_HINTS.get(task, "General binary matching task.")
    is_em_task = str(task).strip() == "entity_matching"
    system_prompt = (
        "You design symbolic decision algorithms for tabular entity linkage tasks. "
        "Return JSON only. Do not include Markdown fences or any extra text."
    )

    if is_em_task:
        base_header = (
            f"Task: {task}\n"
            f"Objective: {objective}\n"
            "Constraint: Use only provided feature names. Build symbolic scoring logic only "
            "(no neural model or external calls).\n"
            "\n"
            f"Task Hint:\n{task_hint}\n"
            "\n"
            f"Operator examples: {', '.join(operator_examples)}\n"
            f"\nFeature Pool ({len(feature_pool)}): {json.dumps(list(feature_pool), ensure_ascii=False)}\n"
        )
    else:
        base_header = (
            f"Task: {task}\n"
            f"Task Hint: {task_hint}\n"
            f"Objective: {objective}\n"
            "Constraint: Use only provided feature names. Build symbolic scoring logic only "
            "(no neural model or external calls).\n"
            "Operator examples (not fixed templates; free combination allowed): "
            f"{', '.join(operator_examples)}\n"
            f"Feature Pool ({len(feature_pool)}): {json.dumps(list(feature_pool), ensure_ascii=False)}\n"
        )

    exact_count = int(num_channels_exact)
    min_count = int(min_num_channels)
    if exact_count <= 0 and min_count <= 0:
        raise ValueError("num_channels_exact or min_num_channels must be > 0 for v2")
    role_json = json.dumps(list(channel_roles), ensure_ascii=False) if channel_roles else "[]"
    role_examples_json = json.dumps(list(role_examples), ensure_ascii=False)
    if is_em_task:
        schema = (
            "Required JSON schema:\n"
            "{\n"
            "  \"spec_version\": \"v2\",\n"
            f"  \"task\": \"{task}\",\n"
            "  \"feature_pool_used\": [\"...\"],\n"
            "  \"channels\": [\n"
            "    {\n"
            "      \"name\": \"...\",\n"
            "      \"role\": \"...\",\n"
            "      \"expression\": \"...\",\n"
            "      \"rationale\": \"short rationale\"\n"
            "    }\n"
            "  ],\n"
            "  \"aggregation\": {\n"
            "    \"method\": \"weighted_sum\",\n"
            "    \"weights\": [0.25, 0.25, 0.25, 0.25],\n"
            "    \"bias\": 0.0,\n"
            "    \"postprocess\": \"sigmoid\"\n"
            "  },\n"
            "  \"output_name\": \"sym_feature_v2\",\n"
            "  \"notes\": \"short rationale\"\n"
            "}\n"
        )
    else:
        schema = (
            "Required JSON schema:\n"
            "{\n"
            "  \"spec_version\": \"v2\",\n"
            f"  \"task\": \"{task}\",\n"
            "  \"feature_pool_used\": [\"...\"],\n"
            "  \"channels\": [\n"
            "    {\n"
            "      \"name\": \"...\",\n"
            "      \"role\": \"...\",\n"
            "      \"expression\": \"...\",\n"
            "      \"output_range_hint\": [0.0, 1.0],\n"
            "      \"rationale\": \"short rationale\"\n"
            "    }\n"
            "  ],\n"
            "  \"aggregation\": {\n"
            "    \"method\": \"weighted_sum\",\n"
            "    \"weights\": [0.25, 0.25, 0.25, 0.25],\n"
            "    \"bias\": 0.0,\n"
            "    \"postprocess\": \"sigmoid\"\n"
            "  },\n"
            "  \"output_name\": \"sym_feature_v2\",\n"
            "  \"output_range_hint\": [0.0, 1.0],\n"
            "  \"notes\": \"short rationale\"\n"
            "}\n"
        )
    constraints = "Hard constraints:\n"
    if is_em_task:
        if exact_count > 0:
            constraints += f"1) Build exactly {exact_count} channels.\n"
            constraints += (
                f"2) Define exactly {exact_count} distinct role names yourself; "
                "role names should be short snake_case and semantically meaningful.\n"
            )
        else:
            constraints += f"1) Build at least {min_count} channels.\n"
            constraints += (
                f"2) Define at least {min_count} distinct role names yourself; "
                "role names should be short snake_case and semantically meaningful.\n"
            )
        constraints += "3) Every channel must target a distinct perspective and have a distinct expression.\n"
        constraints += "4) Keep channel logic compact: each channel should use 1-2 atoms and avoid long nested expressions.\n"
        constraints += (
            "5) Prefer decomposition over aggressive fusion: keep overlap/containment, serialization, "
            "semantic, numeric, and profile evidence reasonably disentangled.\n"
        )
        constraints += "6) At least 3 channels should use simple symbolic transforms (not raw single atom).\n"
        constraints += "7) Keep aggregation simple and interpretable; avoid over-smoothed all-channel blending.\n"
        constraints += "8) feature_pool_used must include all features referenced by all channels.\n"
        constraints += "9) JSON must be parseable by json.loads without edits.\n"
    else:
        if exact_count > 0:
            constraints += f"1) Build exactly {exact_count} channels.\n"
        else:
            constraints += f"1) Build at least {min_count} channels.\n"
        if channel_roles:
            constraints += f"2) Use role list exactly once each: {role_json}.\n"
        else:
            if exact_count > 0:
                constraints += (
                    f"2) Define exactly {exact_count} distinct role names yourself; "
                    "role names should be short snake_case and semantically meaningful.\n"
                )
            else:
                constraints += (
                    f"2) Define at least {min_count} distinct role names yourself; "
                    "role names should be short snake_case and semantically meaningful.\n"
                )
        constraints += "3) Every channel must target a distinct perspective and have a distinct expression.\n"
        constraints += "4) No two channels may use the same feature-set signature.\n"
        constraints += "5) Keep expressions numerically stable (avoid divide-by-zero; use safe_div if needed).\n"
        constraints += "6) feature_pool_used must include all features referenced by all channels.\n"
        constraints += "7) JSON must be parseable by json.loads without edits.\n"
        constraints += "8) Respect feature-card and dataset-context semantics; avoid incompatible feature mixing.\n"
        constraints += "9) decision is optional; if omitted evaluator defaults to threshold=0.5 and positive_if='>='.\n"
        if not channel_roles:
            constraints += (
                f"Role examples for inspiration (not mandatory): {role_examples_json}\n"
            )
        next_rule_idx = 10
        if int(min_passthrough_channels) > 0:
            constraints += (
                f"{next_rule_idx}) Build at least {int(min_passthrough_channels)} passthrough channels: "
                "their expression must be exactly one atom feature name from Feature Pool (no wrapper function).\n"
            )
            next_rule_idx += 1
        constraints += (
            f"{next_rule_idx}) Atoms from the same group family are correlated/redundant; "
            "do not treat them as independent evidence by default.\n"
        )
        next_rule_idx += 1
        constraints += (
            f"{next_rule_idx}) Use complementary evidence by default: when available, include at least one "
            "anchor channel that combines atoms from >=2 different evidence families.\n"
        )
        next_rule_idx += 1
        if bool(enable_single_atom_hint):
            constraints += (
                f"{next_rule_idx}) A single strong atom feature can be a valid channel expression; "
                "do not force every channel to combine many atoms. If used, ensure rationale explains why "
                "that single atom is robust and sufficient.\n"
            )
            next_rule_idx += 1
        if bool(enable_coverage_constraint):
            constraints += (
                f"{next_rule_idx}) Maximize atom coverage across channels: cover as many distinct atoms in "
                "feature_pool as possible.\n"
            )
    user_prompt = base_header + schema + constraints

    if str(dataset_context_block).strip():
        user_prompt += "\n" + dataset_context_block
    # Keep EM prompt compact: skip verbose task-group catalog block for EM only.
    if str(task_group_catalog_block).strip() and not is_em_task:
        user_prompt += "\n" + task_group_catalog_block
    if str(group_token_block).strip():
        user_prompt += "\n" + group_token_block
    if str(feature_cards_block).strip():
        user_prompt += "\n" + feature_cards_block
    return {"system": system_prompt, "user": user_prompt}


def _strip_output_range_hints_for_em_spec(doc: Dict[str, Any], *, task: str) -> Dict[str, Any]:
    """For EM generation, enforce no output_range_hint to avoid implicit clipping drift."""
    if str(task).strip() != "entity_matching" or not isinstance(doc, dict):
        return doc
    doc.pop("output_range_hint", None)
    channels = doc.get("channels", None)
    if isinstance(channels, list):
        for ch in channels:
            if isinstance(ch, dict):
                ch.pop("output_range_hint", None)
    return doc


def _is_passthrough_expression(expression: str, *, feature_pool_set: Set[str]) -> bool:
    expr = str(expression).strip()
    return expr in feature_pool_set


def _build_repair_user_prompt(
    *,
    base_user_prompt: str,
    violations: Sequence[str],
    attempt_no: int,
    total_attempts: int,
) -> str:
    lines = [base_user_prompt, "", f"REPAIR_INSTRUCTIONS attempt={attempt_no}/{total_attempts}:"]
    lines.append("Regenerate JSON and strictly satisfy the following failed constraints:")
    for idx, item in enumerate(violations, start=1):
        lines.append(f"- [{idx}] {item}")
    lines.append("Do not output explanation text. Output JSON object only.")
    return "\n".join(lines)


def _audit_generated_spec(
    *,
    spec: Any,
    feature_pool: Sequence[str],
    min_passthrough_channels: int,
    max_passthrough_ratio: Optional[float],
    exact_num_channels: int,
    min_num_channels: int,
) -> Dict[str, Any]:
    feature_pool_set = set(str(x) for x in feature_pool)
    channels = list(getattr(spec, "channels", []) or [])
    if not channels:
        return {
            "violations": ["v2 spec has no channels after validation."],
            "warnings": [],
            "passthrough_count": 0,
            "channel_count": 0,
            "passthrough_ratio_actual": 0.0,
            "non_passthrough_singleton_channels": [],
            "non_passthrough_no_anchor_channels": [],
        }

    passthrough_channels: List[str] = []
    non_passthrough_singleton_channels: List[str] = []
    non_passthrough_no_anchor_channels: List[str] = []
    passthrough_atom_names: Set[str] = set()
    atom_usage_counter: Counter[str] = Counter()
    used_atoms: Set[str] = set()

    for ch in channels:
        ch_name = str(getattr(ch, "name", "")).strip() or "unnamed_channel"
        ch_expr = str(getattr(ch, "expression", "")).strip()
        ch_features = set(str(x) for x in getattr(ch, "feature_names", ()) if str(x))
        atom_usage_counter.update(ch_features)
        used_atoms.update(ch_features)
        if _is_passthrough_expression(ch_expr, feature_pool_set=feature_pool_set):
            passthrough_channels.append(ch_name)
            passthrough_atom_names.add(ch_expr)
            continue
        if len(ch_features) <= 1:
            non_passthrough_singleton_channels.append(ch_name)

    channel_count = len(channels)
    passthrough_count = len(passthrough_channels)
    passthrough_ratio_actual = float(passthrough_count) / float(channel_count) if channel_count > 0 else 0.0

    violations: List[str] = []
    if int(exact_num_channels) > 0:
        if channel_count != int(exact_num_channels):
            violations.append(
                f"channel_count={channel_count} does not equal required exact_num_channels={int(exact_num_channels)}"
            )
    elif int(min_num_channels) > 0:
        if channel_count < int(min_num_channels):
            violations.append(
                f"channel_count={channel_count} is below required min_num_channels={int(min_num_channels)}"
            )
    if passthrough_count < int(min_passthrough_channels):
        violations.append(
            f"passthrough_count={passthrough_count} is below required min_passthrough_channels={int(min_passthrough_channels)}"
        )
    if max_passthrough_ratio is not None and passthrough_ratio_actual > float(max_passthrough_ratio):
        violations.append(
            "passthrough_ratio_actual="
            f"{passthrough_ratio_actual:.6f} exceeds max_passthrough_ratio={float(max_passthrough_ratio):.6f}"
        )

    warnings: List[str] = []
    if non_passthrough_singleton_channels:
        warnings.append(
            "non_passthrough_singleton_channels="
            + ",".join(non_passthrough_singleton_channels)
        )
    if non_passthrough_no_anchor_channels:
        warnings.append(
            "non_passthrough_no_anchor_channels="
            + ",".join(non_passthrough_no_anchor_channels)
        )
    if atom_usage_counter:
        high_reuse = [f"{k}:{v}" for k, v in atom_usage_counter.items() if int(v) >= 3]
        if high_reuse:
            warnings.append("high_atom_reuse(>=3)=" + ",".join(sorted(high_reuse)))

    return {
        "violations": violations,
        "warnings": warnings,
        "channel_count": channel_count,
        "passthrough_count": passthrough_count,
        "passthrough_ratio_actual": passthrough_ratio_actual,
        "passthrough_channels": passthrough_channels,
        "passthrough_atom_names": sorted(passthrough_atom_names),
        "non_passthrough_singleton_channels": non_passthrough_singleton_channels,
        "non_passthrough_no_anchor_channels": non_passthrough_no_anchor_channels,
        "atom_usage_counter": dict(sorted(atom_usage_counter.items())),
    }


def _call_llm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: float,
    max_completion_tokens: int,
    reasoning_effort: str,
    temperature: float | None,
) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("openai package is required. Install in ajoint env first.") from exc

    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    kwargs: Dict[str, Any] = {
        "api_key": str(api_key).strip(),
    }
    base = str(base_url).strip()
    if base:
        kwargs["base_url"] = base.rstrip("/")
    client = OpenAI(**kwargs)

    def _extract_text(message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out: List[str] = []
            for item in content:
                if isinstance(item, str):
                    out.append(item)
                    continue
                if isinstance(item, Mapping):
                    text_val = item.get("text", None)
                    if isinstance(text_val, str):
                        out.append(text_val)
                        continue
                    if item.get("type", None) == "output_text":
                        inner = item.get("text", "")
                        if isinstance(inner, str):
                            out.append(inner)
            return "".join(out)
        return ""

    def _make_request(*, candidate_model: str, token_budget: int):
        request_kwargs: Dict[str, Any] = {
            "model": str(candidate_model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "timeout": float(timeout_sec),
        }
        if int(token_budget) > 0:
            request_kwargs["max_completion_tokens"] = int(token_budget)
        effort = str(reasoning_effort).strip().lower()
        if effort in {"low", "medium", "high"}:
            request_kwargs["reasoning_effort"] = effort
        if temperature is not None:
            request_kwargs["temperature"] = float(temperature)
        return client.chat.completions.create(**request_kwargs)

    errors: List[str] = []
    call_records: List[Dict[str, Any]] = []
    for candidate in _model_candidates(model):
        started = time.perf_counter()
        usage_fields = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        try:
            token_budget = max(1, int(max_completion_tokens))
            resp = _make_request(candidate_model=str(candidate), token_budget=token_budget)
            usage_fields = _extract_usage_fields(resp)
            if not resp.choices:
                raise RuntimeError("LLM returned empty choices.")
            choice = resp.choices[0]
            content = _extract_text(choice.message).strip()
            if not content:
                finish_reason = str(getattr(choice, "finish_reason", "") or "").strip().lower()
                if finish_reason == "length":
                    retry_budget = max(token_budget * 4, 8000)
                    retry_resp = _make_request(candidate_model=str(candidate), token_budget=retry_budget)
                    usage_retry = _extract_usage_fields(retry_resp)
                    usage_fields = {
                        "prompt_tokens": (usage_fields["prompt_tokens"] or 0) + (usage_retry["prompt_tokens"] or 0),
                        "completion_tokens": (usage_fields["completion_tokens"] or 0) + (usage_retry["completion_tokens"] or 0),
                        "total_tokens": (usage_fields["total_tokens"] or 0) + (usage_retry["total_tokens"] or 0),
                    }
                    if not retry_resp.choices:
                        raise RuntimeError("LLM retry returned empty choices.")
                    retry_choice = retry_resp.choices[0]
                    retry_content = _extract_text(retry_choice.message).strip()
                    if retry_content:
                        content = retry_content
                    else:
                        retry_reason = str(getattr(retry_choice, "finish_reason", "") or "").strip()
                        raise RuntimeError(
                            f"LLM returned empty message content after retry (finish_reason={retry_reason!r})."
                        )
                else:
                    raise RuntimeError(f"LLM returned empty message content (finish_reason={finish_reason!r}).")
            elapsed_sec = float(time.perf_counter() - started)
            call_records.append(
                {
                    "candidate_model": str(candidate),
                    "status": "ok",
                    "elapsed_sec": elapsed_sec,
                    "prompt_tokens": usage_fields["prompt_tokens"],
                    "completion_tokens": usage_fields["completion_tokens"],
                    "total_tokens": usage_fields["total_tokens"],
                }
            )
            return str(content), call_records
        except Exception as exc:  # noqa: BLE001
            elapsed_sec = float(time.perf_counter() - started)
            call_records.append(
                {
                    "candidate_model": str(candidate),
                    "status": "error",
                    "elapsed_sec": elapsed_sec,
                    "prompt_tokens": usage_fields["prompt_tokens"],
                    "completion_tokens": usage_fields["completion_tokens"],
                    "total_tokens": usage_fields["total_tokens"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise LLMCallFailure("All model candidates failed: " + " | ".join(errors), call_records)


def _build_dry_run_v2(
    *,
    task: str,
    feature_pool: Sequence[str],
    num_channels: int,
    channel_roles: Sequence[str],
    min_passthrough_channels: int,
    anchor_atoms: Sequence[str],
) -> Dict[str, Any]:
    if len(feature_pool) < 2 and int(num_channels) > 1:
        raise ValueError("dry-run v2 requires at least 2 features when num_channels > 1")

    pool = list(feature_pool)
    n = len(pool)
    pool_set = set(pool)
    channels: List[Dict[str, Any]] = []
    used_signatures = set()
    channel_signatures: List[Tuple[str, ...]] = []
    used_features = set()

    passthrough_atoms: List[str] = []
    seen_passthrough = set()
    for atom in list(anchor_atoms) + list(pool):
        key = str(atom).strip()
        if not key or key in seen_passthrough or key not in pool_set:
            continue
        seen_passthrough.add(key)
        passthrough_atoms.append(key)
    passthrough_atoms = passthrough_atoms[: int(min_passthrough_channels)]

    resolved_roles = _resolve_channel_roles(
        task=str(task),
        num_channels=int(num_channels),
        provided_roles=list(channel_roles),
    )

    is_em_task = str(task).strip() == "entity_matching"
    for idx in range(int(num_channels)):
        role = str(resolved_roles[idx]).strip()

        if idx < len(passthrough_atoms):
            atom = passthrough_atoms[idx]
            expr = atom
            signature = tuple([atom])
            used_signatures.add(signature)
            used_features.add(atom)
            channel_obj = {
                "name": f"{role}_channel",
                "role": role,
                "expression": expr,
                "rationale": f"dry-run role={role} passthrough_atom={atom}",
            }
            if not is_em_task:
                channel_obj["output_range_hint"] = [0.0, 1.0]
            channels.append(channel_obj)
            channel_signatures.append(signature)
            continue

        found = False
        # Coverage-first pairing:
        # 1) anchor one atom by channel index
        # 2) choose a distant partner first (idx + num_channels) to increase distinct atom usage
        # 3) keep signature unique
        a = pool[idx % n]
        base_partner_shift = int(num_channels) % n
        if base_partner_shift == 0:
            base_partner_shift = 1
        for shift in range(n):
            b = pool[(idx + base_partner_shift + shift) % n]
            if b == a:
                continue
            signature = tuple(sorted({a, b}))
            if signature in used_signatures:
                continue
            used_signatures.add(signature)
            expr = f"clip(sigmoid(avg({a}, {b})), 0.0, 1.0)"
            used_features.update(signature)
            found = True
            break

        if not found:
            raise ValueError(
                "dry-run v2 could not build distinct channel feature signatures; "
                "reduce num_channels or provide richer feature pool"
            )

        channel_obj = {
            "name": f"{role}_channel",
            "role": role,
            "expression": expr,
            "rationale": f"dry-run role={role}",
        }
        if not is_em_task:
            channel_obj["output_range_hint"] = [0.0, 1.0]
        channels.append(channel_obj)
        channel_signatures.append(signature)

    # Coverage-first repair: rewrite trailing channels to include uncovered atoms while
    # preserving signature uniqueness.
    uncovered = [f for f in pool if f not in used_features]
    for miss_idx, miss in enumerate(uncovered):
        if not channels:
            break
        ch_idx = len(channels) - 1 - (miss_idx % len(channels))
        old_sig = channel_signatures[ch_idx]
        if old_sig in used_signatures:
            used_signatures.remove(old_sig)

        replacement_sig = None
        replacement_expr = None
        for partner in pool:
            if partner == miss:
                continue
            cand_sig = tuple(sorted({miss, partner}))
            if cand_sig in used_signatures:
                continue
            replacement_sig = cand_sig
            replacement_expr = f"clip(sigmoid(avg({miss}, {partner})), 0.0, 1.0)"
            break

        # If all signatures are exhausted, restore and skip this uncovered atom.
        if replacement_sig is None or replacement_expr is None:
            used_signatures.add(old_sig)
            continue

        channels[ch_idx]["expression"] = replacement_expr
        channels[ch_idx]["rationale"] = f"dry-run role={channels[ch_idx]['role']} coverage_repair"
        channel_signatures[ch_idx] = replacement_sig
        used_signatures.add(replacement_sig)
        used_features.update(replacement_sig)

    weights = [float(1.0 / len(channels)) for _ in channels]
    out = {
        "spec_version": "v2",
        "task": task,
        "feature_pool_used": sorted(used_features),
        "channels": channels,
        "aggregation": {
            "method": "weighted_sum",
            "weights": weights,
            "bias": 0.0,
            "postprocess": "sigmoid",
        },
        "output_name": "sym_feature_v2",
        "notes": "dry_run v2 bootstrap spec",
    }
    if not is_em_task:
        out["output_range_hint"] = [0.0, 1.0]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v2-style symbolic feature spec JSON with GPT-5 (v3 prompt).")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--feature-pool", type=str, default="", help="Comma-separated feature names.")
    parser.add_argument("--feature-pool-file", type=str, default="", help="txt/json file for feature names.")
    parser.add_argument("--replay-npz", type=str, default="", help="Load pair_feature_order from replay npz.")
    parser.add_argument(
        "--include-extra-features",
        type=str,
        default="none",
        choices=["none", "degree", "gnn", "all"],
        help="Extra features when --replay-npz is provided.",
    )
    parser.add_argument(
        "--spec-version",
        type=str,
        default="v2",
        choices=["v2"],
        help="Fixed to v2 in this script.",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Optional exact number of symbolic channels. If omitted, LLM chooses the count.",
    )
    parser.add_argument(
        "--min-num-channels",
        type=int,
        default=0,
        help="Minimum number of symbolic channels when --num-channels is omitted. <=0 means default to atom-pool size.",
    )
    parser.add_argument(
        "--channel-roles",
        type=str,
        default="",
        help="Optional. Comma-separated roles. If omitted, LLM defines roles by itself.",
    )
    parser.add_argument(
        "--task-hint-strong-atoms-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split used to rank strong atoms for Task Hint.",
    )
    parser.add_argument(
        "--task-hint-strong-atoms-topk",
        type=int,
        default=8,
        help="How many strong atoms to include in Task Hint.",
    )
    parser.add_argument(
        "--task-hint-atom-dicts",
        type=str,
        default=",".join(DEFAULT_ATOM_DICTIONARY_PATHS),
        help="Comma-separated atom_dictionary.json paths used for strong-atom priors.",
    )
    parser.add_argument(
        "--dataset-context-file",
        type=str,
        default="",
        help="Optional dataset-aware context json (train-only statistics).",
    )
    parser.add_argument(
        "--dataset-context-max-atoms",
        type=int,
        default=16,
        help="Maximum dataset-context atoms injected into prompt.",
    )
    parser.add_argument(
        "--allow-dataset-context",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 0, do not inject dataset context into prompt even when dataset-context-file is provided.",
    )
    parser.add_argument(
        "--enable-coverage-constraint",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, add hard prompt constraint to maximize atom coverage across channels.",
    )
    parser.add_argument("--objective", type=str, default="Maximize F1 with interpretable symbolic score.")
    parser.add_argument("--output", type=str, required=True, help="Output json path.")
    parser.add_argument(
        "--summary-output",
        type=str,
        default="",
        help="Optional path to persist final summary JSON.",
    )
    parser.add_argument(
        "--feature-cards-file",
        type=str,
        default=_default_feature_cards_path(),
        help="JSON file with per-feature semantic cards.",
    )
    parser.add_argument(
        "--feature-cards-max",
        type=int,
        default=80,
        help="Max feature cards to inject into prompt; <=0 disables cards.",
    )
    parser.add_argument(
        "--feature-cards-required",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, fail when cards file is missing/empty for requested feature pool.",
    )
    parser.add_argument(
        "--disallow-group-tokens",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, reject feature-pool entries that are group tokens in feature cards.",
    )
    parser.add_argument(
        "--group-token-prompt-max",
        type=int,
        default=32,
        help="Maximum number of detected group tokens injected into prompt hints; <=0 disables.",
    )
    parser.add_argument(
        "--enable-single-atom-hint",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, prompt explicitly tells LLM that a single strong atom feature can be effective.",
    )
    parser.add_argument(
        "--passthrough-ratio",
        type=float,
        default=_default_passthrough_ratio(),
        help="Minimum passthrough channel ratio in [0,1].",
    )
    parser.add_argument(
        "--max-audit-passthrough-ratio",
        type=float,
        default=-1.0,
        help="Maximum allowed audit_passthrough_ratio_actual in [0,1]. Negative disables this audit gate.",
    )
    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=1,
        help="Maximum number of LLM repair regeneration attempts after failed structural audit.",
    )
    parser.add_argument(
        "--dump-prompt",
        type=str,
        default="",
        help="Optional path to dump rendered system/user prompt for audit.",
    )
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--api-key-file", type=str, default=str(DEFAULT_API_KEY_FILE))
    parser.add_argument("--api-key-label", type=str, default="")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-completion-tokens", type=int, default=3200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for API call. None means API default.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["", "low", "medium", "high"],
        help="Reasoning effort hint for GPT-5 chat completion.",
    )
    parser.add_argument("--dry-run", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    task = str(args.task).strip()
    spec_version = "v2"

    feature_pool = _load_feature_pool(args)
    if not feature_pool:
        raise ValueError("Feature pool is empty. Provide --feature-pool or --replay-npz.")

    channel_roles = _parse_csv(args.channel_roles)
    requested_num_channels = int(args.num_channels) if args.num_channels is not None else 0
    requested_min_num_channels = int(args.min_num_channels)
    num_channels = requested_num_channels
    if num_channels <= 0 and channel_roles:
        num_channels = len(channel_roles)
    if num_channels > 0:
        if requested_min_num_channels > num_channels:
            raise ValueError(
                f"--min-num-channels ({requested_min_num_channels}) cannot exceed exact --num-channels ({num_channels})"
            )
        min_num_channels = num_channels
    else:
        min_num_channels = requested_min_num_channels if requested_min_num_channels > 0 else len(feature_pool)
    if min_num_channels <= 0:
        raise ValueError("Resolved min_num_channels must be > 0")
    if channel_roles:
        if len(channel_roles) != num_channels:
            raise ValueError(
                f"len(channel_roles) must equal num_channels ({num_channels}), got {len(channel_roles)}"
            )
        role_set = set()
        for role in channel_roles:
            token = role.lower()
            if token in role_set:
                raise ValueError(f"Duplicate channel role: {role!r}")
            role_set.add(token)
    role_examples = _task_role_examples(task=task, num_channels=(num_channels if num_channels > 0 else min_num_channels))

    atom_dict_paths = _parse_csv(str(args.task_hint_atom_dicts))
    strong_atom_split = str(args.task_hint_strong_atoms_split).strip().lower() or "test"
    task_hint_strong_atoms: Dict[str, List[Dict[str, Any]]] = {}
    if atom_dict_paths:
        task_hint_strong_atoms = _load_task_hint_strong_atoms(
            atom_dict_paths=atom_dict_paths,
            split=strong_atom_split,
            top_k=max(0, int(args.task_hint_strong_atoms_topk)),
        )
    task_hint_rows = list(task_hint_strong_atoms.get(task, []))
    task_hint_prior = _soft_prior_summary_from_rows(strong_atom_rows=task_hint_rows)
    rendered_task_hint = _render_task_hint(
        task=task,
        strong_atom_rows=task_hint_rows,
    )

    cards = _load_feature_cards(str(args.feature_cards_file))
    group_tokens = _detect_group_tokens(feature_pool=feature_pool, cards=cards)
    if int(args.disallow_group_tokens) == 1 and group_tokens:
        details = []
        for item in group_tokens:
            token = str(item.get("token", ""))
            expands = item.get("expands_to", [])
            if isinstance(expands, list) and expands:
                details.append(f"{token} -> {','.join(str(x) for x in expands)}")
            else:
                details.append(token)
        raise ValueError(
            "feature pool contains group tokens (atom features required): "
            + "; ".join(details)
        )

    if int(args.feature_cards_required) == 1:
        if not cards:
            raise ValueError(f"feature cards are required but unavailable: {args.feature_cards_file}")
        missing = [name for name in feature_pool if name not in cards]
        if missing:
            raise ValueError(
                "feature cards missing for requested features: " + ", ".join(sorted(missing))
            )

    feature_cards_block = _build_feature_cards_block(
        feature_pool=feature_pool,
        cards=cards,
        max_cards=int(args.feature_cards_max),
        compact=(task == "entity_matching"),
    )

    allow_dataset_context = bool(int(args.allow_dataset_context))
    dataset_context_requested = bool(str(args.dataset_context_file).strip())
    dataset_context_ignored = dataset_context_requested and not allow_dataset_context
    if allow_dataset_context:
        dataset_context = _load_dataset_context(str(args.dataset_context_file))
        dataset_context_block = _build_dataset_context_block(
            dataset_context,
            max_atoms=int(args.dataset_context_max_atoms),
        )
    else:
        dataset_context = {}
        dataset_context_block = ""

    group_token_block = _build_group_token_block(
        group_tokens=group_tokens,
        max_items=int(args.group_token_prompt_max),
    )
    task_group_catalog_block = _build_task_group_catalog_block(
        task=task,
        feature_pool=feature_pool,
        cards=cards,
        max_items=int(args.group_token_prompt_max),
    )
    passthrough_ratio = _normalize_ratio(float(args.passthrough_ratio), name="passthrough_ratio")
    max_audit_passthrough_ratio: Optional[float]
    if float(args.max_audit_passthrough_ratio) < 0.0:
        max_audit_passthrough_ratio = None
    else:
        max_audit_passthrough_ratio = _normalize_ratio(
            float(args.max_audit_passthrough_ratio),
            name="max_audit_passthrough_ratio",
        )
    min_passthrough_channels = _compute_min_passthrough_channels(
        num_channels=(num_channels if num_channels > 0 else min_num_channels),
        ratio=passthrough_ratio,
    )
    max_repair_attempts = max(0, int(args.max_repair_attempts))

    operator_examples = [
        "a+b",
        "a-b",
        "a*b",
        "safe_div(a,b)",
        "sigmoid(x)",
        "clip(x,0,1)",
        "abs(x)",
        "where(a>b,x,y)",
        "avg(a,b,c)",
    ]
    prompt = _build_prompt(
        task=task,
        feature_pool=feature_pool,
        objective=str(args.objective),
        operator_examples=operator_examples,
        feature_cards_block=feature_cards_block,
        task_group_catalog_block=task_group_catalog_block,
        group_token_block=group_token_block,
        dataset_context_block=dataset_context_block,
        num_channels_exact=num_channels,
        min_num_channels=min_num_channels,
        channel_roles=channel_roles,
        role_examples=role_examples,
        rendered_task_hint=rendered_task_hint,
        enable_coverage_constraint=bool(int(args.enable_coverage_constraint)),
        enable_single_atom_hint=bool(int(args.enable_single_atom_hint)),
        min_passthrough_channels=min_passthrough_channels,
    )

    if str(args.dump_prompt).strip():
        dump_path = Path(str(args.dump_prompt))
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_payload = {
            "task": task,
            "spec_version": spec_version,
            "feature_pool": list(feature_pool),
            "cards_injected": [name for name in feature_pool if name in cards][: int(max(0, int(args.feature_cards_max)))],
            "cards_missing": [name for name in feature_pool if name not in cards],
            "dataset_context_file": str(args.dataset_context_file),
            "dataset_context_requested": dataset_context_requested,
            "dataset_context_injected": allow_dataset_context,
            "dataset_context_ignored": dataset_context_ignored,
            "dataset_context_summary": {
                "dataset_name": dataset_context.get("dataset_name", "") if dataset_context else "",
                "task": dataset_context.get("task", "") if dataset_context else "",
                "selected_atoms": len(dataset_context.get("selected_atoms", [])) if dataset_context else 0,
            },
            "enable_coverage_constraint": bool(int(args.enable_coverage_constraint)),
            "enable_single_atom_hint": bool(int(args.enable_single_atom_hint)),
            "group_token_prompt_max": int(args.group_token_prompt_max),
            "task_group_catalog_tokens": list(TASK_GROUP_TOKEN_CATALOG.get(task, [])),
            "group_tokens_detected": group_tokens,
            "task_hint_strong_atoms_topk": int(args.task_hint_strong_atoms_topk),
            "task_hint_prior_atoms": list(task_hint_prior.get("atoms", [])),
            "task_hint_prior_families": list(task_hint_prior.get("families", [])),
            "num_channels_exact": num_channels,
            "min_num_channels": min_num_channels,
            "rendered_task_hint": rendered_task_hint,
            "channel_roles_input": list(channel_roles),
            "channel_role_examples": list(role_examples),
            "passthrough_ratio": passthrough_ratio,
            "max_audit_passthrough_ratio": max_audit_passthrough_ratio,
            "min_passthrough_channels": min_passthrough_channels,
            "max_repair_attempts": max_repair_attempts,
            "system_prompt": prompt["system"],
            "user_prompt": prompt["user"],
        }
        dump_path.write_text(json.dumps(dump_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    dry_run = int(args.dry_run) == 1
    discovered_api_key = ""
    if not dry_run and not str(args.api_key).strip() and str(args.api_key_file).strip():
        discovered_api_key = _discover_api_key(
            Path(str(args.api_key_file)).expanduser(),
            label=str(args.api_key_label),
        )
    base_user_prompt = prompt["user"]
    current_user_prompt = base_user_prompt
    attempt_records: List[Dict[str, Any]] = []
    llm_call_records: List[Dict[str, Any]] = []
    final_spec = None
    final_llm_raw = ""
    final_audit: Dict[str, Any] = {}

    for attempt_idx in range(max_repair_attempts + 1):
        attempt_no = int(attempt_idx + 1)
        total_attempts = int(max_repair_attempts + 1)
        try:
            if dry_run:
                generated = _build_dry_run_v2(
                    task=task,
                    feature_pool=feature_pool,
                    num_channels=(num_channels if num_channels > 0 else min_num_channels),
                    channel_roles=channel_roles,
                    min_passthrough_channels=min_passthrough_channels,
                    anchor_atoms=[],
                )
                llm_raw = json.dumps(generated, ensure_ascii=False)
            else:
                llm_raw, call_records = _call_llm(
                    base_url=str(args.base_url),
                    api_key=str(args.api_key).strip() or str(discovered_api_key).strip(),
                    model=str(args.model),
                    system_prompt=prompt["system"],
                    user_prompt=current_user_prompt,
                    timeout_sec=float(args.timeout_sec),
                    max_completion_tokens=int(args.max_completion_tokens),
                    reasoning_effort=str(args.reasoning_effort),
                    temperature=args.temperature,
                )
                for rec in call_records:
                    rec["repair_attempt"] = attempt_no
                    llm_call_records.append(rec)

            parsed = extract_json_object(llm_raw)
            parsed = _strip_output_range_hints_for_em_spec(parsed, task=task)
            spec = validate_symbolic_feature_spec(
                parsed,
                expected_task=task,
                allowed_features=feature_pool,
            )
            if str(spec.spec_version).strip().lower() != "v2":
                raise ValueError(f"Generator returned non-v2 spec: {spec.spec_version}")

            attempt_records.append(
                {
                    "attempt": attempt_no,
                    "status": "ok",
                    "channel_count": int(len(spec.channels)),
                }
            )

            audit = _audit_generated_spec(
                spec=spec,
                feature_pool=feature_pool,
                min_passthrough_channels=min_passthrough_channels,
                max_passthrough_ratio=max_audit_passthrough_ratio,
                exact_num_channels=num_channels,
                min_num_channels=min_num_channels,
            )
            if audit.get("violations"):
                raise ValueError("; ".join(str(x) for x in audit.get("violations", [])))

            final_spec = spec
            final_llm_raw = llm_raw
            final_audit = audit
            break
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, LLMCallFailure):
                for rec in exc.call_records:
                    rec["repair_attempt"] = attempt_no
                    llm_call_records.append(rec)
            error_text = f"{type(exc).__name__}: {exc}"
            attempt_records.append(
                {
                    "attempt": attempt_no,
                    "status": "error",
                    "error": error_text,
                }
            )
            if dry_run or attempt_idx >= max_repair_attempts:
                # Do not raise here; emit a structured failure summary with
                # accumulated usage/time stats after the loop.
                break
            current_user_prompt = _build_repair_user_prompt(
                base_user_prompt=base_user_prompt,
                violations=[error_text],
                attempt_no=attempt_no + 1,
                total_attempts=total_attempts,
            )
            continue

    if final_spec is None:
        failure_summary = {
            "ok": False,
            "task": task,
            "spec_version": spec_version,
            "output": str(Path(args.output).resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": str(args.model),
            "reasoning_effort": str(args.reasoning_effort),
            "temperature": args.temperature,
            "max_repair_attempts": max_repair_attempts,
            "repair_attempts_used": len(attempt_records),
            "attempt_records": attempt_records,
            "llm_api_request_count": int(len(llm_call_records)),
            "llm_api_success_count": int(sum(1 for rec in llm_call_records if rec.get("status") == "ok")),
            "llm_api_error_count": int(sum(1 for rec in llm_call_records if rec.get("status") != "ok")),
            "llm_prompt_tokens_total": int(sum(int(rec.get("prompt_tokens") or 0) for rec in llm_call_records)),
            "llm_completion_tokens_total": int(sum(int(rec.get("completion_tokens") or 0) for rec in llm_call_records)),
            "llm_total_tokens_total": int(sum(int(rec.get("total_tokens") or 0) for rec in llm_call_records)),
            "llm_elapsed_sec_total": float(sum(float(rec.get("elapsed_sec") or 0.0) for rec in llm_call_records)),
            "llm_calls": llm_call_records,
            "error": "failed to produce a valid symbolic spec after repair attempts",
        }
        if str(args.summary_output).strip():
            summary_path = Path(str(args.summary_output)).expanduser()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            failure_summary["summary_output"] = str(summary_path.resolve())
            summary_path.write_text(json.dumps(failure_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(failure_summary, ensure_ascii=False, indent=2))
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_symbolic_feature_spec(final_spec, out_path)

    summary = {
        "ok": True,
        "task": final_spec.task,
        "spec_version": final_spec.spec_version,
        "spec_id": final_spec.spec_id,
        "spec_hash": final_spec.spec_hash,
        "feature_pool_used": list(final_spec.feature_pool_used),
        "expression": final_spec.expression,
        "channel_count": int(len(final_spec.channels)),
        "channel_roles": [ch.role for ch in final_spec.channels],
        "output": str(out_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
        "temperature": args.temperature,
        "allowed_functions": allowed_symbolic_function_names(),
        "feature_cards_file": str(Path(args.feature_cards_file).expanduser()),
        "feature_cards_injected": int(sum(1 for name in feature_pool if name in cards)),
        "feature_cards_missing": [name for name in feature_pool if name not in cards],
        "group_tokens_detected": group_tokens,
        "task_group_catalog_tokens": list(TASK_GROUP_TOKEN_CATALOG.get(task, [])),
        "group_token_prompt_max": int(args.group_token_prompt_max),
        "dataset_context_file": str(args.dataset_context_file),
        "dataset_context_requested": dataset_context_requested,
        "dataset_context_injected": allow_dataset_context,
        "dataset_context_ignored": dataset_context_ignored,
        "dataset_context_atoms": int(len(dataset_context.get("selected_atoms", []))) if dataset_context else 0,
        "enable_coverage_constraint": bool(int(args.enable_coverage_constraint)),
        "enable_single_atom_hint": bool(int(args.enable_single_atom_hint)),
        "task_hint_strong_atoms_topk": int(args.task_hint_strong_atoms_topk),
            "task_hint_prior_atoms": list(task_hint_prior.get("atoms", [])),
            "task_hint_prior_families": list(task_hint_prior.get("families", [])),
            "num_channels_exact": num_channels,
            "min_num_channels": min_num_channels,
            "channel_roles_input": list(channel_roles),
        "channel_role_examples": list(role_examples),
        "passthrough_ratio": passthrough_ratio,
        "max_audit_passthrough_ratio": max_audit_passthrough_ratio,
        "min_passthrough_channels": min_passthrough_channels,
        "max_repair_attempts": max_repair_attempts,
        "repair_attempts_used": len(attempt_records),
        "attempt_records": attempt_records,
        "llm_api_request_count": int(len(llm_call_records)),
        "llm_api_success_count": int(sum(1 for rec in llm_call_records if rec.get("status") == "ok")),
        "llm_api_error_count": int(sum(1 for rec in llm_call_records if rec.get("status") != "ok")),
        "llm_prompt_tokens_total": int(sum(int(rec.get("prompt_tokens") or 0) for rec in llm_call_records)),
        "llm_completion_tokens_total": int(sum(int(rec.get("completion_tokens") or 0) for rec in llm_call_records)),
        "llm_total_tokens_total": int(sum(int(rec.get("total_tokens") or 0) for rec in llm_call_records)),
        "llm_elapsed_sec_total": float(sum(float(rec.get("elapsed_sec") or 0.0) for rec in llm_call_records)),
        "llm_calls": llm_call_records,
        "audit_warnings": list(final_audit.get("warnings", [])),
        "audit_passthrough_count": int(final_audit.get("passthrough_count", 0)),
        "audit_passthrough_ratio_actual": float(final_audit.get("passthrough_ratio_actual", 0.0)),
    }
    if str(args.dump_prompt).strip():
        summary["prompt_dump"] = str(Path(args.dump_prompt).resolve())
    if final_llm_raw:
        summary["raw_text_chars"] = len(final_llm_raw)
    if str(args.summary_output).strip():
        summary_path = Path(str(args.summary_output)).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary["summary_output"] = str(summary_path.resolve())
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
