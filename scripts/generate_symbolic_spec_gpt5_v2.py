#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
        "Row-pair matching: high precision with robust borderline recall. "
        "In prior EM runs, row_avg_len_ratio and row_numeric_ratio_sim were often weak/unstable; "
        "use them conservatively and only with strong supporting signals."
    ),
    "joinable_table_search": "Column-pair joinability: containment/coverage/value consistency are important.",
    "union_table_search": "Table-pair unionability: bidirectional overlap and size compatibility matter.",
    "schema_matching": "Column semantic alignment: header/value consistency and stability are important.",
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


def _parse_csv(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


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
        }
    return out


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


def _build_feature_cards_block(
    *,
    feature_pool: Sequence[str],
    cards: Mapping[str, Mapping[str, str]],
    max_cards: int,
) -> str:
    lines: List[str] = []
    if int(max_cards) <= 0:
        return ""
    for name in feature_pool[: int(max_cards)]:
        key = str(name)
        meta = cards.get(key, {})
        if meta and _is_group_card(meta):
            # Group tokens are filtered by default; skip here as a second safety net.
            continue
        definition = str(meta.get("definition", "")).strip() or "No curated definition provided."
        formula = str(meta.get("formula", "")).strip() or "Use as provided in feature pool."
        range_hint = str(meta.get("range", "")).strip() or "Unknown."
        caution = str(meta.get("caution", "")).strip() or "Treat uncertain behavior conservatively."
        lines.append(f"- {key}")
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
    dataset_context_block: str,
    num_channels: int,
    channel_roles: Sequence[str],
    enable_coverage_constraint: bool,
    enable_em_prompt_boost: bool,
) -> Dict[str, str]:
    task_hint = TASK_HINTS.get(task, "General binary matching task.")
    system_prompt = (
        "You design symbolic decision algorithms for tabular entity linkage tasks. "
        "Return JSON only. Do not include Markdown fences or any extra text."
    )

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

    if int(num_channels) <= 0:
        raise ValueError("num_channels must be > 0 for v2")
    if not channel_roles:
        raise ValueError("channel_roles must be provided for v2")

    role_json = json.dumps(list(channel_roles), ensure_ascii=False)
    is_em_task = str(task).strip() == "entity_matching"
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
    constraints = (
        "Hard constraints:\n"
        f"1) Build exactly {int(num_channels)} channels.\n"
        f"2) Use role list exactly once each: {role_json}.\n"
        "3) Every channel must target a distinct perspective and have a distinct expression.\n"
        "4) No two channels may use the same feature-set signature.\n"
        "5) Keep expressions numerically stable (avoid divide-by-zero; use safe_div if needed).\n"
        "6) feature_pool_used must include all features referenced by all channels.\n"
        "7) JSON must be parseable by json.loads without edits.\n"
        "8) Respect feature-card and dataset-context semantics; avoid incompatible feature mixing.\n"
        "9) decision is optional; if omitted evaluator defaults to threshold=0.5 and positive_if='>='.\n"
    )
    if bool(enable_coverage_constraint):
        constraints += "10) Maximize atom coverage across channels: cover as many distinct atoms in feature_pool as possible.\n"
    user_prompt = base_header + schema + constraints

    if str(dataset_context_block).strip():
        user_prompt += "\n" + dataset_context_block
    if str(feature_cards_block).strip():
        user_prompt += "\n" + feature_cards_block
    return {"system": system_prompt, "user": user_prompt}


def _strip_output_range_hints_for_em_spec(doc: Dict[str, Any], *, task: str) -> Dict[str, Any]:
    if str(task).strip() != "entity_matching" or not isinstance(doc, dict):
        return doc
    doc.pop("output_range_hint", None)
    channels = doc.get("channels", None)
    if isinstance(channels, list):
        for ch in channels:
            if isinstance(ch, dict):
                ch.pop("output_range_hint", None)
    return doc


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
) -> str:
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

    errors: List[str] = []
    for candidate in _model_candidates(model):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": str(candidate),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "timeout": float(timeout_sec),
            }
            if int(max_completion_tokens) > 0:
                request_kwargs["max_completion_tokens"] = int(max_completion_tokens)
            effort = str(reasoning_effort).strip().lower()
            if effort in {"low", "medium", "high"}:
                request_kwargs["reasoning_effort"] = effort
            if temperature is not None:
                request_kwargs["temperature"] = float(temperature)

            resp = client.chat.completions.create(**request_kwargs)
            if not resp.choices:
                raise RuntimeError("LLM returned empty choices.")
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned empty message content.")
            return str(content)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All model candidates failed: " + " | ".join(errors))


def _build_dry_run_v2(
    *,
    task: str,
    feature_pool: Sequence[str],
    num_channels: int,
    channel_roles: Sequence[str],
) -> Dict[str, Any]:
    if len(feature_pool) < 2 and int(num_channels) > 1:
        raise ValueError("dry-run v2 requires at least 2 features when num_channels > 1")

    pool = list(feature_pool)
    n = len(pool)
    channels: List[Dict[str, Any]] = []
    used_signatures = set()
    channel_signatures: List[Tuple[str, ...]] = []
    used_features = set()

    is_em_task = str(task).strip() == "entity_matching"
    for idx in range(int(num_channels)):
        role = str(channel_roles[idx]).strip()
        if not role:
            raise ValueError(f"channel_roles[{idx}] is empty")

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
    parser = argparse.ArgumentParser(description="Generate v2 symbolic feature spec JSON with GPT-5.")
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
        required=True,
        help="Required. Number of symbolic channels.",
    )
    parser.add_argument(
        "--channel-roles",
        type=str,
        required=True,
        help="Required. Comma-separated roles, length must equal num-channels.",
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
        "--enable-coverage-constraint",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, add hard prompt constraint to maximize atom coverage across channels.",
    )
    parser.add_argument(
        "--enable-em-prompt-boost",
        type=int,
        default=0,
        choices=[0, 1],
        help="Compatibility flag (currently no-op).",
    )
    parser.add_argument("--objective", type=str, default="Maximize F1 with interpretable symbolic score.")
    parser.add_argument("--output", type=str, required=True, help="Output json path.")
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
        "--dump-prompt",
        type=str,
        default="",
        help="Optional path to dump rendered system/user prompt for audit.",
    )
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-completion-tokens", type=int, default=1800)
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
    num_channels = int(args.num_channels)
    if num_channels <= 0:
        raise ValueError("--num-channels must be > 0")
    if not channel_roles:
        raise ValueError("--channel-roles is required")
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
    )

    dataset_context = _load_dataset_context(str(args.dataset_context_file))
    dataset_context_block = _build_dataset_context_block(
        dataset_context,
        max_atoms=int(args.dataset_context_max_atoms),
    )

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
        dataset_context_block=dataset_context_block,
        num_channels=num_channels,
        channel_roles=channel_roles,
        enable_coverage_constraint=bool(int(args.enable_coverage_constraint)),
        enable_em_prompt_boost=bool(int(args.enable_em_prompt_boost)),
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
            "dataset_context_summary": {
                "dataset_name": dataset_context.get("dataset_name", "") if dataset_context else "",
                "task": dataset_context.get("task", "") if dataset_context else "",
                "selected_atoms": len(dataset_context.get("selected_atoms", [])) if dataset_context else 0,
            },
            "enable_coverage_constraint": bool(int(args.enable_coverage_constraint)),
            "enable_em_prompt_boost": bool(int(args.enable_em_prompt_boost)),
            "system_prompt": prompt["system"],
            "user_prompt": prompt["user"],
        }
        dump_path.write_text(json.dumps(dump_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if int(args.dry_run) == 1:
        generated = _build_dry_run_v2(
            task=task,
            feature_pool=feature_pool,
            num_channels=num_channels,
            channel_roles=channel_roles,
        )
        llm_raw = json.dumps(generated, ensure_ascii=False)
    else:
        llm_raw = _call_llm(
            base_url=str(args.base_url),
            api_key=str(args.api_key),
            model=str(args.model),
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
            timeout_sec=float(args.timeout_sec),
            max_completion_tokens=int(args.max_completion_tokens),
            reasoning_effort=str(args.reasoning_effort),
            temperature=args.temperature,
        )

    parsed = extract_json_object(llm_raw)
    parsed = _strip_output_range_hints_for_em_spec(parsed, task=task)
    spec = validate_symbolic_feature_spec(
        parsed,
        expected_task=task,
        allowed_features=feature_pool,
    )
    if str(spec.spec_version).strip().lower() != "v2":
        raise ValueError(f"Generator returned non-v2 spec: {spec.spec_version}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_symbolic_feature_spec(spec, out_path)

    summary = {
        "ok": True,
        "task": spec.task,
        "spec_version": spec.spec_version,
        "spec_id": spec.spec_id,
        "spec_hash": spec.spec_hash,
        "feature_pool_used": list(spec.feature_pool_used),
        "expression": spec.expression,
        "channel_count": int(len(spec.channels)),
        "channel_roles": [ch.role for ch in spec.channels],
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
        "dataset_context_file": str(args.dataset_context_file),
        "dataset_context_atoms": int(len(dataset_context.get("selected_atoms", []))) if dataset_context else 0,
        "enable_coverage_constraint": bool(int(args.enable_coverage_constraint)),
        "enable_em_prompt_boost": bool(int(args.enable_em_prompt_boost)),
    }
    if str(args.dump_prompt).strip():
        summary["prompt_dump"] = str(Path(args.dump_prompt).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
