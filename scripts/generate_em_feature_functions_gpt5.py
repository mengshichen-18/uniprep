#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generated_feature_runtime import GeneratedFeatureSpec, save_generated_feature_specs, validate_generated_feature_spec  # noqa: E402


DEFAULT_EXAMPLES_FILE = _ROOT / "generated_feature_examples" / "entity_matching_fixed_examples.json"
DEFAULT_FEATURE_CARDS_FILE = _ROOT / "symbolic_feature_cards.json"
DEFAULT_API_KEY_FILE = _ROOT.parent / "0325_policy_pro" / "LIGHTNING_API_KEY.md"
DEFAULT_TEACHER_FEATURE_POOL = [
    "row_emb_cosine",
    "row_emb_l1_sim",
    "row_value_jaccard",
    "row_value_containment_max",
    "row_token_jaccard",
    "row_nonempty_ratio",
    "row_numeric_ratio_sim",
    "row_avg_len_ratio",
    "row_serial_token_jaccard",
    "row_serial_edit_similarity",
    "row_numeric_value_overlap",
    "row_serial_char3_jaccard",
    "row_serial_char4_jaccard",
    "row_token_idf_jaccard",
    "row_numeric_rel_diff_sim",
]
DEFAULT_PRESERVED_FEATURES = [
    "row_value_jaccard",
    "row_serial_token_jaccard",
    "row_numeric_rel_diff_sim",
]
DEFAULT_TASK_DESCRIPTION = (
    "Entity matching on heterogeneous table rows. Each candidate pair links two rows that may describe the same real-world "
    "entity even when cell values are noisy, partially missing, reordered, or formatted differently."
)
DEFAULT_DATA_DESCRIPTION = (
    "Each row is represented by normalized cell values, row-level token/value sets, serialized row text, header-aware "
    "signals, simple numeric summaries, and row embeddings. Good atom features should be smooth, interpretable, and "
    "robust across datasets with different schemas and value styles."
)
DEFAULT_SELECTION_NOTES = (
    "The teacher pool is manually defined. The preserved exemplar atoms are also manually selected by us; the model must "
    "treat them as examples of desired feature style and as protected atoms that should not be regenerated."
)


def _load_json(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def _load_examples(path: Path) -> List[Mapping[str, Any]]:
    payload = _load_json(path)
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"'features' must be a list in {path}")
    return [item for item in features if isinstance(item, Mapping)]


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _select_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    preserved_feature_names: Sequence[str],
) -> List[Mapping[str, Any]]:
    wanted = {str(name).strip() for name in preserved_feature_names if str(name).strip()}
    if not wanted:
        return list(examples)
    out: List[Mapping[str, Any]] = []
    for item in examples:
        based_on = item.get("example_based_on", [])
        tokens = [str(x).strip() for x in based_on] if isinstance(based_on, list) else [str(based_on).strip()]
        if any(token in wanted for token in tokens):
            out.append(item)
    return out


def _load_feature_cards(path: Path, feature_names: Sequence[str]) -> Dict[str, Mapping[str, Any]]:
    payload = _load_json(path)
    features = payload.get("features", {})
    if not isinstance(features, Mapping):
        return {}
    out: Dict[str, Mapping[str, Any]] = {}
    for name in feature_names:
        meta = features.get(str(name), None)
        if isinstance(meta, Mapping):
            out[str(name)] = meta
    return out


def _load_dataset_context(path: str) -> Dict[str, Any]:
    token = str(path).strip()
    if not token:
        return {}
    p = Path(token).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"dataset context file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dataset context must be a JSON object.")
    return dict(payload)


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
                return str(match.group(1)).strip().strip('\'"')
    generic = re.findall(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{15,}", raw)
    return str(generic[-1]).strip() if generic else ""


def _discover_api_key(path: Path, *, label: str = "") -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _discover_api_key_from_text(text, label=label)


def _ctx_schema_text() -> str:
    return """Available ctx structure for compute_feature(ctx):
- ctx["stats_a"], ctx["stats_b"]: row-level dictionaries from checkpoint-02 EM row stats
  keys include:
  - value_set: set[str]
  - token_set: set[str]
  - header_token_set: set[str]
  - header_value_token_set: set[str]
  - serial_token_set: set[str]
  - serial_text: str
  - numeric_value_set: set[str]
  - header_to_value: dict[str, str]
  - nonempty_count: int
  - numeric_ratio: float
  - avg_len: float
  - numeric_median: float
- ctx["emb_a"], ctx["emb_b"]: numpy arrays for row embeddings
- ctx["helpers"]: dict of safe helper callables
  - helpers["safe_ratio_float"](a, b)
  - helpers["token_jaccard"](tokens_a, tokens_b)
  - helpers["normalized_edit_similarity"](text_a, text_b)
  - helpers["char_ngram_jaccard"](text_a, text_b, n)
  - helpers["cosine_similarity"](vec_a, vec_b)
  - helpers["l1_similarity"](vec_a, vec_b)
  - helpers["numeric_overlap_max"](values_a, values_b)
  - helpers["clip01"](x)
  - helpers["set_jaccard"](set_a, set_b)
"""


def _build_teacher_pool_block(
    *,
    teacher_feature_pool: Sequence[str],
    cards: Mapping[str, Mapping[str, Any]],
) -> str:
    if not teacher_feature_pool:
        return ""
    lines: List[str] = []
    lines.append(f"Teacher atom pool ({len(teacher_feature_pool)} atoms):")
    lines.append(json.dumps(list(teacher_feature_pool), ensure_ascii=False))
    lines.append("Teacher-pool semantic cards:")
    for name in teacher_feature_pool:
        meta = cards.get(str(name), {})
        definition = str(meta.get("definition", "")).strip() or "No curated definition provided."
        formula = str(meta.get("formula", "")).strip() or "No curated formula provided."
        range_hint = str(meta.get("range", "")).strip() or "Unknown."
        caution = str(meta.get("caution", "")).strip() or "Use conservatively."
        lines.append(
            f"- {name}: definition={definition} ; formula={formula} ; range={range_hint} ; caution={caution}"
        )
    return "\n".join(lines) + "\n"


def _build_dataset_context_block(context: Mapping[str, Any]) -> str:
    if not context:
        return ""
    dataset = str(context.get("dataset_name", "")).strip()
    task = str(context.get("task", "")).strip()
    split = str(context.get("split", "train")).strip() or "train"
    atoms = context.get("selected_atoms", [])
    if not isinstance(atoms, list):
        atoms = []
    lines: List[str] = []
    lines.append("Dataset-aware context (use as soft guidance only):")
    if dataset:
        lines.append(f"- dataset: {dataset}")
    if task:
        lines.append(f"- task: {task}")
    lines.append(f"- split: {split} (train-only context)")
    used = 0
    for item in atoms:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        auc = item.get("auc", None)
        best_f1 = item.get("best_f1", None)
        missing = item.get("missing_rate", None)
        lines.append(f"- atom={name}: auc={auc}, best_f1={best_f1}, missing={missing}")
        used += 1
        if used >= 8:
            break
    if used <= 0:
        lines.append("- no selected_atoms payload was provided")
    return "\n".join(lines) + "\n"


def _build_prompt(
    *,
    feature_request: str,
    num_features: int,
    target_total_pool_size: int,
    preserved_feature_names: Sequence[str],
    teacher_feature_pool: Sequence[str],
    forbidden_feature_names: Sequence[str],
    fixed_examples: Sequence[Mapping[str, Any]],
    teacher_cards: Mapping[str, Mapping[str, Any]],
    task_description: str,
    data_description: str,
    selection_notes: str,
    dataset_context: Mapping[str, Any],
) -> Dict[str, str]:
    system = (
        "You design Python feature functions for entity matching. "
        "Return JSON only with a top-level object {\"features\": [...]} and no markdown."
    )

    example_lines: List[str] = []
    for item in fixed_examples:
        feature_name = str(item.get("feature_name", "")).strip()
        desc = str(item.get("description", "")).strip()
        code = str(item.get("code", "")).strip()
        example_lines.append(f"- {feature_name}: {desc}\n```python\n{code}\n```")
    teacher_pool_block = _build_teacher_pool_block(
        teacher_feature_pool=teacher_feature_pool,
        cards=teacher_cards,
    )
    dataset_context_block = _build_dataset_context_block(dataset_context)
    exemplar_count = len([name for name in preserved_feature_names if str(name).strip()])
    total_pool = len([name for name in teacher_feature_pool if str(name).strip()])
    total_target = int(target_total_pool_size) if int(target_total_pool_size) > 0 else (exemplar_count + int(num_features))
    forbidden_names = [str(name).strip() for name in forbidden_feature_names if str(name).strip()]

    user = (
        f"Task: entity_matching\n"
        f"Task description: {str(task_description).strip()}\n"
        f"Data description: {str(data_description).strip()}\n"
        f"Goal: generate {int(num_features)} new row-pair feature function(s).\n"
        f"Teacher pool size: {total_pool}\n"
        f"Human-selected exemplar count: {exemplar_count}\n"
        f"Target completed pool size after generation: {total_target}\n"
        f"Feature request: {feature_request}\n"
        f"Selection notes: {str(selection_notes).strip()}\n"
        f"Preserved fixed features already kept in the model and should be treated as examples, not regenerated: "
        f"{json.dumps(list(preserved_feature_names), ensure_ascii=False)}\n\n"
        f"Forbidden feature names (do not reuse; choose genuinely new names): {json.dumps(forbidden_names, ensure_ascii=False)}\n\n"
        "Hard constraints:\n"
        "1) The teacher pool and the preserved exemplar atoms are human-defined; treat them as trusted guidance, not as outputs to regenerate.\n"
        "2) Each feature must be a genuinely new feature name, not one of the preserved fixed features and not one of the forbidden feature names.\n"
        "3) New features should complement the exemplars and broaden the pool rather than duplicate the same evidence family.\n"
        "4) At least half of the generated features should use evidence families that are not identical to a preserved exemplar's single formula.\n"
        "5) Each feature must define Python code with exactly one function: def compute_feature(ctx):\n"
        "6) Do not import anything. Do not use file IO, network IO, eval, exec, globals, classes, decorators, or exceptions.\n"
        "7) Use only ctx fields and helper functions described below.\n"
        "8) Return a single numeric scalar convertible to float.\n"
        "9) Prefer smooth numeric features over brittle hard-threshold rules.\n"
        "10) Keep each function short and interpretable.\n"
        "11) Provide fields: feature_name, task, scope, version, description, inputs_used, code, fallback_value, range_hint, example_based_on.\n"
        "12) task must be entity_matching and scope must be row_pair.\n"
        "13) range_hint should be omitted or be a finite [lo, hi].\n"
        "14) example_based_on must be a JSON array of strings, even when there is only one item.\n"
        "15) Allowed builtin calls are only: abs, min, max, float, int, len, sum, sorted, bool, list, set, tuple, str.\n"
        "16) Allowed non-helper methods are only: get, keys, intersection, union, append.\n"
        "17) Do not define nested functions, lambdas, local classes, or helper closures inside compute_feature.\n"
        "18) Avoid any other method call pattern.\n\n"
        + _ctx_schema_text()
        + "\n"
        + teacher_pool_block
        + ("\n" + dataset_context_block if dataset_context_block else "")
        + "\nPreserved fixed-feature examples:\n"
        + "\n".join(example_lines)
        + "\n\nReturn JSON only."
    )
    return {"system": system, "user": user}


def _extract_json_object(text: str) -> Mapping[str, Any]:
    raw = str(text).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, Mapping):
        raise ValueError("Model output JSON must be an object.")
    return obj


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    out_path = Path(str(path)).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_validation_report(
    *,
    specs: Sequence[GeneratedFeatureSpec],
    dry_run: bool,
    output_path: str,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "output": str(Path(output_path).resolve()),
        "feature_count": int(len(specs)),
        "feature_names": [str(spec.feature_name) for spec in specs],
        "features": [
            {
                "feature_name": str(spec.feature_name),
                "task": str(spec.task),
                "scope": str(spec.scope),
                "version": str(spec.version),
                "fallback_value": float(spec.fallback_value),
                "range_hint": list(spec.range_hint) if spec.range_hint is not None else None,
                "inputs_used": [str(item) for item in spec.inputs_used],
                "example_based_on": [str(item) for item in spec.example_based_on],
            }
            for spec in specs
        ],
    }


def _call_llm(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: float,
    reasoning_effort: str,
    max_completion_tokens: int,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("openai package is required to generate feature functions.") from exc

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or --api-key.")

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if str(base_url).strip():
        kwargs["base_url"] = str(base_url).strip().rstrip("/")
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

    def _make_request(token_budget: int):
        request_kwargs: Dict[str, Any] = {
            "model": str(model).strip(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "timeout": float(timeout_sec),
        }
        effort = str(reasoning_effort).strip().lower()
        if effort in {"low", "medium", "high"}:
            request_kwargs["reasoning_effort"] = effort
        if int(token_budget) > 0:
            request_kwargs["max_completion_tokens"] = int(token_budget)
        return client.chat.completions.create(**request_kwargs)

    token_budget = max(1, int(max_completion_tokens))
    resp = _make_request(token_budget)
    if not resp.choices:
        raise RuntimeError("LLM returned empty choices.")
    choice = resp.choices[0]
    message = choice.message
    content = _extract_text(message).strip()
    if content:
        return content

    # GPT-5 may spend the entire completion budget on reasoning tokens and
    # return an empty assistant content string with finish_reason='length'.
    finish_reason = str(getattr(choice, "finish_reason", "") or "").strip().lower()
    if finish_reason == "length":
        retry_budget = max(token_budget * 4, 8000)
        retry_resp = _make_request(retry_budget)
        if not retry_resp.choices:
            raise RuntimeError("LLM retry returned empty choices.")
        retry_choice = retry_resp.choices[0]
        retry_content = _extract_text(retry_choice.message).strip()
        if retry_content:
            return retry_content
        retry_reason = str(getattr(retry_choice, "finish_reason", "") or "").strip()
        raise RuntimeError(f"LLM returned empty content after retry (finish_reason={retry_reason!r}).")

    raise RuntimeError(f"LLM returned empty content (finish_reason={finish_reason!r}).")


def _build_dry_run_feature_docs() -> List[Dict[str, Any]]:
    return [
        {
            "feature_name": "row_header_serial_blend",
            "task": "entity_matching",
            "scope": "row_pair",
            "version": "v1",
            "description": "Blend header:value token overlap with serialized token overlap to capture schema-aware lexical agreement.",
            "inputs_used": [
                "stats_a.header_value_token_set",
                "stats_b.header_value_token_set",
                "stats_a.serial_token_set",
                "stats_b.serial_token_set",
                "helpers.token_jaccard",
                "helpers.clip01",
            ],
            "fallback_value": 0.0,
            "range_hint": [0.0, 1.0],
            "example_based_on": [
                "row_serial_token_jaccard",
                "row_header_token_jaccard",
            ],
            "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    hv = helpers[\"token_jaccard\"](list(stats_a[\"header_value_token_set\"]), list(stats_b[\"header_value_token_set\"]))\n    serial = helpers[\"token_jaccard\"](list(stats_a[\"serial_token_set\"]), list(stats_b[\"serial_token_set\"]))\n    return helpers[\"clip01\"](0.6 * hv + 0.4 * serial)\n",
        },
        {
            "feature_name": "row_density_median_agreement",
            "task": "entity_matching",
            "scope": "row_pair",
            "version": "v1",
            "description": "Couple row density similarity with numeric median agreement to reward structured rows with consistent numeric scale.",
            "inputs_used": [
                "stats_a.nonempty_count",
                "stats_b.nonempty_count",
                "stats_a.numeric_median",
                "stats_b.numeric_median",
                "helpers.safe_ratio_float",
                "helpers.clip01",
            ],
            "fallback_value": 0.0,
            "range_hint": [0.0, 1.0],
            "example_based_on": [
                "row_nonempty_ratio",
                "row_numeric_rel_diff_sim",
            ],
            "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    density = helpers[\"safe_ratio_float\"](stats_a[\"nonempty_count\"], stats_b[\"nonempty_count\"])\n    med_a = float(stats_a[\"numeric_median\"])\n    med_b = float(stats_b[\"numeric_median\"])\n    denom = max(abs(med_a), abs(med_b), 1.0)\n    rel = abs(med_a - med_b) / float(denom)\n    numeric = 1.0 / (1.0 + rel)\n    return helpers[\"clip01\"](0.5 * density + 0.5 * numeric)\n",
        },
        {
            "feature_name": "row_numeric_set_overlap_bridge",
            "task": "entity_matching",
            "scope": "row_pair",
            "version": "v1",
            "description": "Blend numeric-set overlap with relative median similarity to reward rows that share numeric content and scale.",
            "inputs_used": [
                "stats_a.numeric_value_set",
                "stats_b.numeric_value_set",
                "stats_a.numeric_median",
                "stats_b.numeric_median",
                "helpers.numeric_overlap_max",
                "helpers.clip01",
            ],
            "fallback_value": 0.0,
            "range_hint": [0.0, 1.0],
            "example_based_on": [
                "row_numeric_value_overlap",
                "row_numeric_rel_diff_sim",
            ],
            "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    overlap = helpers[\"numeric_overlap_max\"](stats_a[\"numeric_value_set\"], stats_b[\"numeric_value_set\"])\n    med_a = float(stats_a[\"numeric_median\"])\n    med_b = float(stats_b[\"numeric_median\"])\n    denom = max(abs(med_a), abs(med_b), 1.0)\n    rel = abs(med_a - med_b) / float(denom)\n    median_sim = 1.0 / (1.0 + rel)\n    return helpers[\"clip01\"](0.55 * overlap + 0.45 * median_sim)\n",
        },
        {
            "feature_name": "row_serial_edit_density_hybrid",
            "task": "entity_matching",
            "scope": "row_pair",
            "version": "v1",
            "description": "Combine serialized-row edit similarity with row density agreement to capture near-matches under small formatting drift.",
            "inputs_used": [
                "stats_a.serial_text",
                "stats_b.serial_text",
                "stats_a.nonempty_count",
                "stats_b.nonempty_count",
                "helpers.normalized_edit_similarity",
                "helpers.safe_ratio_float",
                "helpers.clip01",
            ],
            "fallback_value": 0.0,
            "range_hint": [0.0, 1.0],
            "example_based_on": [
                "row_serial_edit_similarity",
                "row_nonempty_ratio",
            ],
            "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    edit_sim = helpers[\"normalized_edit_similarity\"](stats_a[\"serial_text\"], stats_b[\"serial_text\"])\n    density = helpers[\"safe_ratio_float\"](stats_a[\"nonempty_count\"], stats_b[\"nonempty_count\"])\n    return helpers[\"clip01\"](0.65 * edit_sim + 0.35 * density)\n",
        },
        {
            "feature_name": "row_header_exact_token_agreement",
            "task": "entity_matching",
            "scope": "row_pair",
            "version": "v1",
            "description": "Use shared-header exactness together with header token overlap to capture schema-consistent row matches.",
            "inputs_used": [
                "stats_a.header_to_value",
                "stats_b.header_to_value",
                "stats_a.header_token_set",
                "stats_b.header_token_set",
                "helpers.token_jaccard",
                "helpers.clip01",
            ],
            "fallback_value": 0.0,
            "range_hint": [0.0, 1.0],
            "example_based_on": [
                "row_header_value_exact_ratio",
                "row_header_jaccard",
            ],
            "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    map_a = stats_a[\"header_to_value\"]\n    map_b = stats_b[\"header_to_value\"]\n    shared = list(set(map_a.keys()).intersection(set(map_b.keys())))\n    exact = 0.0\n    if shared:\n        match_count = 0\n        for key in shared:\n            if str(map_a.get(key, \"\")) == str(map_b.get(key, \"\")):\n                match_count += 1\n        exact = float(match_count) / float(len(shared))\n    header_tok = helpers[\"token_jaccard\"](list(stats_a[\"header_token_set\"]), list(stats_b[\"header_token_set\"]))\n    return helpers[\"clip01\"](0.6 * exact + 0.4 * header_tok)\n",
        },
        {
            "feature_name": "row_chargram_token_bridge",
            "task": "entity_matching",
            "scope": "row_pair",
            "version": "v1",
            "description": "Bridge row token overlap with serialized 3-gram overlap to stabilize lexical evidence under spacing and ordering noise.",
            "inputs_used": [
                "stats_a.token_set",
                "stats_b.token_set",
                "stats_a.serial_text",
                "stats_b.serial_text",
                "helpers.token_jaccard",
                "helpers.char_ngram_jaccard",
                "helpers.clip01",
            ],
            "fallback_value": 0.0,
            "range_hint": [0.0, 1.0],
            "example_based_on": [
                "row_token_jaccard",
                "row_serial_char3_jaccard",
            ],
            "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    tok = helpers[\"token_jaccard\"](list(stats_a[\"token_set\"]), list(stats_b[\"token_set\"]))\n    char3 = helpers[\"char_ngram_jaccard\"](stats_a[\"serial_text\"], stats_b[\"serial_text\"], 3)\n    return helpers[\"clip01\"](0.5 * tok + 0.5 * char3)\n",
        },
        {
            "feature_name": "row_embedding_numeric_bridge",
            "task": "entity_matching",
            "scope": "row_pair",
            "version": "v1",
            "description": "Combine embedding similarity with numeric-scale agreement so semantic closeness is discounted when numeric signatures diverge strongly.",
            "inputs_used": [
                "emb_a",
                "emb_b",
                "stats_a.numeric_median",
                "stats_b.numeric_median",
                "helpers.cosine_similarity",
                "helpers.clip01",
            ],
            "fallback_value": 0.0,
            "range_hint": [0.0, 1.0],
            "example_based_on": [
                "row_emb_cosine",
                "row_numeric_rel_diff_sim",
            ],
            "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    emb_a = ctx[\"emb_a\"]\n    emb_b = ctx[\"emb_b\"]\n    helpers = ctx[\"helpers\"]\n    cosine = helpers[\"cosine_similarity\"](emb_a, emb_b)\n    cosine01 = helpers[\"clip01\"](0.5 * (cosine + 1.0))\n    med_a = float(stats_a[\"numeric_median\"])\n    med_b = float(stats_b[\"numeric_median\"])\n    denom = max(abs(med_a), abs(med_b), 1.0)\n    rel = abs(med_a - med_b) / float(denom)\n    numeric = 1.0 / (1.0 + rel)\n    return helpers[\"clip01\"](0.55 * cosine01 + 0.45 * numeric)\n",
        },
    ]


def _build_dry_run_features(*, num_features: int) -> List[GeneratedFeatureSpec]:
    docs = _build_dry_run_feature_docs()
    if int(num_features) > 0:
        docs = docs[: int(num_features)]
    return [validate_generated_feature_spec(doc) for doc in docs]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate EM feature functions (Python code) for 0428_featgen.")
    ap.add_argument("--output", type=str, required=True, help="Output JSON file for generated feature specs.")
    ap.add_argument(
        "--feature-request",
        type=str,
        default=(
            "Starting from a human-defined teacher atom pool and three human-selected exemplar atoms, generate smooth, "
            "interpretable row-pair atom features that complement the exemplars and help complete a compact EM pool."
        ),
    )
    ap.add_argument("--num-features", type=int, default=7)
    ap.add_argument("--target-total-pool-size", type=int, default=10)
    ap.add_argument("--examples-file", type=str, default=str(DEFAULT_EXAMPLES_FILE))
    ap.add_argument("--feature-cards-file", type=str, default=str(DEFAULT_FEATURE_CARDS_FILE))
    ap.add_argument(
        "--teacher-feature-pool",
        type=str,
        default=",".join(DEFAULT_TEACHER_FEATURE_POOL),
        help="Comma-separated human-defined teacher atom pool used as the background catalog.",
    )
    ap.add_argument(
        "--preserved-features",
        type=str,
        default=",".join(DEFAULT_PRESERVED_FEATURES),
        help="Comma-separated human-selected exemplar atom names used as examples and exclusions.",
    )
    ap.add_argument("--task-description", type=str, default=DEFAULT_TASK_DESCRIPTION)
    ap.add_argument("--data-description", type=str, default=DEFAULT_DATA_DESCRIPTION)
    ap.add_argument("--selection-notes", type=str, default=DEFAULT_SELECTION_NOTES)
    ap.add_argument("--dataset-context-json", type=str, default="")
    ap.add_argument("--model", type=str, default="gpt-5-mini")
    ap.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--api-key-file", type=str, default=str(DEFAULT_API_KEY_FILE))
    ap.add_argument("--api-key-label", type=str, default="")
    ap.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--reasoning-effort", type=str, default="low", choices=["", "low", "medium", "high"])
    ap.add_argument("--max-completion-tokens", type=int, default=2200)
    ap.add_argument("--dump-prompt", type=str, default="")
    ap.add_argument("--dump-response", type=str, default="")
    ap.add_argument("--dump-validation", type=str, default="")
    ap.add_argument("--dry-run", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    preserved_feature_names = _parse_csv(args.preserved_features)
    teacher_feature_pool = _parse_csv(args.teacher_feature_pool)
    example_path = Path(str(args.examples_file)).expanduser()
    card_path = Path(str(args.feature_cards_file)).expanduser()
    fixed_examples = _select_examples(
        _load_examples(example_path),
        preserved_feature_names=preserved_feature_names,
    )
    teacher_cards = _load_feature_cards(card_path, teacher_feature_pool)
    em_atomic_cards = _load_feature_cards(card_path, DEFAULT_TEACHER_FEATURE_POOL + ["row_header_jaccard", "row_header_value_exact_ratio", "row_header_token_jaccard"])
    forbidden_feature_names = sorted(set(teacher_feature_pool).union(set(em_atomic_cards.keys())))
    dataset_context = _load_dataset_context(str(args.dataset_context_json))
    discovered_api_key = ""
    if not str(args.api_key).strip() and str(args.api_key_file).strip():
        discovered_api_key = _discover_api_key(
            Path(str(args.api_key_file)).expanduser(),
            label=str(args.api_key_label),
        )
    prompt = _build_prompt(
        feature_request=str(args.feature_request),
        num_features=max(1, int(args.num_features)),
        target_total_pool_size=max(0, int(args.target_total_pool_size)),
        preserved_feature_names=preserved_feature_names,
        teacher_feature_pool=teacher_feature_pool,
        forbidden_feature_names=forbidden_feature_names,
        fixed_examples=fixed_examples,
        teacher_cards=teacher_cards,
        task_description=str(args.task_description),
        data_description=str(args.data_description),
        selection_notes=str(args.selection_notes),
        dataset_context=dataset_context,
    )

    if str(args.dump_prompt).strip():
        _write_json(str(args.dump_prompt), prompt)

    if int(args.dry_run) == 1:
        specs = _build_dry_run_features(num_features=max(1, int(args.num_features)))
        save_generated_feature_specs(specs, args.output)
        report = _build_validation_report(specs=specs, dry_run=True, output_path=str(args.output))
        if str(args.dump_response).strip():
            _write_json(str(args.dump_response), {"dry_run": True, "features": [spec.to_dict() for spec in specs]})
        if str(args.dump_validation).strip():
            _write_json(str(args.dump_validation), report)
        print(json.dumps(report, ensure_ascii=False))
        return 0

    raw = _call_llm(
        api_key=str(args.api_key).strip() or str(discovered_api_key).strip(),
        base_url=str(args.base_url),
        model=str(args.model),
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        timeout_sec=float(args.timeout_sec),
        reasoning_effort=str(args.reasoning_effort),
        max_completion_tokens=int(args.max_completion_tokens),
    )
    if str(args.dump_response).strip():
        response_path = Path(str(args.dump_response)).expanduser()
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(str(raw), encoding="utf-8")
    parsed = _extract_json_object(raw)
    features = parsed.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("Model output must contain non-empty features list.")
    seen_names = set()
    for item in features:
        if not isinstance(item, Mapping):
            raise ValueError("Each generated feature must be a JSON object.")
        feature_name = str(item.get("feature_name", "")).strip()
        if feature_name in forbidden_feature_names:
            raise ValueError(f"Generated feature name collides with forbidden existing atom: {feature_name}")
        if feature_name in seen_names:
            raise ValueError(f"Duplicate generated feature name in one response: {feature_name}")
        seen_names.add(feature_name)
    specs = [validate_generated_feature_spec(item) for item in features]
    save_generated_feature_specs(specs, args.output)
    report = _build_validation_report(specs=specs, dry_run=False, output_path=str(args.output))
    if str(args.dump_validation).strip():
        _write_json(str(args.dump_validation), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
