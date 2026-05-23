#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generated_feature_runtime import GeneratedFeatureSpec, save_generated_feature_specs, validate_generated_feature_spec  # noqa: E402
from task_featgen_config import TASK_FEATGEN_CONFIGS  # noqa: E402


DEFAULT_FEATURE_CARDS_FILE = _ROOT / "symbolic_feature_cards.json"
DEFAULT_API_KEY_FILE = _ROOT.parent / "0325_policy_pro" / "LIGHTNING_API_KEY.md"


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


def _task_config(task: str) -> Mapping[str, Any]:
    token = str(task).strip()
    if token not in TASK_FEATGEN_CONFIGS:
        raise ValueError(f"Unsupported task={task!r}; expected one of {sorted(TASK_FEATGEN_CONFIGS.keys())}")
    return TASK_FEATGEN_CONFIGS[token]


def _default_examples_file(task: str) -> Path:
    return _ROOT / "generated_feature_examples" / f"{str(task).strip()}_fixed_examples.json"


def _default_feature_request(task: str) -> str:
    scope = str(_task_config(task).get("task_scope", "pair")).replace("_", " ")
    return (
        "Starting from a few human-selected exemplar atoms plus the task and data descriptions, generate smooth, "
        f"interpretable {scope} atom features that complement the exemplars and help complete a compact "
        f"{str(task).strip()} feature pool."
    )


def _task_soft_guidance(task: str) -> str:
    token = str(task).strip()
    if token == "entity_matching":
        return (
            "EM-specific soft guidance:\n"
            "- Lean toward a balanced mix of serial/alignment evidence, token-salience or rare-token evidence, "
            "value-containment or coverage evidence, and a small amount of header-aware agreement when the data supports it.\n"
            "- Header-aware or aligned-value features are especially useful when they reveal agreement that plain "
            "bag-of-token overlap would miss, but they should complement rather than crowd out the rest of the pool.\n"
            "- Weighted rare-token matches, smooth containment-style scores, and edit-style alignment cues are good "
            "directions when they add a genuinely different perspective.\n"
            "- A strong generated set usually contains at least one smooth containment or coverage view and at least "
            "one header-aware or aligned-value view, while still leaving room for serial and token-salience signals.\n"
            "- Avoid spending too many slots on near-duplicate token-overlap formulas, on multiple header-only variants "
            "that say almost the same thing, or on weak length-only tweaks.\n\n"
        )
    return ""


def _sanitize_selection_notes(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return ""
    kept: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", raw):
        token = str(sentence).strip()
        if not token:
            continue
        lowered = token.lower()
        if "teacher pool" in lowered or "teacher atom" in lowered:
            continue
        kept.append(token)
    return " ".join(kept).strip()


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
                return str(match.group(1)).strip().strip("'\"")
    generic = re.findall(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{15,}", raw)
    return str(generic[-1]).strip() if generic else ""


def _discover_api_key(path: Path, *, label: str = "") -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _discover_api_key_from_text(text, label=label)


def _ctx_schema_text(task: str) -> str:
    task = str(task).strip()
    if task == "entity_matching":
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
  - helpers["weighted_jaccard"](tokens_a, tokens_b)
  - helpers["normalized_edit_similarity"](text_a, text_b)
  - helpers["char_ngram_jaccard"](text_a, text_b, n)
  - helpers["cosine_similarity"](vec_a, vec_b)
  - helpers["l1_similarity"](vec_a, vec_b)
  - helpers["numeric_overlap_max"](values_a, values_b)
  - helpers["clip01"](x)
  - helpers["set_jaccard"](set_a, set_b)
"""
    if task == "joinable_table_search":
        return """Available ctx structure for compute_feature(ctx):
- ctx["stats_a"], ctx["stats_b"]: column-level dictionaries
  keys include:
  - value_set: set[str]
  - values: list[str]
  - unique_ratio: float
  - numeric_ratio: float
  - avg_len: float
  - header_tokens: set[str]
  - header_text: str
- ctx["helpers"]: dict of safe helper callables
  - helpers["safe_ratio_float"](a, b)
  - helpers["token_jaccard"](tokens_a, tokens_b)
  - helpers["normalized_edit_similarity"](text_a, text_b)
  - helpers["clip01"](x)
  - helpers["set_jaccard"](set_a, set_b)
"""
    if task == "schema_matching":
        return """Available ctx structure for compute_feature(ctx):
- ctx["stats_a"], ctx["stats_b"]: column-level dictionaries
  keys include:
  - value_set: set[str]
  - unique_ratio: float
  - missing_ratio: float
  - numeric_ratio: float
  - avg_len: float
  - header_tokens: set[str]
  - header_text: str
- ctx["helpers"]: dict of safe helper callables
  - helpers["safe_ratio_float"](a, b)
  - helpers["token_jaccard"](tokens_a, tokens_b)
  - helpers["normalized_edit_similarity"](text_a, text_b)
  - helpers["clip01"](x)
  - helpers["set_jaccard"](set_a, set_b)
"""
    if task == "union_table_search":
        return """Available ctx structure for compute_feature(ctx):
- ctx["stats_a"], ctx["stats_b"]: table-level dictionaries
  keys include:
  - column_value_sets: list[set[str]]
  - header_tokens: set[str]
  - num_rows: int
  - num_cols: int
- ctx["helpers"]: dict of safe helper callables
  - helpers["safe_ratio_float"](a, b)
  - helpers["token_jaccard"](tokens_a, tokens_b)
  - helpers["normalized_edit_similarity"](text_a, text_b)
  - helpers["clip01"](x)
  - helpers["set_jaccard"](set_a, set_b)
"""
    raise ValueError(f"Unsupported task={task!r}")


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
    task: str,
    scope: str,
    feature_request: str,
    num_features: int,
    target_total_pool_size: int,
    preserved_feature_names: Sequence[str],
    protected_feature_names: Sequence[str],
    fixed_examples: Sequence[Mapping[str, Any]],
    task_description: str,
    data_description: str,
    selection_notes: str,
) -> Dict[str, str]:
    system = (
        "You design Python atom feature functions for table representation and matching tasks. "
        "Return JSON only with a top-level object {\"features\": [...]} and no markdown."
    )

    example_lines: List[str] = []
    for item in fixed_examples:
        feature_name = str(item.get("feature_name", "")).strip()
        desc = str(item.get("description", "")).strip()
        code = str(item.get("code", "")).strip()
        example_lines.append(f"- {feature_name}: {desc}\n```python\n{code}\n```")
    exemplar_count = len([name for name in preserved_feature_names if str(name).strip()])
    protected_names = [str(name).strip() for name in protected_feature_names if str(name).strip()]
    total_target = int(target_total_pool_size) if int(target_total_pool_size) > 0 else (exemplar_count + int(num_features))
    sanitized_selection_notes = _sanitize_selection_notes(selection_notes)

    user = (
        f"Task: {task}\n"
        f"Task scope: {scope}\n"
        f"Task description: {str(task_description).strip()}\n"
        f"Data description: {str(data_description).strip()}\n"
        f"Goal: generate {int(num_features)} new {scope} feature function(s).\n"
        f"Human-selected exemplar count: {exemplar_count}\n"
        f"Target completed pool size after generation: {total_target}\n"
        f"Feature request: {feature_request}\n"
        f"Selection notes: {sanitized_selection_notes}\n"
        f"Preserved fixed features already kept as human exemplars and should be treated as examples, not regenerated: "
        f"{json.dumps(list(preserved_feature_names), ensure_ascii=False)}\n"
        f"Protected atom names already reserved in the current pool and must not be reused: "
        f"{json.dumps(protected_names, ensure_ascii=False)}\n\n"
        "Hard constraints:\n"
        "1) The preserved exemplar atoms are human-defined; treat them as trusted examples of style and signal quality, not as outputs to regenerate.\n"
        "2) Each feature must not reuse one of the protected atom names already reserved in the current pool.\n"
        "3) New features should complement the exemplars and broaden the pool rather than duplicate the same evidence family.\n"
        "4) At least half of the generated features should use evidence families that are not identical to a preserved exemplar's single formula.\n"
        "5) Each feature must define Python code with exactly one function: def compute_feature(ctx):\n"
        "6) Do not import anything. Do not use file IO, network IO, eval, exec, globals, classes, decorators, or exceptions.\n"
        "7) Use only ctx fields and helper functions described below.\n"
        "8) Return a single numeric scalar convertible to float.\n"
        "9) Prefer smooth numeric features over brittle hard-threshold rules.\n"
        "10) Keep each function short and interpretable.\n"
        "11) Provide fields: feature_name, task, scope, version, description, inputs_used, code, fallback_value, range_hint, example_based_on.\n"
        f"12) task must be {task} and scope must be {scope}.\n"
        "13) range_hint should be omitted or be a finite [lo, hi].\n"
        "14) example_based_on must be a JSON array of strings, even when there is only one item.\n"
        "15) Every example_based_on item should name one or more preserved exemplars that most directly inspired the feature.\n"
        f"15.5) Return exactly {int(num_features)} feature objects in the features list, no more and no fewer.\n"
        "16) Allowed builtin calls are only: abs, min, max, range, float, int, len, sum, sorted, bool, list, set, tuple, str.\n"
        "17) Allowed non-helper methods are only: get, keys, intersection, union, append.\n"
        "18) Do not define nested functions, lambdas, local classes, or helper closures inside compute_feature.\n"
        "19) Avoid any other method call pattern.\n\n"
        "20) Prefer simple explicit loops and assignments. Avoid nested def, lambda, decorators, local helper functions, or multiple top-level functions.\n"
        "21) If you need repeated logic, duplicate a few lines instead of defining a nested helper.\n"
        "22) Prefer direct helpers[...] calls or `helpers = ctx[\"helpers\"]`; avoid more creative aliasing patterns.\n"
        "23) Prefer plain loops over comprehensions or fancy Python shortcuts when either version would work.\n"
        "24) Keep code compatible with a strict AST validator: do not use imports, with-statements, try/except, yield, async, class definitions, or while-loops.\n\n"
        "Exploration guidance:\n"
        "25) Treat the preserved exemplars as anchor points for style and signal quality, not as formulas to clone.\n"
        "26) Explore multiple angles around the exemplars: for example, directional vs symmetric comparisons, raw-token vs header-aware views, coarse serial patterns vs higher-precision alignment, and lexical overlap vs numeric/profile summaries when the task supports them.\n"
        "27) Across the generated set, aim for viewpoint diversity so that several features are not just tiny formula variations on the same underlying statistic.\n"
        "28) Prefer features that expose a genuinely different perspective over simple metric swaps like jaccard vs dice vs overlap coefficient on the same exact inputs.\n"
        "29) If two candidate ideas feel too similar, keep the clearer one and use the other slot to explore a new angle suggested by the exemplars and task/data descriptions.\n\n"
        "30) When exemplars suggest one strong local pattern, explore nearby but distinct views that remain smooth and interpretable instead of cloning that same pattern repeatedly.\n\n"
        + _task_soft_guidance(task)
        + _ctx_schema_text(task)
        + "\n"
        + "\nPreserved fixed-feature examples:\n"
        + "\n".join(example_lines)
        + "\n\nReturn JSON only."
    )
    return {"system": system, "user": user}


def _build_repair_prompt(
    *,
    base_prompt: Mapping[str, str],
    raw_text: str,
    error_text: str,
) -> Dict[str, str]:
    system = str(base_prompt.get("system", "")).strip()
    user = (
        str(base_prompt.get("user", "")).rstrip()
        + "\n\nValidation failed on your previous JSON output.\n"
        + "Rewrite the full JSON so every feature satisfies the validator.\n"
        + "Keep the same task and scope, keep the output schema unchanged, and preserve the good semantic ideas when possible.\n"
        + "Important repair guidance:\n"
        + "- Return exactly the requested number of feature objects in the features list; do not add extras and do not omit any.\n"
        + "- If the previous output had the wrong feature count, first fix the count before changing anything else.\n"
        + "- Define exactly one top-level function named compute_feature for each feature.\n"
        + "- Do not define nested functions.\n"
        + "- Use direct builtins or helpers[...] calls only.\n"
        + "- Prefer explicit loops over comprehensions if there is any doubt.\n"
        + "- Avoid unsupported clever syntax.\n"
        + f"\nValidator error:\n{str(error_text).strip()}\n"
        + f"\nPrevious output to repair:\n{str(raw_text).strip()}\n"
        + "\nReturn JSON only."
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


def _build_validation_report(
    *,
    specs: Sequence[GeneratedFeatureSpec],
    dry_run: bool,
    output_path: str,
    llm_call_records: Optional[Sequence[Mapping[str, Any]]] = None,
    wall_elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
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
    if llm_call_records is not None:
        call_records = [dict(item) for item in llm_call_records]
        report.update(
            {
                "llm_api_request_count": int(len(call_records)),
                "llm_api_success_count": int(sum(1 for rec in call_records if rec.get("status") == "ok")),
                "llm_api_error_count": int(sum(1 for rec in call_records if rec.get("status") != "ok")),
                "llm_prompt_tokens_total": int(sum(int(rec.get("prompt_tokens") or 0) for rec in call_records)),
                "llm_completion_tokens_total": int(sum(int(rec.get("completion_tokens") or 0) for rec in call_records)),
                "llm_total_tokens_total": int(sum(int(rec.get("total_tokens") or 0) for rec in call_records)),
                "llm_elapsed_sec_total": float(sum(float(rec.get("elapsed_sec") or 0.0) for rec in call_records)),
                "llm_calls": call_records,
            }
        )
    if wall_elapsed_sec is not None:
        report["wall_elapsed_sec"] = float(wall_elapsed_sec)
    return report


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
) -> Tuple[str, List[Dict[str, Any]]]:
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
    attempted_budgets: List[int] = []
    last_finish_reason = ""
    budgets = [
        token_budget,
        max(token_budget * 2, 4000),
        max(token_budget * 4, 8000),
        max(token_budget * 6, 12000),
    ]
    call_records: List[Dict[str, Any]] = []

    for budget in budgets:
        budget = max(1, int(budget))
        if budget in attempted_budgets:
            continue
        attempted_budgets.append(budget)
        started = time.perf_counter()
        usage_fields = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        try:
            resp = _make_request(budget)
            usage_fields = _extract_usage_fields(resp)
            if not resp.choices:
                raise RuntimeError("LLM returned empty choices.")
            choice = resp.choices[0]
            message = choice.message
            content = _extract_text(message).strip()
            elapsed_sec = float(time.perf_counter() - started)
            if content:
                call_records.append(
                    {
                        "candidate_model": str(model).strip(),
                        "status": "ok",
                        "attempted_budget": int(budget),
                        "elapsed_sec": elapsed_sec,
                        "prompt_tokens": usage_fields["prompt_tokens"],
                        "completion_tokens": usage_fields["completion_tokens"],
                        "total_tokens": usage_fields["total_tokens"],
                    }
                )
                return content, call_records
            last_finish_reason = str(getattr(choice, "finish_reason", "") or "").strip().lower()
            call_records.append(
                {
                    "candidate_model": str(model).strip(),
                    "status": "empty",
                    "attempted_budget": int(budget),
                    "finish_reason": last_finish_reason,
                    "elapsed_sec": elapsed_sec,
                    "prompt_tokens": usage_fields["prompt_tokens"],
                    "completion_tokens": usage_fields["completion_tokens"],
                    "total_tokens": usage_fields["total_tokens"],
                }
            )
            if last_finish_reason != "length":
                break
        except Exception as exc:  # noqa: BLE001
            elapsed_sec = float(time.perf_counter() - started)
            call_records.append(
                {
                    "candidate_model": str(model).strip(),
                    "status": "error",
                    "attempted_budget": int(budget),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": elapsed_sec,
                    "prompt_tokens": usage_fields["prompt_tokens"],
                    "completion_tokens": usage_fields["completion_tokens"],
                    "total_tokens": usage_fields["total_tokens"],
                }
            )
            raise

    if last_finish_reason == "length":
        raise RuntimeError(
            "LLM returned empty content after length retries "
            f"(attempted_budgets={attempted_budgets}, finish_reason='length')."
        )

    raise RuntimeError(
        "LLM returned empty content "
        f"(finish_reason={last_finish_reason!r}, attempted_budgets={attempted_budgets})."
    )


def _dry_run_docs_for_task(task: str) -> List[Dict[str, Any]]:
    task = str(task).strip()
    if task == "entity_matching":
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
                "example_based_on": ["row_serial_token_jaccard", "row_value_jaccard"],
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
                "example_based_on": ["row_nonempty_ratio", "row_numeric_rel_diff_sim"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    density = helpers[\"safe_ratio_float\"](stats_a[\"nonempty_count\"], stats_b[\"nonempty_count\"])\n    med_a = float(stats_a[\"numeric_median\"])\n    med_b = float(stats_b[\"numeric_median\"])\n    denom = max(abs(med_a), abs(med_b), 1.0)\n    rel = abs(med_a - med_b) / float(denom)\n    numeric = 1.0 / (1.0 + rel)\n    return helpers[\"clip01\"](0.5 * density + 0.5 * numeric)\n",
            },
            {
                "feature_name": "row_serial_edit_density_hybrid",
                "task": "entity_matching",
                "scope": "row_pair",
                "version": "v1",
                "description": "Combine serialized-row edit similarity with row density agreement to capture near-matches under formatting drift.",
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
                "example_based_on": ["row_serial_edit_similarity", "row_nonempty_ratio"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    edit_sim = helpers[\"normalized_edit_similarity\"](stats_a[\"serial_text\"], stats_b[\"serial_text\"])\n    density = helpers[\"safe_ratio_float\"](stats_a[\"nonempty_count\"], stats_b[\"nonempty_count\"])\n    return helpers[\"clip01\"](0.65 * edit_sim + 0.35 * density)\n",
            },
        ]
    if task == "joinable_table_search":
        return [
            {
                "feature_name": "jts_header_value_overlap_blend",
                "task": "joinable_table_search",
                "scope": "column_pair",
                "version": "v1",
                "description": "Blend header token similarity with column value-set overlap for schema-aware joinability evidence.",
                "inputs_used": [
                    "stats_a.header_tokens",
                    "stats_b.header_tokens",
                    "stats_a.value_set",
                    "stats_b.value_set",
                    "helpers.token_jaccard",
                    "helpers.set_jaccard",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["jaccard", "header_token_jaccard"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    header_sim = helpers[\"token_jaccard\"](list(stats_a[\"header_tokens\"]), list(stats_b[\"header_tokens\"]))\n    value_sim = helpers[\"set_jaccard\"](stats_a[\"value_set\"], stats_b[\"value_set\"])\n    return helpers[\"clip01\"](0.4 * header_sim + 0.6 * value_sim)\n",
            },
            {
                "feature_name": "jts_profile_balance_smooth",
                "task": "joinable_table_search",
                "scope": "column_pair",
                "version": "v1",
                "description": "Average profile-ratio agreement across uniqueness, numeric ratio, and average length.",
                "inputs_used": [
                    "stats_a.unique_ratio",
                    "stats_b.unique_ratio",
                    "stats_a.numeric_ratio",
                    "stats_b.numeric_ratio",
                    "stats_a.avg_len",
                    "stats_b.avg_len",
                    "helpers.safe_ratio_float",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["unique_ratio_sim", "numeric_ratio_sim", "avg_len_ratio_sim"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    u = helpers[\"safe_ratio_float\"](stats_a[\"unique_ratio\"], stats_b[\"unique_ratio\"])\n    n = helpers[\"safe_ratio_float\"](stats_a[\"numeric_ratio\"], stats_b[\"numeric_ratio\"])\n    l = helpers[\"safe_ratio_float\"](stats_a[\"avg_len\"], stats_b[\"avg_len\"])\n    return helpers[\"clip01\"]((u + n + l) / 3.0)\n",
            },
            {
                "feature_name": "jts_coverage_header_bridge",
                "task": "joinable_table_search",
                "scope": "column_pair",
                "version": "v1",
                "description": "Combine max directional coverage with normalized header edit similarity.",
                "inputs_used": [
                    "stats_a.value_set",
                    "stats_b.value_set",
                    "stats_a.header_text",
                    "stats_b.header_text",
                    "helpers.normalized_edit_similarity",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["coverage_max", "header_edit_similarity"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    values_a = stats_a[\"value_set\"]\n    values_b = stats_b[\"value_set\"]\n    inter = len(values_a.intersection(values_b))\n    cov_a = float(inter) / float(len(values_a)) if len(values_a) > 0 else 0.0\n    cov_b = float(inter) / float(len(values_b)) if len(values_b) > 0 else 0.0\n    coverage = max(cov_a, cov_b)\n    header_sim = helpers[\"normalized_edit_similarity\"](stats_a[\"header_text\"], stats_b[\"header_text\"])\n    return helpers[\"clip01\"](0.7 * coverage + 0.3 * header_sim)\n",
            },
        ]
    if task == "schema_matching":
        return [
            {
                "feature_name": "sm_header_value_blend",
                "task": "schema_matching",
                "scope": "column_pair",
                "version": "v1",
                "description": "Blend header token similarity with value-set overlap for schema alignment.",
                "inputs_used": [
                    "stats_a.header_tokens",
                    "stats_b.header_tokens",
                    "stats_a.value_set",
                    "stats_b.value_set",
                    "helpers.token_jaccard",
                    "helpers.set_jaccard",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["header_token_jaccard", "value_jaccard"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    header_sim = helpers[\"token_jaccard\"](list(stats_a[\"header_tokens\"]), list(stats_b[\"header_tokens\"]))\n    value_sim = helpers[\"set_jaccard\"](stats_a[\"value_set\"], stats_b[\"value_set\"])\n    return helpers[\"clip01\"](0.55 * header_sim + 0.45 * value_sim)\n",
            },
            {
                "feature_name": "sm_profile_balance_smooth",
                "task": "schema_matching",
                "scope": "column_pair",
                "version": "v1",
                "description": "Average agreement across uniqueness, missingness, numeric ratio, and average length.",
                "inputs_used": [
                    "stats_a.unique_ratio",
                    "stats_b.unique_ratio",
                    "stats_a.missing_ratio",
                    "stats_b.missing_ratio",
                    "stats_a.numeric_ratio",
                    "stats_b.numeric_ratio",
                    "stats_a.avg_len",
                    "stats_b.avg_len",
                    "helpers.safe_ratio_float",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["unique_ratio_sim", "missing_ratio_sim", "numeric_ratio_sim", "avg_len_ratio_sim"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    u = helpers[\"safe_ratio_float\"](stats_a[\"unique_ratio\"], stats_b[\"unique_ratio\"])\n    miss = helpers[\"safe_ratio_float\"](1.0 - stats_a[\"missing_ratio\"], 1.0 - stats_b[\"missing_ratio\"])\n    n = helpers[\"safe_ratio_float\"](stats_a[\"numeric_ratio\"], stats_b[\"numeric_ratio\"])\n    l = helpers[\"safe_ratio_float\"](stats_a[\"avg_len\"], stats_b[\"avg_len\"])\n    return helpers[\"clip01\"]((u + miss + n + l) / 4.0)\n",
            },
            {
                "feature_name": "sm_header_edit_missing_bridge",
                "task": "schema_matching",
                "scope": "column_pair",
                "version": "v1",
                "description": "Combine header edit similarity with observed-value coverage similarity.",
                "inputs_used": [
                    "stats_a.header_text",
                    "stats_b.header_text",
                    "stats_a.missing_ratio",
                    "stats_b.missing_ratio",
                    "helpers.normalized_edit_similarity",
                    "helpers.safe_ratio_float",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["header_edit_similarity", "missing_ratio_sim"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    header_sim = helpers[\"normalized_edit_similarity\"](stats_a[\"header_text\"], stats_b[\"header_text\"])\n    obs_sim = helpers[\"safe_ratio_float\"](1.0 - stats_a[\"missing_ratio\"], 1.0 - stats_b[\"missing_ratio\"])\n    return helpers[\"clip01\"](0.6 * header_sim + 0.4 * obs_sim)\n",
            },
        ]
    if task == "union_table_search":
        return [
            {
                "feature_name": "uts_header_size_blend",
                "task": "union_table_search",
                "scope": "table_pair",
                "version": "v1",
                "description": "Blend header-token overlap with row and column count compatibility.",
                "inputs_used": [
                    "stats_a.header_tokens",
                    "stats_b.header_tokens",
                    "stats_a.num_rows",
                    "stats_b.num_rows",
                    "stats_a.num_cols",
                    "stats_b.num_cols",
                    "helpers.token_jaccard",
                    "helpers.safe_ratio_float",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["header_jaccard", "col_count_ratio", "row_count_ratio"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    header_sim = helpers[\"token_jaccard\"](list(stats_a[\"header_tokens\"]), list(stats_b[\"header_tokens\"]))\n    col_sim = helpers[\"safe_ratio_float\"](stats_a[\"num_cols\"], stats_b[\"num_cols\"])\n    row_sim = helpers[\"safe_ratio_float\"](stats_a[\"num_rows\"], stats_b[\"num_rows\"])\n    return helpers[\"clip01\"]((header_sim + col_sim + row_sim) / 3.0)\n",
            },
            {
                "feature_name": "uts_overlap_size_bridge",
                "task": "union_table_search",
                "scope": "table_pair",
                "version": "v1",
                "description": "Combine best directional column-overlap mean with table-size compatibility.",
                "inputs_used": [
                    "stats_a.column_value_sets",
                    "stats_b.column_value_sets",
                    "stats_a.num_cols",
                    "stats_b.num_cols",
                    "stats_a.num_rows",
                    "stats_b.num_rows",
                    "helpers.set_jaccard",
                    "helpers.safe_ratio_float",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["col_overlap_a2b_mean", "col_count_ratio", "row_count_ratio"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    cols_a = stats_a[\"column_value_sets\"]\n    cols_b = stats_b[\"column_value_sets\"]\n    best_sum = 0.0\n    used = 0\n    for values_a in cols_a:\n        best = 0.0\n        for values_b in cols_b:\n            score = helpers[\"set_jaccard\"](values_a, values_b)\n            if score > best:\n                best = score\n        best_sum += best\n        used += 1\n    overlap = float(best_sum) / float(used) if used > 0 else 0.0\n    col_sim = helpers[\"safe_ratio_float\"](stats_a[\"num_cols\"], stats_b[\"num_cols\"])\n    row_sim = helpers[\"safe_ratio_float\"](stats_a[\"num_rows\"], stats_b[\"num_rows\"])\n    size_sim = 0.5 * col_sim + 0.5 * row_sim\n    return helpers[\"clip01\"](0.7 * overlap + 0.3 * size_sim)\n",
            },
            {
                "feature_name": "uts_bidirectional_header_bridge",
                "task": "union_table_search",
                "scope": "table_pair",
                "version": "v1",
                "description": "Blend bidirectional overlap balance with header compatibility.",
                "inputs_used": [
                    "stats_a.column_value_sets",
                    "stats_b.column_value_sets",
                    "stats_a.header_tokens",
                    "stats_b.header_tokens",
                    "helpers.set_jaccard",
                    "helpers.token_jaccard",
                    "helpers.clip01",
                ],
                "fallback_value": 0.0,
                "range_hint": [0.0, 1.0],
                "example_based_on": ["col_overlap_a2b_mean", "col_overlap_b2a_mean", "header_jaccard"],
                "code": "def compute_feature(ctx):\n    stats_a = ctx[\"stats_a\"]\n    stats_b = ctx[\"stats_b\"]\n    helpers = ctx[\"helpers\"]\n    cols_a = stats_a[\"column_value_sets\"]\n    cols_b = stats_b[\"column_value_sets\"]\n    sum_a = 0.0\n    cnt_a = 0\n    for values_a in cols_a:\n        best = 0.0\n        for values_b in cols_b:\n            score = helpers[\"set_jaccard\"](values_a, values_b)\n            if score > best:\n                best = score\n        sum_a += best\n        cnt_a += 1\n    sum_b = 0.0\n    cnt_b = 0\n    for values_b in cols_b:\n        best = 0.0\n        for values_a in cols_a:\n            score = helpers[\"set_jaccard\"](values_a, values_b)\n            if score > best:\n                best = score\n        sum_b += best\n        cnt_b += 1\n    a2b = float(sum_a) / float(cnt_a) if cnt_a > 0 else 0.0\n    b2a = float(sum_b) / float(cnt_b) if cnt_b > 0 else 0.0\n    balance = 1.0 - abs(a2b - b2a)\n    header_sim = helpers[\"token_jaccard\"](list(stats_a[\"header_tokens\"]), list(stats_b[\"header_tokens\"]))\n    return helpers[\"clip01\"](0.5 * balance + 0.5 * header_sim)\n",
            },
        ]
    raise ValueError(f"Unsupported task={task!r}")


def _build_dry_run_features(*, task: str, num_features: int) -> List[GeneratedFeatureSpec]:
    docs = _dry_run_docs_for_task(task)
    target = max(1, int(num_features))
    if target > len(docs):
        expanded: List[Dict[str, Any]] = []
        for idx in range(target):
            base = dict(docs[idx % len(docs)])
            if idx >= len(docs):
                base_name = str(base.get("feature_name", "")).strip()
                base["feature_name"] = f"{base_name}_dryrun{idx + 1:02d}"
                base["version"] = f"v1_dryrun{idx + 1:02d}"
                desc = str(base.get("description", "")).strip()
                base["description"] = f"{desc} Dry-run replica {idx + 1}."
            expanded.append(base)
        docs = expanded
    else:
        docs = docs[:target]
    return [validate_generated_feature_spec(doc, expected_task=str(task)) for doc in docs]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate task-aware feature functions (Python code) for 0428_featgen.")
    ap.add_argument("--task", type=str, required=True, choices=sorted(TASK_FEATGEN_CONFIGS.keys()))
    ap.add_argument("--output", type=str, required=True, help="Output JSON file for generated feature specs.")
    ap.add_argument("--feature-request", type=str, default="")
    ap.add_argument("--num-features", type=int, default=7)
    ap.add_argument("--target-total-pool-size", type=int, default=10)
    ap.add_argument("--examples-file", type=str, default="")
    ap.add_argument("--feature-cards-file", type=str, default=str(DEFAULT_FEATURE_CARDS_FILE))
    ap.add_argument("--teacher-feature-pool", type=str, default="")
    ap.add_argument(
        "--preserved-features",
        type=str,
        default="",
        help="Comma-separated human-selected exemplar atom names used as examples and exclusions.",
    )
    ap.add_argument(
        "--protected-feature-names",
        type=str,
        default="",
        help="Comma-separated atom names already reserved in the current pool and forbidden for generation.",
    )
    ap.add_argument("--task-description", type=str, default="")
    ap.add_argument("--data-description", type=str, default="")
    ap.add_argument("--selection-notes", type=str, default="")
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
    ap.add_argument("--summary-output", type=str, default="")
    ap.add_argument("--dry-run", type=int, default=0, choices=[0, 1])
    ap.add_argument("--max-repair-attempts", type=int, default=1)
    ap.add_argument("--allow-extra-features-truncate", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()
    started_total = time.perf_counter()

    task = str(args.task).strip()
    cfg = _task_config(task)
    scope = str(cfg.get("task_scope", "row_pair")).strip()
    preserved_feature_names = _parse_csv(args.preserved_features) or list(cfg.get("human_exemplar_atoms", []))
    protected_feature_names = _parse_csv(args.protected_feature_names) or list(preserved_feature_names)
    examples_file = str(args.examples_file).strip() or str(_default_examples_file(task))
    task_description = str(args.task_description).strip() or str(cfg.get("task_description", "")).strip()
    data_description = str(args.data_description).strip() or str(cfg.get("data_description", "")).strip()
    selection_notes = str(args.selection_notes).strip() or str(cfg.get("selection_notes", "")).strip()
    feature_request = str(args.feature_request).strip() or _default_feature_request(task)

    example_path = Path(examples_file).expanduser()
    fixed_examples = _select_examples(
        _load_examples(example_path),
        preserved_feature_names=preserved_feature_names,
    )
    discovered_api_key = ""
    if not str(args.api_key).strip() and str(args.api_key_file).strip():
        discovered_api_key = _discover_api_key(
            Path(str(args.api_key_file)).expanduser(),
            label=str(args.api_key_label),
        )
    prompt = _build_prompt(
        task=task,
        scope=scope,
        feature_request=feature_request,
        num_features=max(1, int(args.num_features)),
        target_total_pool_size=max(0, int(args.target_total_pool_size)),
        preserved_feature_names=preserved_feature_names,
        protected_feature_names=protected_feature_names,
        fixed_examples=fixed_examples,
        task_description=task_description,
        data_description=data_description,
        selection_notes=selection_notes,
    )

    if str(args.dump_prompt).strip():
        _write_json(str(args.dump_prompt), prompt)

    if int(args.dry_run) == 1:
        specs = _build_dry_run_features(task=task, num_features=max(1, int(args.num_features)))
        save_generated_feature_specs(specs, args.output)
        report = _build_validation_report(
            specs=specs,
            dry_run=True,
            output_path=str(args.output),
            llm_call_records=[],
            wall_elapsed_sec=float(time.perf_counter() - started_total),
        )
        if str(args.dump_response).strip():
            _write_json(str(args.dump_response), {"dry_run": True, "features": [spec.to_dict() for spec in specs]})
        if str(args.dump_validation).strip():
            _write_json(str(args.dump_validation), report)
        if str(args.summary_output).strip():
            _write_json(str(args.summary_output), report)
        print(json.dumps(report, ensure_ascii=False))
        return 0

    api_key = str(args.api_key).strip() or str(discovered_api_key).strip()
    active_prompt = dict(prompt)
    raw = ""
    last_error = ""
    specs: List[GeneratedFeatureSpec] = []
    all_llm_call_records: List[Dict[str, Any]] = []
    for attempt in range(max(1, int(args.max_repair_attempts)) + 1):
        raw, llm_call_records = _call_llm(
            api_key=api_key,
            base_url=str(args.base_url),
            model=str(args.model),
            system_prompt=active_prompt["system"],
            user_prompt=active_prompt["user"],
            timeout_sec=float(args.timeout_sec),
            reasoning_effort=str(args.reasoning_effort),
            max_completion_tokens=int(args.max_completion_tokens),
        )
        all_llm_call_records.extend(llm_call_records)
        try:
            parsed = _extract_json_object(raw)
            features = parsed.get("features", [])
            if not isinstance(features, list) or not features:
                raise ValueError("Model output must contain non-empty features list.")
            requested_num_features = max(1, int(args.num_features))
            if len(features) > requested_num_features and int(args.allow_extra_features_truncate) == 1:
                features = list(features[:requested_num_features])
            if len(features) != requested_num_features:
                raise ValueError(
                    f"Model output must contain exactly {requested_num_features} features; got {len(features)}."
                )
            seen_names = set()
            for item in features:
                if not isinstance(item, Mapping):
                    raise ValueError("Each generated feature must be a JSON object.")
                feature_name = str(item.get("feature_name", "")).strip()
                if feature_name in protected_feature_names:
                    raise ValueError(f"Generated feature name collides with protected pool atom: {feature_name}")
                if feature_name in seen_names:
                    raise ValueError(f"Duplicate generated feature name in one response: {feature_name}")
                seen_names.add(feature_name)
            specs = [validate_generated_feature_spec(item, expected_task=task, expected_scope=scope) for item in features]
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max(1, int(args.max_repair_attempts)):
                if str(args.dump_response).strip():
                    response_path = Path(str(args.dump_response)).expanduser()
                    response_path.parent.mkdir(parents=True, exist_ok=True)
                    response_path.write_text(str(raw), encoding="utf-8")
                raise
            active_prompt = _build_repair_prompt(base_prompt=prompt, raw_text=raw, error_text=last_error)

    if str(args.dump_response).strip():
        response_path = Path(str(args.dump_response)).expanduser()
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(str(raw), encoding="utf-8")
    save_generated_feature_specs(specs, args.output)
    report = _build_validation_report(
        specs=specs,
        dry_run=False,
        output_path=str(args.output),
        llm_call_records=all_llm_call_records,
        wall_elapsed_sec=float(time.perf_counter() - started_total),
    )
    if str(args.dump_validation).strip():
        _write_json(str(args.dump_validation), report)
    if str(args.summary_output).strip():
        _write_json(str(args.summary_output), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
