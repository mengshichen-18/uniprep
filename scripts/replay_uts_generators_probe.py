#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_task_feature_functions_gpt5 as atom_mod  # noqa: E402
from generated_feature_runtime import save_generated_feature_specs, validate_generated_feature_spec  # noqa: E402


PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")
API_KEY_FILE = ROOT.parent / "0325_policy_pro" / "LIGHTNING_API_KEY.md"
API_KEY_LABEL = "0428_KEY"
ATOM_MODEL = "gpt-5"
ATOM_REASONING_EFFORT = "low"
ATOM_MAX_COMPLETION_TOKENS = 2200
ATOM_TIMEOUT_SEC = 120.0
ATOM_MAX_REPAIR_ATTEMPTS = 2

SOURCE_ROOT = ROOT / "outputs" / "featgen_main14_seed5_gpu0" / "seed0" / "union_table_search"
DATASETS = ["wikidbs", "santos_benchmark", "magellan"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _extract_symbolic_cmd(log_path: Path) -> List[str]:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("[RUN] "):
            return shlex.split(line[len("[RUN] ") :].strip())
    raise RuntimeError(f"symbolic command not found in log: {log_path}")


def _call_atom_once(
    *,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    timeout_sec: float,
    reasoning_effort: str,
    request_records: List[Dict[str, Any]],
    live_log_path: Path,
) -> str:
    from openai import OpenAI

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if str(base_url).strip():
        kwargs["base_url"] = str(base_url).strip()
    client = OpenAI(**kwargs)

    def _extract_text(message: Any) -> str:
        direct = getattr(message, "content", "")
        if isinstance(direct, str):
            return direct
        if isinstance(direct, list):
            out: List[str] = []
            for item in direct:
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

    def _make_request(token_budget: int) -> str:
        request_kwargs: Dict[str, Any] = {
            "model": ATOM_MODEL,
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

        print(
            json.dumps(
                {
                    "phase": "atom_request_start",
                    "budget": int(token_budget),
                    "model": ATOM_MODEL,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        started = time.perf_counter()
        resp = client.chat.completions.create(**request_kwargs)
        ended = time.perf_counter()
        choice = resp.choices[0] if resp.choices else None
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        content = _extract_text(choice.message if choice else None).strip() if choice else ""
        usage = getattr(resp, "usage", None)
        record = {
            "budget": int(token_budget),
            "elapsed_sec": ended - started,
            "finish_reason": finish_reason,
            "content_len": len(content),
        }
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, key, None)
                if value is not None:
                    record[key] = int(value)
        request_records.append(record)
        _append_jsonl(live_log_path, {"kind": "atom_request_record", **record})
        print(json.dumps({"phase": "atom_request_done", **record}, ensure_ascii=False), flush=True)
        return content

    token_budget = max(1, int(max_completion_tokens))
    attempted_budgets: List[int] = []
    last_finish_reason = ""
    budgets = [
        token_budget,
        max(token_budget * 2, 4000),
        max(token_budget * 4, 8000),
        max(token_budget * 6, 12000),
    ]
    for budget in budgets:
        budget = max(1, int(budget))
        if budget in attempted_budgets:
            continue
        attempted_budgets.append(budget)
        content = _make_request(budget)
        if content:
            return content
        last_finish_reason = str(request_records[-1].get("finish_reason", "") or "").strip().lower()
        if last_finish_reason != "length":
            break

    if last_finish_reason == "length":
        raise RuntimeError(
            "LLM returned empty content after length retries "
            f"(attempted_budgets={attempted_budgets}, finish_reason='length')."
        )
    raise RuntimeError(
        "LLM returned empty content "
        f"(finish_reason={last_finish_reason!r}, attempted_budgets={attempted_budgets})."
    )


def _run_atom_probe(dataset: str, out_dir: Path) -> Dict[str, Any]:
    src_dir = SOURCE_ROOT / dataset
    manifest = _load_json(src_dir / f"union_table_search__{dataset}__generation_manifest.json")
    prompt = _load_json(src_dir / f"union_table_search__{dataset}__atom_prompt.json")

    api_key = atom_mod._discover_api_key(API_KEY_FILE, label=API_KEY_LABEL)
    if not api_key:
        raise RuntimeError(f"API key not found in {API_KEY_FILE}")

    protected = list(manifest.get("generator_exemplars", []))
    num_features = int(manifest["num_generated_features"])

    active_prompt = dict(prompt)
    raw = ""
    last_error = ""
    specs = []
    request_records: List[Dict[str, Any]] = []
    attempt_records: List[Dict[str, Any]] = []
    live_log_path = out_dir / f"union_table_search__{dataset}__atom_request_records.jsonl"
    started_total = time.perf_counter()
    for attempt in range(max(1, ATOM_MAX_REPAIR_ATTEMPTS) + 1):
        print(json.dumps({"phase": "atom_attempt_start", "dataset": dataset, "attempt_index": attempt + 1}, ensure_ascii=False), flush=True)
        req_start_index = len(request_records)
        raw = _call_atom_once(
            api_key=api_key,
            base_url="",
            system_prompt=active_prompt["system"],
            user_prompt=active_prompt["user"],
            max_completion_tokens=ATOM_MAX_COMPLETION_TOKENS,
            timeout_sec=ATOM_TIMEOUT_SEC,
            reasoning_effort=ATOM_REASONING_EFFORT,
            request_records=request_records,
            live_log_path=live_log_path,
        )
        current_requests = request_records[req_start_index:]
        try:
            parsed = atom_mod._extract_json_object(raw)
            features = parsed.get("features", [])
            if not isinstance(features, list) or not features:
                raise ValueError("Model output must contain non-empty features list.")
            if len(features) != num_features:
                raise ValueError(f"Model output must contain exactly {num_features} features; got {len(features)}.")
            seen_names = set()
            for item in features:
                if not isinstance(item, Mapping):
                    raise ValueError("Each generated feature must be a JSON object.")
                feature_name = str(item.get("feature_name", "")).strip()
                if feature_name in protected:
                    raise ValueError(f"Generated feature name collides with protected pool atom: {feature_name}")
                if feature_name in seen_names:
                    raise ValueError(f"Duplicate generated feature name in one response: {feature_name}")
                seen_names.add(feature_name)
            specs = [
                validate_generated_feature_spec(item, expected_task="union_table_search", expected_scope="table_pair")
                for item in features
            ]
            attempt_records.append(
                {
                    "attempt_index": attempt + 1,
                    "request_count": len(current_requests),
                    "elapsed_sec": sum(float(item["elapsed_sec"]) for item in current_requests),
                    "status": "success",
                }
            )
            print(
                json.dumps(
                    {"phase": "atom_attempt_success", "dataset": dataset, "attempt_index": attempt + 1},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            attempt_records.append(
                {
                    "attempt_index": attempt + 1,
                    "request_count": len(current_requests),
                    "elapsed_sec": sum(float(item["elapsed_sec"]) for item in current_requests),
                    "status": "repair",
                    "error": last_error,
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "atom_attempt_repair",
                        "dataset": dataset,
                        "attempt_index": attempt + 1,
                        "error": last_error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if attempt >= max(1, ATOM_MAX_REPAIR_ATTEMPTS):
                raise
            active_prompt = atom_mod._build_repair_prompt(base_prompt=prompt, raw_text=raw, error_text=last_error)
    ended_total = time.perf_counter()

    generated_path = out_dir / f"union_table_search__{dataset}__generated_atoms.json"
    validation_path = out_dir / f"union_table_search__{dataset}__atom_validation.json"
    response_path = out_dir / f"union_table_search__{dataset}__atom_response.txt"
    prompt_path = out_dir / f"union_table_search__{dataset}__atom_prompt.json"
    summary_path = out_dir / f"union_table_search__{dataset}__atom_probe_summary.json"

    save_generated_feature_specs(specs, str(generated_path))
    validation_report = atom_mod._build_validation_report(specs=specs, dry_run=False, output_path=str(generated_path))
    _write_json(validation_path, validation_report)
    response_path.write_text(str(raw), encoding="utf-8")
    _write_json(prompt_path, prompt)

    summary = {
        "dataset": dataset,
        "task": "union_table_search",
        "model": ATOM_MODEL,
        "reasoning_effort": ATOM_REASONING_EFFORT,
        "max_completion_tokens": ATOM_MAX_COMPLETION_TOKENS,
        "max_repair_attempts": ATOM_MAX_REPAIR_ATTEMPTS,
        "feature_count_expected": num_features,
        "feature_count_returned": len(specs),
        "total_elapsed_sec": ended_total - started_total,
        "llm_api_request_count": len(request_records),
        "repair_attempts_used": len(attempt_records),
        "had_repair": any(item["status"] == "repair" for item in attempt_records[:-1]) or (
            attempt_records and attempt_records[-1]["status"] == "repair"
        ),
        "request_records": request_records,
        "attempt_records": attempt_records,
        "last_error": last_error,
        "artifacts": {
            "generated_atoms": str(generated_path),
            "validation": str(validation_path),
            "response": str(response_path),
            "prompt": str(prompt_path),
        },
    }
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _run_symbolic_probe(dataset: str, out_dir: Path) -> Dict[str, Any]:
    log_path = ROOT / "outputs" / "tmux_logs" / "featgen_main14_seed5_gpu0" / f"featgen_main14_seed5_gpu0_s0_uts_{dataset}.log"
    cmd = _extract_symbolic_cmd(log_path)

    repl = {
        str(SOURCE_ROOT / dataset / f"union_table_search__{dataset}__symbolic_auto_min14.json"): str(
            out_dir / f"union_table_search__{dataset}__symbolic_auto_min14.json"
        ),
        str(SOURCE_ROOT / dataset / f"union_table_search__{dataset}__symbolic_summary.json"): str(
            out_dir / f"union_table_search__{dataset}__symbolic_summary.json"
        ),
        str(SOURCE_ROOT / dataset / f"union_table_search__{dataset}__symbolic_prompt.json"): str(
            out_dir / f"union_table_search__{dataset}__symbolic_prompt.json"
        ),
    }
    patched_cmd = [repl.get(token, token) for token in cmd]

    print(json.dumps({"phase": "symbolic_start", "dataset": dataset}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    proc = subprocess.run(patched_cmd, capture_output=True, text=True, cwd=str(ROOT))
    ended = time.perf_counter()

    stdout_path = out_dir / f"union_table_search__{dataset}__symbolic_stdout.log"
    stderr_path = out_dir / f"union_table_search__{dataset}__symbolic_stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"symbolic replay failed for {dataset} with code {proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )

    summary_path = out_dir / f"union_table_search__{dataset}__symbolic_summary.json"
    summary = _load_json(summary_path)
    probe_summary = {
        "dataset": dataset,
        "wall_elapsed_sec": ended - started,
        "returncode": proc.returncode,
        "llm_api_request_count": summary.get("llm_api_request_count"),
        "repair_attempts_used": summary.get("repair_attempts_used"),
        "llm_elapsed_sec_total": summary.get("llm_elapsed_sec_total"),
        "summary_path": str(summary_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    print(json.dumps({"phase": "symbolic_done", "dataset": dataset, "wall_elapsed_sec": ended - started}, ensure_ascii=False), flush=True)
    _write_json(out_dir / f"union_table_search__{dataset}__symbolic_probe_summary.json", probe_summary)
    return probe_summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay UTS atom+symbolic generation and record request timing.")
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]

    overall: Dict[str, Any] = {"task": "union_table_search", "datasets": {}}
    for dataset in datasets:
        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        atom_summary = _run_atom_probe(dataset, dataset_dir)
        symbolic_summary = _run_symbolic_probe(dataset, dataset_dir)
        overall["datasets"][dataset] = {
            "atom": atom_summary,
            "symbolic": symbolic_summary,
        }
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "atom_requests": atom_summary["llm_api_request_count"],
                    "atom_total_elapsed_sec": atom_summary["total_elapsed_sec"],
                    "symbolic_requests": symbolic_summary["llm_api_request_count"],
                    "symbolic_llm_elapsed_sec_total": symbolic_summary["llm_elapsed_sec_total"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    _write_json(output_root / "overall_probe_summary.json", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
