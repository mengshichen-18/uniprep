import argparse
import gc
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Sequence

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(1, _REPO_ROOT)

from ajoint_graph_data import HierGraph
from ajoint_trainer import GraphLinkPredictionTrainer
from generated_feature_runtime import load_generated_feature_registry
from runtime_profile import close_runtime_profiler, init_runtime_profiler, profile_phase, record_profile_event
from v3c_spec import (
    GLOBAL_V3C_CONFIG,
    PAIR_FEATURE_DEFAULTS,
    REFERENCE_RUN_LOGS,
    TASK_PERMUTATIONS,
    task_v3c_config,
)


LOG_DIR = "./logs/0325_policy_pro_train"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FNAME = os.path.join(LOG_DIR, f"ajoint_0325_policy_pro_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FNAME, "w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def _log_json(tag: str, payload: Dict) -> None:
    try:
        logger.info("%s %s", tag, json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        logger.info("%s %s", tag, payload)


def _set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _configure_determinism(seed: int, *, deterministic: bool) -> None:
    _set_global_seed(seed)
    if not deterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        logger.warning("[Deterministic] torch.use_deterministic_algorithms is unavailable; continuing without it.")


def _parse_feature_list(raw: str) -> List[str]:
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    return [item for item in tokens if item != "none"]


def _append_unique(base: Sequence[str], extra: Sequence[str]) -> List[str]:
    out: List[str] = [str(item) for item in base]
    seen = set(out)
    for item in extra:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _load_generated_feature_names(specs_path: str, *, expected_task: str, expected_scope: str) -> List[str]:
    path = str(specs_path).strip()
    if not path:
        return []
    registry = load_generated_feature_registry(
        path,
        expected_task=str(expected_task),
        expected_scope=str(expected_scope),
    )
    return list(registry.feature_names)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen v3c runner: task-specific structured correction on top of 0308_base_em."
    )
    parser.add_argument("--dataset", type=str, default="wikidbs", help="Dataset name.")
    parser.add_argument(
        "--graph_data_dir",
        type=str,
        default="",
        help="Optional absolute/relative graph data directory. If set, overrides dataset+split auto path.",
    )
    parser.add_argument(
        "--dataset_split_tag",
        type=str,
        default="no_token",
        choices=["no_token", "040303_no_token"],
        help="Used only when --graph_data_dir is not set. Final path: ./data/{dataset}_{dataset_split_tag}",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional run tag used for replay archive names. Defaults to profile_run_name when empty.",
    )
    parser.add_argument(
        "--replay_archive_root",
        type=str,
        default="",
        help="Optional archive root for per-run replay/scores npz outputs.",
    )
    parser.add_argument("--task_permutation", type=int, default=0, help="Only affects task run order.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Early stopping patience for validation F1.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=192, help="Batch size.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension.")
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="10,5",
        help="Comma-separated neighbor sampling sizes per GNN layer (e.g., 10,5).",
    )
    parser.add_argument("--gnn_layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--gnn_type", type=str, default="our", choices=["gcn", "gat", "sage", "our"])
    parser.add_argument("--device", type=str, default="cuda:0", help="Device name.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--prepool_cells", type=int, default=0, choices=[0, 1])
    parser.add_argument("--prepool_row_alpha", type=float, default=0.0)
    parser.add_argument("--prepool_col_alpha", type=float, default=0.0)
    parser.add_argument("--prepool_rows", type=int, default=0, choices=[0, 1])
    parser.add_argument("--prepool_cols", type=int, default=0, choices=[0, 1])
    parser.add_argument("--prepool_chunk_size", type=int, default=16384)
    parser.add_argument("--drop_cell_edges", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--em_drop_cell_edges",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Task-specific override for EM drop_cell_edges. -1 means use global --drop_cell_edges.",
    )
    parser.add_argument(
        "--jts_drop_cell_edges",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Task-specific override for JTS drop_cell_edges. -1 means use global --drop_cell_edges.",
    )
    parser.add_argument(
        "--uts_drop_cell_edges",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Task-specific override for UTS drop_cell_edges. -1 means use global --drop_cell_edges.",
    )
    parser.add_argument(
        "--sm_drop_cell_edges",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Task-specific override for SM drop_cell_edges. -1 means use global --drop_cell_edges.",
    )
    parser.add_argument(
        "--em_graph_data_dir",
        type=str,
        default="",
        help="Optional override graph data dir for EM supervision splits. Empty uses base dataset graph dir.",
    )
    parser.add_argument(
        "--jts_graph_data_dir",
        type=str,
        default="",
        help="Optional override graph data dir for JTS supervision splits. Empty uses base dataset graph dir.",
    )
    parser.add_argument(
        "--uts_graph_data_dir",
        type=str,
        default="",
        help="Optional override graph data dir for UTS supervision splits. Empty uses base dataset graph dir.",
    )
    parser.add_argument(
        "--sm_graph_data_dir",
        type=str,
        default="",
        help="Optional override graph data dir for SM supervision splits. Empty uses base dataset graph dir.",
    )
    parser.add_argument(
        "--em_pair_features",
        type=str,
        default=",".join(PAIR_FEATURE_DEFAULTS["entity_matching"]),
        help="[Legacy name] EM symbolic source features (groups/atoms). Prefer --em_symbolic_source_features.",
    )
    parser.add_argument(
        "--em_symbolic_source_features",
        type=str,
        default="",
        help="EM symbolic-source features (groups/atoms) used to compute symbolic channels.",
    )
    parser.add_argument(
        "--em_decoder_pair_features",
        type=str,
        default="",
        help="[Legacy name] Optional EM decoder static subset. Prefer --em_decoder_static_features.",
    )
    parser.add_argument(
        "--em_decoder_static_features",
        type=str,
        default="",
        help="EM decoder static features appended to MLP base input. Empty => same as source (coupled) unless decouple mode enforces empty.",
    )
    parser.add_argument("--em_table_root", type=str, default="")
    parser.add_argument(
        "--em_auto_pos_weight",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, auto-compute EM BCE pos_weight from train split (neg/pos, capped).",
    )
    parser.add_argument(
        "--em_pos_weight_cap",
        type=float,
        default=4.0,
        help="Upper bound for auto EM pos_weight; <=0 disables capping.",
    )
    parser.add_argument(
        "--em_use_interactions",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, enable EM interaction block concat(u,v,|u-v|,u*v) in edge encoder.",
    )
    parser.add_argument(
        "--em_loss",
        type=str,
        default="bce",
        choices=["bce", "focal", "bce_contrastive"],
        help="EM training loss: BCE, focal BCE, or BCE + contrastive auxiliary loss.",
    )
    parser.add_argument(
        "--em_focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma for --em_loss=focal.",
    )
    parser.add_argument(
        "--em_focal_alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha for --em_loss=focal.",
    )
    parser.add_argument(
        "--em_contrastive_weight",
        type=float,
        default=0.0,
        help="Auxiliary contrastive loss weight for --em_loss=bce_contrastive.",
    )
    parser.add_argument(
        "--em_pair_feat_norm",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, fit EM pair-feature normalization stats on train split and reuse on val/test.",
    )
    parser.add_argument(
        "--online_symbolic_spec_path",
        type=str,
        default="",
        help="Optional symbolic spec path for current task online symbolic augmentation.",
    )
    parser.add_argument(
        "--online_symbolic_spec_template",
        type=str,
        default="",
        help="Optional template for per-task spec path, e.g. '/.../c12/{task}/cand_01.json'. Supports {task},{dataset}.",
    )
    parser.add_argument(
        "--online_symbolic_repr",
        type=str,
        default="auto",
        choices=["auto", "concat", "aggregation"],
        help="How online symbolic features are represented before appending to decoder input.",
    )
    parser.add_argument(
        "--online_symbolic_normalize",
        type=str,
        default="none",
        choices=["none", "zscore"],
        help="Optional train-only normalization for online symbolic features.",
    )
    parser.add_argument(
        "--online_symbolic_tile_repeat",
        type=int,
        default=1,
        help="Repeat count for online symbolic channels before appending to decoder input.",
    )
    parser.add_argument(
        "--em_decoder_width_mult",
        type=float,
        default=1.0,
        help="Width multiplier for EM decoder head. 1.0 keeps current width.",
    )
    parser.add_argument(
        "--em_hard_neg_ratio",
        type=float,
        default=0.0,
        help="Hard negative ratio (vs #positive train edges) for EM epoch-wise mining.",
    )
    parser.add_argument(
        "--em_hard_neg_warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs before enabling EM hard-negative mining.",
    )
    parser.add_argument(
        "--em_row_stats_mode",
        type=str,
        default="full",
        choices=["required", "full"],
        help="EM row-stat loading mode: required (only rows referenced by EM supervision) or full (load full tables).",
    )
    parser.add_argument(
        "--em_pair_cache_mode",
        type=str,
        default="readwrite",
        choices=["off", "readwrite"],
        help="EM split feature cache mode: off disables disk cache, readwrite enables load/save reuse.",
    )
    parser.add_argument(
        "--em_pair_cache_root",
        type=str,
        default="",
        help="Optional root for EM pair-feature cache files.",
    )
    parser.add_argument(
        "--em_generated_feature_specs_path",
        type=str,
        default="",
        help="Optional JSON file or directory of LLM-generated EM feature functions to append to fixed EM features.",
    )
    parser.add_argument(
        "--jts_generated_feature_specs_path",
        type=str,
        default="",
        help="Optional JSON file or directory of LLM-generated JTS feature functions to append to fixed JTS features.",
    )
    parser.add_argument(
        "--sm_generated_feature_specs_path",
        type=str,
        default="",
        help="Optional JSON file or directory of LLM-generated SM feature functions to append to fixed SM features.",
    )
    parser.add_argument(
        "--uts_generated_feature_specs_path",
        type=str,
        default="",
        help="Optional JSON file or directory of LLM-generated UTS feature functions to append to fixed UTS features.",
    )
    parser.add_argument(
        "--debug_max_train_edges",
        type=int,
        default=0,
        help="Optional debug cap for the number of train supervision edges kept after loading.",
    )
    parser.add_argument(
        "--debug_max_val_edges",
        type=int,
        default=0,
        help="Optional debug cap for the number of validation supervision edges kept after loading.",
    )
    parser.add_argument(
        "--debug_max_test_edges",
        type=int,
        default=0,
        help="Optional debug cap for the number of test supervision edges kept after loading.",
    )
    parser.add_argument(
        "--jts_pair_features",
        type=str,
        default=",".join(PAIR_FEATURE_DEFAULTS["joinable_table_search"]),
        help="[Legacy name] JTS symbolic source features. Prefer --jts_symbolic_source_features.",
    )
    parser.add_argument(
        "--jts_symbolic_source_features",
        type=str,
        default="",
        help="JTS symbolic-source features (groups/atoms).",
    )
    parser.add_argument(
        "--jts_decoder_static_features",
        type=str,
        default="",
        help="JTS decoder static features. Empty => same as source (coupled).",
    )
    parser.add_argument("--jts_table_root", type=str, default="")
    parser.add_argument(
        "--uts_pair_features",
        type=str,
        default=",".join(PAIR_FEATURE_DEFAULTS["union_table_search"]),
        help="[Legacy name] UTS symbolic source features. Prefer --uts_symbolic_source_features.",
    )
    parser.add_argument(
        "--uts_symbolic_source_features",
        type=str,
        default="",
        help="UTS symbolic-source features (groups/atoms).",
    )
    parser.add_argument(
        "--uts_decoder_static_features",
        type=str,
        default="",
        help="UTS decoder static features. Empty => same as source (coupled).",
    )
    parser.add_argument("--uts_table_root", type=str, default="")
    parser.add_argument(
        "--sm_pair_features",
        type=str,
        default=",".join(PAIR_FEATURE_DEFAULTS["schema_matching"]),
        help="[Legacy name] SM symbolic source features. Prefer --sm_symbolic_source_features.",
    )
    parser.add_argument(
        "--sm_symbolic_source_features",
        type=str,
        default="",
        help="SM symbolic-source features (groups/atoms).",
    )
    parser.add_argument(
        "--sm_decoder_static_features",
        type=str,
        default="",
        help="SM decoder static features. Empty => same as source (coupled).",
    )
    parser.add_argument("--sm_table_root", type=str, default="")
    parser.add_argument(
        "--feature_wiring_mode",
        type=str,
        default="decoupled",
        choices=["decoupled", "coupled"],
        help="Unified feature wiring switch: 'decoupled' uses decoder static lists; 'coupled' mirrors symbolic source lists.",
    )
    parser.add_argument(
        "--allow_empty_decoder_static_features",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1 in decoupled mode, keep empty decoder static feature lists as empty (no auto-fallback to source features).",
    )
    parser.add_argument("--profile_runtime", type=int, default=0, choices=[0, 1])
    parser.add_argument("--profile_dir", type=str, default="logs/0318_v3c/profile")
    parser.add_argument("--profile_interval_sec", type=float, default=0.5)
    parser.add_argument("--profile_run_name", type=str, default="")
    parser.add_argument("--limit_tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1])
    return parser


def _resolve_graph_data_dir(*, dataset: str, dataset_split_tag: str, graph_data_dir_override: str) -> str:
    if str(graph_data_dir_override).strip():
        return str(graph_data_dir_override).strip()
    return f"./data/{dataset}_{dataset_split_tag}"


def _resolve_online_symbolic_for_task(args, task: str) -> Dict[str, object]:
    template = str(args.online_symbolic_spec_template).strip()
    if template:
        spec_path = template.format(task=str(task), dataset=str(args.dataset))
    elif str(args.online_symbolic_spec_path).strip():
        spec_path = str(args.online_symbolic_spec_path).strip()
    else:
        spec_path = ""
    repr_mode = str(args.online_symbolic_repr).strip().lower() or "auto"
    normalize_mode = str(args.online_symbolic_normalize).strip().lower() or "none"
    tile_repeat = max(1, int(args.online_symbolic_tile_repeat))
    return {
        "spec_path": str(spec_path),
        "repr": str(repr_mode),
        "normalize": str(normalize_mode),
        "tile_repeat": int(tile_repeat),
    }


def _build_run_params(
    args,
    tasks: List[str],
    profiler_enabled: bool,
    profiler_run_name: str,
    run_tag: str,
    graph_data_dir: str,
    em_pair_features: List[str],
    em_decoder_static_features: List[str],
    jts_pair_features: List[str],
    jts_decoder_static_features: List[str],
    uts_pair_features: List[str],
    uts_decoder_static_features: List[str],
    sm_pair_features: List[str],
    sm_decoder_static_features: List[str],
) -> Dict[str, object]:
    task_eval_cfg: Dict[str, Dict[str, object]] = {}
    for task in tasks:
        cfg = task_v3c_config(task)
        task_eval_cfg[task] = {
            "eval_threshold_mode": str(cfg.get("eval_threshold_mode", "val_best")),
            "eval_fixed_threshold": float(cfg.get("eval_fixed_threshold", 0.5)),
            "training_online_rerank_enabled": False,
        }
    return {
        "dataset": args.dataset,
        "dataset_split_tag": str(args.dataset_split_tag),
        "graph_data_dir": os.path.abspath(graph_data_dir),
        "task_permutation": int(args.task_permutation),
        "tasks": tasks,
        "device": args.device,
        "variant": "0318_v3c",
        "frozen_profile": "v3c",
        "reference_run_logs": REFERENCE_RUN_LOGS,
        "task_graph_mode": "independent_reload_no_updates",
        "seed": int(args.seed),
        "num_neighbors": [int(x.strip()) for x in str(args.num_neighbors).split(",") if x.strip()],
        "early_stopping_patience": int(args.early_stopping_patience),
        "drop_cell_edges": int(args.drop_cell_edges),
        "task_drop_cell_edges_overrides": {
            "entity_matching": int(args.em_drop_cell_edges),
            "joinable_table_search": int(args.jts_drop_cell_edges),
            "union_table_search": int(args.uts_drop_cell_edges),
            "schema_matching": int(args.sm_drop_cell_edges),
        },
        "task_graph_data_dir_overrides": {
            "entity_matching": str(args.em_graph_data_dir).strip(),
            "joinable_table_search": str(args.jts_graph_data_dir).strip(),
            "union_table_search": str(args.uts_graph_data_dir).strip(),
            "schema_matching": str(args.sm_graph_data_dir).strip(),
        },
        "em_pair_features": em_pair_features,
        "em_decoder_static_features": em_decoder_static_features,
        "em_auto_pos_weight": int(args.em_auto_pos_weight),
        "em_pos_weight_cap": float(args.em_pos_weight_cap),
        "em_use_interactions": int(args.em_use_interactions),
        "em_loss": str(args.em_loss),
        "em_focal_gamma": float(args.em_focal_gamma),
        "em_focal_alpha": float(args.em_focal_alpha),
        "em_contrastive_weight": float(args.em_contrastive_weight),
        "em_pair_feat_norm": int(args.em_pair_feat_norm),
        "online_symbolic_spec_path": str(args.online_symbolic_spec_path),
        "online_symbolic_spec_template": str(args.online_symbolic_spec_template),
        "online_symbolic_repr": str(args.online_symbolic_repr),
        "online_symbolic_normalize": str(args.online_symbolic_normalize),
        "online_symbolic_tile_repeat": int(args.online_symbolic_tile_repeat),
        "em_decoder_width_mult": float(args.em_decoder_width_mult),
        "em_hard_neg_ratio": float(args.em_hard_neg_ratio),
        "em_hard_neg_warmup_epochs": int(args.em_hard_neg_warmup_epochs),
        "em_row_stats_mode": str(args.em_row_stats_mode),
        "em_pair_cache_mode": str(args.em_pair_cache_mode),
        "em_pair_cache_root": str(args.em_pair_cache_root),
        "em_generated_feature_specs_path": str(args.em_generated_feature_specs_path),
        "jts_generated_feature_specs_path": str(args.jts_generated_feature_specs_path),
        "sm_generated_feature_specs_path": str(args.sm_generated_feature_specs_path),
        "uts_generated_feature_specs_path": str(args.uts_generated_feature_specs_path),
        "debug_max_train_edges": int(args.debug_max_train_edges),
        "debug_max_val_edges": int(args.debug_max_val_edges),
        "debug_max_test_edges": int(args.debug_max_test_edges),
        "jts_pair_features": jts_pair_features,
        "jts_decoder_static_features": jts_decoder_static_features,
        "uts_pair_features": uts_pair_features,
        "uts_decoder_static_features": uts_decoder_static_features,
        "sm_pair_features": sm_pair_features,
        "sm_decoder_static_features": sm_decoder_static_features,
        "allow_empty_decoder_static_features": int(args.allow_empty_decoder_static_features),
        "feature_wiring_mode": str(args.feature_wiring_mode),
        "symbolic_static_decouple": int(
            1 if str(args.feature_wiring_mode).strip().lower() in ("decoupled", "decouple", "") else 0
        ),
        "global_v3c_eval_defaults": {
            "eval_threshold_mode": str(GLOBAL_V3C_CONFIG.get("eval_threshold_mode", "val_best")),
            "eval_fixed_threshold": float(GLOBAL_V3C_CONFIG.get("eval_fixed_threshold", 0.5)),
            "training_online_rerank_enabled": False,
        },
        "task_v3c_eval_config": task_eval_cfg,
        "run_tag": run_tag,
        "replay_archive_root": os.path.abspath(args.replay_archive_root) if args.replay_archive_root else "",
        "profile_runtime": profiler_enabled,
        "profile_dir": os.path.abspath(args.profile_dir) if profiler_enabled else "",
        "profile_run_name": profiler_run_name if profiler_enabled else "",
        "limit_tasks": int(args.limit_tasks),
        "deterministic": bool(int(args.deterministic)),
    }


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _configure_determinism(int(args.seed), deterministic=bool(int(args.deterministic)))

    num_neighbors = [int(x.strip()) for x in str(args.num_neighbors).split(",") if x.strip()]
    if not num_neighbors:
        num_neighbors = [10, 5]
    em_pair_features_legacy = _parse_feature_list(args.em_pair_features)
    em_pair_features_new = _parse_feature_list(args.em_symbolic_source_features)
    em_pair_features = em_pair_features_new if em_pair_features_new else em_pair_features_legacy
    em_generated_feature_names = _load_generated_feature_names(
        str(args.em_generated_feature_specs_path),
        expected_task="entity_matching",
        expected_scope="row_pair",
    )
    em_pair_features = _append_unique(em_pair_features, em_generated_feature_names)
    em_decoder_pair_features_legacy = _parse_feature_list(args.em_decoder_pair_features)
    em_decoder_static_features_new = _parse_feature_list(args.em_decoder_static_features)
    em_decoder_static_features = (
        em_decoder_static_features_new
        if em_decoder_static_features_new
        else em_decoder_pair_features_legacy
    )

    jts_pair_features_legacy = _parse_feature_list(args.jts_pair_features)
    jts_pair_features_new = _parse_feature_list(args.jts_symbolic_source_features)
    jts_pair_features = jts_pair_features_new if jts_pair_features_new else jts_pair_features_legacy
    jts_generated_feature_names = _load_generated_feature_names(
        str(args.jts_generated_feature_specs_path),
        expected_task="joinable_table_search",
        expected_scope="column_pair",
    )
    jts_pair_features = _append_unique(jts_pair_features, jts_generated_feature_names)
    jts_decoder_static_features = _parse_feature_list(args.jts_decoder_static_features)

    uts_pair_features_legacy = _parse_feature_list(args.uts_pair_features)
    uts_pair_features_new = _parse_feature_list(args.uts_symbolic_source_features)
    uts_pair_features = uts_pair_features_new if uts_pair_features_new else uts_pair_features_legacy
    uts_generated_feature_names = _load_generated_feature_names(
        str(args.uts_generated_feature_specs_path),
        expected_task="union_table_search",
        expected_scope="table_pair",
    )
    uts_pair_features = _append_unique(uts_pair_features, uts_generated_feature_names)
    uts_decoder_static_features = _parse_feature_list(args.uts_decoder_static_features)

    sm_pair_features_legacy = _parse_feature_list(args.sm_pair_features)
    sm_pair_features_new = _parse_feature_list(args.sm_symbolic_source_features)
    sm_pair_features = sm_pair_features_new if sm_pair_features_new else sm_pair_features_legacy
    sm_generated_feature_names = _load_generated_feature_names(
        str(args.sm_generated_feature_specs_path),
        expected_task="schema_matching",
        expected_scope="column_pair",
    )
    sm_pair_features = _append_unique(sm_pair_features, sm_generated_feature_names)
    sm_decoder_static_features = _parse_feature_list(args.sm_decoder_static_features)

    feature_wiring_mode = str(getattr(args, "feature_wiring_mode", "decoupled")).strip().lower()
    if feature_wiring_mode in ("decoupled", "decouple", ""):
        decouple_flag = 1
    elif feature_wiring_mode in ("coupled", "couple"):
        decouple_flag = 0
    else:
        raise ValueError(
            f"Invalid --feature_wiring_mode={feature_wiring_mode!r}; expected 'decoupled' or 'coupled'."
        )

    if decouple_flag == 0:
        em_decoder_static_features = list(em_pair_features)
        jts_decoder_static_features = list(jts_pair_features)
        uts_decoder_static_features = list(uts_pair_features)
        sm_decoder_static_features = list(sm_pair_features)
    else:
        allow_empty_decoder = int(getattr(args, "allow_empty_decoder_static_features", 0))
        if not allow_empty_decoder:
            if not em_decoder_static_features:
                em_decoder_static_features = list(em_pair_features)
            if not jts_decoder_static_features:
                jts_decoder_static_features = list(jts_pair_features)
            if not uts_decoder_static_features:
                uts_decoder_static_features = list(uts_pair_features)
            if not sm_decoder_static_features:
                sm_decoder_static_features = list(sm_pair_features)

    base_graph_data_dir = _resolve_graph_data_dir(
        dataset=str(args.dataset),
        dataset_split_tag=str(args.dataset_split_tag),
        graph_data_dir_override=str(args.graph_data_dir),
    )
    if not os.path.isdir(base_graph_data_dir):
        raise FileNotFoundError(
            f"Graph data dir not found: {base_graph_data_dir} "
            f"(dataset={args.dataset}, split_tag={args.dataset_split_tag}, override={bool(str(args.graph_data_dir).strip())})"
        )
    task_graph_dir_overrides = {
        "entity_matching": str(args.em_graph_data_dir).strip(),
        "joinable_table_search": str(args.jts_graph_data_dir).strip(),
        "union_table_search": str(args.uts_graph_data_dir).strip(),
        "schema_matching": str(args.sm_graph_data_dir).strip(),
    }
    task_graph_data_dirs = {
        task: (task_graph_dir_overrides.get(task) or base_graph_data_dir)
        for task in ["entity_matching", "joinable_table_search", "union_table_search", "schema_matching"]
    }
    global_drop_cell_edges = int(args.drop_cell_edges)
    task_drop_cell_edges_overrides = {
        "entity_matching": int(args.em_drop_cell_edges),
        "joinable_table_search": int(args.jts_drop_cell_edges),
        "union_table_search": int(args.uts_drop_cell_edges),
        "schema_matching": int(args.sm_drop_cell_edges),
    }
    task_drop_cell_edges = {
        task: (
            task_drop_cell_edges_overrides.get(task)
            if int(task_drop_cell_edges_overrides.get(task, -1)) in (0, 1)
            else global_drop_cell_edges
        )
        for task in ["entity_matching", "joinable_table_search", "union_table_search", "schema_matching"]
    }

    tasks = TASK_PERMUTATIONS.get(int(args.task_permutation), TASK_PERMUTATIONS[0])
    if int(args.limit_tasks) > 0:
        tasks = tasks[: int(args.limit_tasks)]

    profiler_enabled = bool(int(args.profile_runtime))
    profiler_run_name = args.profile_run_name.strip() or (
        f"{args.dataset}_perm{int(args.task_permutation)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_tag = str(args.run_tag).strip() or profiler_run_name
    if profiler_enabled:
        init_runtime_profiler(
            log_dir=args.profile_dir,
            run_name=profiler_run_name,
            interval_sec=float(args.profile_interval_sec),
        )

    _log_json(
        "[RunParams]",
        _build_run_params(
            args,
            tasks,
            profiler_enabled,
            profiler_run_name,
            run_tag,
            base_graph_data_dir,
            em_pair_features,
            em_decoder_static_features,
            jts_pair_features,
            jts_decoder_static_features,
            uts_pair_features,
            uts_decoder_static_features,
            sm_pair_features,
            sm_decoder_static_features,
        ),
    )
    record_profile_event(
        "run_metadata",
        {
            "dataset": args.dataset,
            "dataset_split_tag": str(args.dataset_split_tag),
            "graph_data_dir": os.path.abspath(base_graph_data_dir),
            "task_graph_data_dirs": {k: os.path.abspath(v) for k, v in task_graph_data_dirs.items()},
            "task_drop_cell_edges": task_drop_cell_edges,
            "task_permutation": int(args.task_permutation),
            "tasks": tasks,
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.early_stopping_patience),
            "batch_size": int(args.batch_size),
            "num_neighbors": num_neighbors,
            "num_workers": int(args.num_workers),
            "device_arg": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "profile_runtime": profiler_enabled,
            "variant": "0318_v3c",
            "seed": int(args.seed),
            "deterministic": bool(int(args.deterministic)),
        },
    )
    logger.info("[Static] Tasks are independent. No edge_update or heavy_update is performed in this variant.")
    logger.info("[Static] v3c spec is frozen against the 2026-03-17 reference runs in 0317_task_specific_v3c_full.")
    logger.info("[Static] Online rerank/policy branches are frozen OFF in the current training path.")
    logger.info("[Static] Deterministic mode: %s", str(bool(int(args.deterministic))).lower())

    try:
        for task_idx, task in enumerate(tasks):
            task_cfg = task_v3c_config(task)
            logger.info(f"\n{'=' * 20}\nTraining for task: {task}\n{'=' * 20}")
            record_profile_event("task_start", {"task_idx": int(task_idx), "task": task})

            task_graph_data_dir = task_graph_data_dirs.get(task, base_graph_data_dir)
            task_drop_cell_edges_flag = int(task_drop_cell_edges.get(task, global_drop_cell_edges))
            if not os.path.isdir(task_graph_data_dir):
                raise FileNotFoundError(
                    f"Task graph data dir not found: {task_graph_data_dir} "
                    f"(task={task}, base={base_graph_data_dir})"
                )
            online_symbolic_cfg = _resolve_online_symbolic_for_task(args, task=task)
            if str(online_symbolic_cfg["spec_path"]).strip():
                logger.info(
                    "[OnlineSymbolic] task=%s spec=%s repr=%s normalize=%s tile_repeat=%d",
                    str(task),
                    str(online_symbolic_cfg["spec_path"]),
                    str(online_symbolic_cfg["repr"]),
                    str(online_symbolic_cfg["normalize"]),
                    int(online_symbolic_cfg["tile_repeat"]),
                )
            with profile_phase("task.graph_init", {"task_idx": int(task_idx), "task": task}):
                graph = HierGraph(
                    task_graph_data_dir,
                    cache_namespace="0308_base_em",
                    include_cell_edges=bool(args.prepool_cells) or (not bool(task_drop_cell_edges_flag)),
                )

            with profile_phase("task.trainer_init", {"task_idx": int(task_idx), "task": task}):
                trainer = GraphLinkPredictionTrainer(
                    graph=graph,
                    embedding_dim=768,
                    hidden_dim=args.hidden_dim,
                    learning_rate=args.lr,
                    batch_size=args.batch_size,
                    device=args.device,
                    target_task=task,
                    gnn_type=args.gnn_type,
                    num_gnn_layers=args.gnn_layers,
                    num_neighbors=num_neighbors,
                    num_workers=args.num_workers,
                    prepool_cells=bool(args.prepool_cells),
                    prepool_rows=bool(args.prepool_rows),
                    prepool_cols=bool(args.prepool_cols),
                    prepool_row_alpha=args.prepool_row_alpha,
                    prepool_col_alpha=args.prepool_col_alpha,
                    prepool_chunk_size=args.prepool_chunk_size,
                    drop_cell_edges=bool(task_drop_cell_edges_flag),
                    dataset_name=args.dataset,
                    run_tag=run_tag,
                    replay_archive_root=str(args.replay_archive_root),
                    em_pair_features=em_pair_features,
                    em_decoder_pair_features=em_decoder_static_features,
                    em_table_root=args.em_table_root,
                    em_auto_pos_weight=bool(args.em_auto_pos_weight),
                    em_pos_weight_cap=float(args.em_pos_weight_cap),
                    em_use_interactions=bool(args.em_use_interactions),
                    em_loss=str(args.em_loss),
                    em_focal_gamma=float(args.em_focal_gamma),
                    em_focal_alpha=float(args.em_focal_alpha),
                    em_contrastive_weight=float(args.em_contrastive_weight),
                    em_pair_feat_norm=bool(args.em_pair_feat_norm),
                    em_generated_feature_specs_path=str(args.em_generated_feature_specs_path),
                    jts_generated_feature_specs_path=str(args.jts_generated_feature_specs_path),
                    sm_generated_feature_specs_path=str(args.sm_generated_feature_specs_path),
                    uts_generated_feature_specs_path=str(args.uts_generated_feature_specs_path),
                    online_symbolic_spec_path=str(online_symbolic_cfg["spec_path"]),
                    online_symbolic_repr=str(online_symbolic_cfg["repr"]),
                    online_symbolic_normalize=str(online_symbolic_cfg["normalize"]),
                    online_symbolic_tile_repeat=int(online_symbolic_cfg["tile_repeat"]),
                    em_decoder_width_mult=float(args.em_decoder_width_mult),
                    em_hard_neg_ratio=float(args.em_hard_neg_ratio),
                    em_hard_neg_warmup_epochs=int(args.em_hard_neg_warmup_epochs),
                    em_row_stats_mode=str(args.em_row_stats_mode),
                    em_pair_cache_mode=str(args.em_pair_cache_mode),
                    em_pair_cache_root=str(args.em_pair_cache_root),
                    debug_max_train_edges=int(args.debug_max_train_edges),
                    debug_max_val_edges=int(args.debug_max_val_edges),
                    debug_max_test_edges=int(args.debug_max_test_edges),
                    jts_pair_features=jts_pair_features,
                    jts_decoder_pair_features=jts_decoder_static_features,
                    jts_table_root=args.jts_table_root,
                    sm_pair_features=sm_pair_features,
                    sm_decoder_pair_features=sm_decoder_static_features,
                    sm_table_root=args.sm_table_root,
                    uts_pair_features=uts_pair_features,
                    uts_decoder_pair_features=uts_decoder_static_features,
                    uts_table_root=args.uts_table_root,
                    symbolic_static_decouple=bool(int(decouple_flag)),
                    eval_threshold_mode=str(task_cfg["eval_threshold_mode"]),
                    eval_fixed_threshold=float(task_cfg["eval_fixed_threshold"]),
                    seed=int(args.seed) + int(task_idx) * 100,
                )

            with profile_phase("task.train", {"task_idx": int(task_idx), "task": task, "epochs": int(args.epochs)}):
                trainer.train(num_epochs=args.epochs, early_stopping_patience=int(args.early_stopping_patience))

            del trainer
            del graph
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            record_profile_event("task_end", {"task_idx": int(task_idx), "task": task})
    finally:
        if profiler_enabled:
            close_runtime_profiler()

    logger.info("\nAll tasks completed.")
