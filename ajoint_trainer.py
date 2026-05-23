from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

from ajoint_graph_utils import _filter_edges_by_type_ids
from ajoint_metrics import _binary_classification_metrics, find_best_threshold
from ajoint_models import ContrastiveLoss, GraphLinkPredictor
from ajoint_pair_features import EntityPairFeatureStore, JoinablePairFeatureStore, SchemaPairFeatureStore, UnionPairFeatureStore
from ajoint_prepool import _build_prepool_overlay_cached, _get_node_embeddings_cached
from executor import ResidualReranker
from feature_schema import build_evidence
from runtime_profile import profile_phase, record_profile_event, tensor_nbytes
from symbolic_feature import SymbolicFeatureExecutor, build_feature_map_from_matrix, load_symbolic_feature_spec


logger = logging.getLogger(__name__)

TQDM_DISABLE = not sys.stderr.isatty()


def _safe_token(raw: str, *, default: str = "unknown") -> str:
    token = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(raw).strip())
    token = token.strip("._-")
    return token or default


class LinearBenefitSelector:
    """Low-capacity selector that predicts whether a sample is worth reranking."""

    def __init__(self, *, lr: float = 0.05, weight_decay: float = 1e-3, epochs: int = 200, seed: int = 0):
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.weight_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.feature_dim_: int = 0
        self.pos_count_: int = 0
        self.neg_count_: int = 0

    @staticmethod
    def _normalize_inputs(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float32)
        mean = x.mean(axis=0, keepdims=True)
        scale = x.std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return (x - mean) / scale, mean.astype(np.float32, copy=False), scale.astype(np.float32, copy=False)

    def fit(self, features: np.ndarray, targets: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> bool:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if x.ndim != 2 or len(x) == 0 or len(x) != len(y):
            return False
        pos_count = int((y > 0.5).sum())
        neg_count = int((y <= 0.5).sum())
        if pos_count <= 0 or neg_count <= 0:
            return False

        x_norm, mean, scale = self._normalize_inputs(x)
        weight = np.ones((len(x_norm), 1), dtype=np.float32)
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
            weight = np.clip(weight, 1e-4, None)

        torch.manual_seed(self.seed)
        linear = nn.Linear(x_norm.shape[1], 1)
        optimizer = optim.Adam(linear.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_weight = torch.tensor([max(float(neg_count) / max(pos_count, 1), 1.0)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        x_t = torch.as_tensor(x_norm, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        w_t = torch.as_tensor(weight, dtype=torch.float32)
        linear.train()
        for _ in range(max(self.epochs, 1)):
            optimizer.zero_grad(set_to_none=True)
            logits = linear(x_t)
            loss = criterion(logits, y_t)
            loss = (loss * w_t).sum() / w_t.sum()
            loss.backward()
            optimizer.step()

        self.mean_ = mean.reshape(-1)
        self.scale_ = scale.reshape(-1)
        self.weight_ = linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        self.bias_ = float(linear.bias.detach().cpu().item())
        self.feature_dim_ = int(x.shape[1])
        self.pos_count_ = pos_count
        self.neg_count_ = neg_count
        return True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weight_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Benefit selector is not fitted.")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={x.shape}")
        x_norm = (x - self.mean_.reshape(1, -1)) / self.scale_.reshape(1, -1)
        logits = x_norm @ self.weight_.reshape(-1, 1) + float(self.bias_)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return probs.reshape(-1).astype(np.float32, copy=False)

    def summary(self) -> Dict[str, object]:
        return {
            "feature_dim": int(self.feature_dim_),
            "pos_count": int(self.pos_count_),
            "neg_count": int(self.neg_count_),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "epochs": int(self.epochs),
        }


class LinearResidualHead:
    """Low-capacity ridge regressor for tiny task-specific residual correction."""

    def __init__(self, *, l2: float = 0.5):
        self.l2 = float(l2)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.weight_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.feature_dim_: int = 0

    @staticmethod
    def _normalize_inputs(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float32)
        mean = x.mean(axis=0, keepdims=True)
        scale = x.std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return (x - mean) / scale, mean.astype(np.float32, copy=False), scale.astype(np.float32, copy=False)

    def fit(self, features: np.ndarray, targets: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> bool:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if x.ndim != 2 or len(x) == 0 or len(x) != len(y):
            return False
        x_norm, mean, scale = self._normalize_inputs(x)
        design = np.concatenate([np.ones((len(x_norm), 1), dtype=np.float32), x_norm], axis=1)
        weight = np.ones((len(x_norm), 1), dtype=np.float32)
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
            weight = np.clip(weight, 1e-4, None)
        sqrt_w = np.sqrt(weight)
        xw = design * sqrt_w
        yw = y * sqrt_w
        reg = np.eye(design.shape[1], dtype=np.float32) * float(self.l2)
        reg[0, 0] = 0.0
        try:
            coef = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw).reshape(-1)
        except np.linalg.LinAlgError:
            return False
        self.mean_ = mean.reshape(-1)
        self.scale_ = scale.reshape(-1)
        self.bias_ = float(coef[0])
        self.weight_ = coef[1:].astype(np.float32, copy=False)
        self.feature_dim_ = int(x.shape[1])
        return True

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weight_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Residual head is not fitted.")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={x.shape}")
        x_norm = (x - self.mean_.reshape(1, -1)) / self.scale_.reshape(1, -1)
        preds = x_norm @ self.weight_.reshape(-1, 1) + float(self.bias_)
        return preds.reshape(-1).astype(np.float32, copy=False)

    def summary(self) -> Dict[str, object]:
        return {
            "feature_dim": int(self.feature_dim_),
            "l2": float(self.l2),
        }


class GraphLinkPredictionTrainer:
    """Trainer for link prediction using efficient LinkNeighborLoader."""

    def __init__(
        self,
        graph,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        learning_rate: float = 0.001,
        temperature: float = 0.1,
        batch_size: int = 32,
        device: str = "cuda",
        target_task: str = "entity_matching",
        gnn_type: str = "gat",
        num_gnn_layers: int = 2,
        num_neighbors: List[int] = [15, 10],
        num_workers: int = 4,
        *,
        prepool_cells: bool = True,
        prepool_rows: bool = True,
        prepool_cols: bool = True,
        prepool_row_alpha: float = 0.0,
        prepool_col_alpha: float = 0.0,
        prepool_chunk_size: int = 16384,
        drop_cell_edges: bool = True,
        dataset_name: str = "",
        run_tag: str = "",
        replay_archive_root: str = "",
        em_pair_features: Optional[List[str]] = None,
        em_decoder_pair_features: Optional[List[str]] = None,
        em_table_root: str = "",
        jts_pair_features: Optional[List[str]] = None,
        jts_decoder_pair_features: Optional[List[str]] = None,
        jts_table_root: str = "",
        sm_pair_features: Optional[List[str]] = None,
        sm_decoder_pair_features: Optional[List[str]] = None,
        sm_table_root: str = "",
        uts_pair_features: Optional[List[str]] = None,
        uts_decoder_pair_features: Optional[List[str]] = None,
        uts_table_root: str = "",
        symbolic_static_decouple: bool = True,
        em_auto_pos_weight: bool = False,
        em_pos_weight_cap: float = 5.0,
        em_use_interactions: bool = False,
        em_loss: str = "bce",
        em_focal_gamma: float = 2.0,
        em_focal_alpha: float = 0.25,
        em_contrastive_weight: float = 0.0,
        em_pair_feat_norm: bool = False,
        em_generated_feature_specs_path: str = "",
        jts_generated_feature_specs_path: str = "",
        sm_generated_feature_specs_path: str = "",
        uts_generated_feature_specs_path: str = "",
        online_symbolic_spec_path: str = "",
        online_symbolic_repr: str = "auto",
        online_symbolic_normalize: str = "none",
        online_symbolic_tile_repeat: int = 1,
        em_decoder_width_mult: float = 1.0,
        em_hard_neg_ratio: float = 0.0,
        em_hard_neg_warmup_epochs: int = 5,
        em_row_stats_mode: str = "full",
        em_pair_cache_mode: str = "off",
        em_pair_cache_root: str = "",
        debug_max_train_edges: int = 0,
        debug_max_val_edges: int = 0,
        debug_max_test_edges: int = 0,
        rerank_enable: bool = False,
        rerank_policy_path: str = "",
        rerank_alpha_grid: Optional[List[float]] = None,
        rerank_delta_cap_grid: Optional[List[float]] = None,
        rerank_gate_margin_grid: Optional[List[float]] = None,
        rerank_gate_delta_min_grid: Optional[List[float]] = None,
        rerank_benefit_enable: bool = False,
        rerank_benefit_threshold_grid: Optional[List[float]] = None,
        rerank_benefit_train_epochs: int = 200,
        rerank_benefit_lr: float = 0.05,
        rerank_benefit_l2: float = 1e-3,
        rerank_match_mode: str = "partial",
        rerank_hard_guard_patterns: Optional[List[str]] = None,
        rerank_min_match_ratio: float = 0.67,
        rerank_min_match_ratio_pos: Optional[float] = None,
        rerank_min_match_ratio_neg: Optional[float] = None,
        rerank_scale_delta_by_match: bool = True,
        rerank_shift_s_prior_by_threshold: bool = True,
        rerank_policy_relax: float = 0.0,
        rerank_rule_combination_mode: str = "accumulate",
        rerank_hard_override_eps: float = 1e-3,
        rerank_positive_boundary_only: bool = False,
        rerank_positive_boundary_margin: float = 0.2,
        rerank_negative_boundary_margin: float = 1.0,
        rerank_em_residual_enable: bool = False,
        rerank_em_residual_feature_names: Optional[List[str]] = None,
        rerank_em_residual_margin_grid: Optional[List[float]] = None,
        rerank_em_residual_scale_grid: Optional[List[float]] = None,
        rerank_em_residual_l2: float = 0.5,
        rerank_reference_threshold_mode: str = "val_best",
        rerank_reference_fixed_threshold: float = 0.5,
        rerank_max_active_ratio: float = 1.0,
        eval_threshold_mode: str = "val_best",
        eval_fixed_threshold: float = 0.5,
        seed: int = 0,
    ):
        self.graph = graph
        self.graph_data = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_task = target_task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepool_cells = bool(prepool_cells)
        self.prepool_rows = bool(prepool_rows)
        self.prepool_cols = bool(prepool_cols)
        self.prepool_row_alpha = float(prepool_row_alpha)
        self.prepool_col_alpha = float(prepool_col_alpha)
        self.prepool_chunk_size = int(prepool_chunk_size)
        self.drop_cell_edges = bool(drop_cell_edges)
        self.dataset_name = str(dataset_name).strip()
        self.run_tag = str(run_tag).strip()
        self.replay_archive_root = str(replay_archive_root).strip()
        self._prepool_overlay: Optional[Dict[str, torch.Tensor]] = None
        self.seed = int(seed)
        self.em_pair_feature_store: Optional[EntityPairFeatureStore] = None
        self.em_pair_features = em_pair_features or []
        self.em_decoder_pair_features = em_decoder_pair_features or []
        self.em_table_root = em_table_root
        self.jts_pair_feature_store: Optional[JoinablePairFeatureStore] = None
        self.jts_pair_features = jts_pair_features or []
        self.jts_decoder_pair_features = jts_decoder_pair_features or []
        self.jts_table_root = jts_table_root
        self.sm_pair_feature_store: Optional[SchemaPairFeatureStore] = None
        self.sm_pair_features = sm_pair_features or []
        self.sm_decoder_pair_features = sm_decoder_pair_features or []
        self.sm_table_root = sm_table_root
        self.uts_pair_feature_store: Optional[UnionPairFeatureStore] = None
        self.uts_pair_features = uts_pair_features or []
        self.uts_decoder_pair_features = uts_decoder_pair_features or []
        self.uts_table_root = uts_table_root
        self.symbolic_static_decouple = bool(symbolic_static_decouple)
        self.em_auto_pos_weight = bool(em_auto_pos_weight)
        self.em_pos_weight_cap = float(em_pos_weight_cap)
        self.em_use_interactions = bool(em_use_interactions)
        self.em_loss = str(em_loss).strip().lower()
        if self.em_loss not in {"bce", "focal", "bce_contrastive"}:
            raise ValueError(
                f"Unsupported em_loss={self.em_loss}. Expected one of: bce,focal,bce_contrastive"
            )
        self.em_focal_gamma = float(em_focal_gamma)
        self.em_focal_alpha = float(em_focal_alpha)
        self.em_contrastive_weight = float(em_contrastive_weight)
        self.em_pair_feat_norm = bool(em_pair_feat_norm)
        self.em_generated_feature_specs_path = str(em_generated_feature_specs_path).strip()
        self.jts_generated_feature_specs_path = str(jts_generated_feature_specs_path).strip()
        self.sm_generated_feature_specs_path = str(sm_generated_feature_specs_path).strip()
        self.uts_generated_feature_specs_path = str(uts_generated_feature_specs_path).strip()
        self.online_symbolic_spec_path = str(online_symbolic_spec_path).strip()
        self.online_symbolic_repr = str(online_symbolic_repr).strip().lower() or "auto"
        if self.online_symbolic_repr not in {"auto", "concat", "aggregation"}:
            raise ValueError(
                "Unsupported online_symbolic_repr="
                f"{self.online_symbolic_repr!r}; expected one of: auto,concat,aggregation"
            )
        self.online_symbolic_normalize = str(online_symbolic_normalize).strip().lower() or "none"
        if self.online_symbolic_normalize not in {"none", "zscore"}:
            raise ValueError(
                "Unsupported online_symbolic_normalize="
                f"{self.online_symbolic_normalize!r}; expected one of: none,zscore"
            )
        self.online_symbolic_tile_repeat = max(1, int(online_symbolic_tile_repeat))
        self.em_decoder_width_mult = float(em_decoder_width_mult)
        self.em_hard_neg_ratio = max(0.0, float(em_hard_neg_ratio))
        self.em_hard_neg_warmup_epochs = max(0, int(em_hard_neg_warmup_epochs))
        self.em_row_stats_mode = str(em_row_stats_mode).strip().lower()
        if self.em_row_stats_mode not in {"required", "full"}:
            raise ValueError(
                f"Unsupported em_row_stats_mode={self.em_row_stats_mode}. Expected one of: required,full"
            )
        self.em_pair_cache_mode = str(em_pair_cache_mode).strip().lower()
        if self.em_pair_cache_mode not in {"off", "readwrite"}:
            raise ValueError(
                f"Unsupported em_pair_cache_mode={self.em_pair_cache_mode}. Expected one of: off,readwrite"
            )
        self.em_pair_cache_root = str(em_pair_cache_root).strip()
        self.debug_max_train_edges = max(0, int(debug_max_train_edges))
        self.debug_max_val_edges = max(0, int(debug_max_val_edges))
        self.debug_max_test_edges = max(0, int(debug_max_test_edges))
        self._em_train_pos_weight = 1.0
        self._em_pair_feat_mean: Optional[torch.Tensor] = None
        self._em_pair_feat_std: Optional[torch.Tensor] = None
        self._online_symbolic_enabled = False
        self._online_symbolic_repr_effective = "aggregation"
        self._online_symbolic_executor: Optional[SymbolicFeatureExecutor] = None
        self._online_symbolic_pair_feature_order: List[str] = []
        self._online_symbolic_expected_features: List[str] = []
        self._online_symbolic_raw_dim = 0
        self._online_symbolic_effective_dim = 0
        self._online_symbolic_spec_id = ""
        self._online_symbolic_norm_mean: Optional[torch.Tensor] = None
        self._online_symbolic_norm_std: Optional[torch.Tensor] = None
        self._decoder_pair_keep_indices: Optional[List[int]] = None
        self._decoder_pair_keep_order: List[str] = []
        self._train_edge_label_index_base: Optional[torch.Tensor] = None
        self._train_edge_label_base: Optional[torch.Tensor] = None
        self._train_edge_label_index_active: Optional[torch.Tensor] = None
        self._train_edge_label_active: Optional[torch.Tensor] = None
        self._loader_num_workers = int(num_workers)
        self._row_node_ids_cache: Optional[torch.Tensor] = None
        self._em_feature_build_sec: float = 0.0
        # Online v3c path freezes rerank/policy logic OFF.
        requested_rerank = bool(rerank_enable) or bool(str(rerank_policy_path).strip())
        requested_rerank = requested_rerank or bool(rerank_benefit_enable) or bool(rerank_em_residual_enable)
        if requested_rerank:
            logger.warning(
                "[DeprecatedConfig] rerank/policy arguments are ignored; rerank chain is frozen OFF."
            )
        self.rerank_enable = False
        self.rerank_policy_path = ""
        self.rerank_alpha_grid = [0.25]
        self.rerank_delta_cap_grid = [0.1]
        self.rerank_gate_margin_grid = [1.0]
        self.rerank_gate_delta_min_grid = [0.0]
        self.rerank_benefit_enable = False
        self.rerank_benefit_threshold_grid = [0.0]
        self.rerank_benefit_train_epochs = int(rerank_benefit_train_epochs)
        self.rerank_benefit_lr = float(rerank_benefit_lr)
        self.rerank_benefit_l2 = float(rerank_benefit_l2)
        self.rerank_match_mode = "strict"
        self.rerank_hard_guard_patterns = []
        self.rerank_min_match_ratio = 1.0
        self.rerank_min_match_ratio_pos = 1.0
        self.rerank_min_match_ratio_neg = 1.0
        self.rerank_scale_delta_by_match = False
        self.rerank_shift_s_prior_by_threshold = False
        self.rerank_policy_relax = 0.0
        self.rerank_rule_combination_mode = "accumulate"
        self.rerank_hard_override_eps = float(rerank_hard_override_eps)
        self.rerank_positive_boundary_only = True
        self.rerank_positive_boundary_margin = 0.1
        self.rerank_negative_boundary_margin = 0.2
        self.rerank_em_residual_enable = False
        self.rerank_em_residual_feature_names = []
        self.rerank_em_residual_margin_grid = [0.0]
        self.rerank_em_residual_scale_grid = [0.0]
        self.rerank_em_residual_l2 = float(rerank_em_residual_l2)
        self.rerank_reference_threshold_mode = "val_best"
        self.rerank_reference_fixed_threshold = 0.5
        self.rerank_max_active_ratio = 1.0
        self.eval_threshold_mode = str(eval_threshold_mode).strip().lower()
        if self.eval_threshold_mode not in {"val_best", "fixed"}:
            raise ValueError(f"Unknown eval_threshold_mode={eval_threshold_mode!r}; expected one of: val_best,fixed")
        self.eval_fixed_threshold = float(eval_fixed_threshold)
        if not (0.0 < self.eval_fixed_threshold < 1.0):
            raise ValueError(f"eval_fixed_threshold must be in (0,1), got {self.eval_fixed_threshold}")
        self.reranker: Optional[ResidualReranker] = None
        self.benefit_selector: Optional[LinearBenefitSelector] = None
        self.task_residual_head: Optional[LinearResidualHead] = None
        self._global_degrees = torch.zeros((0,), dtype=torch.float32)

        # Prepare graph data object for PyG Loaders
        with profile_phase("trainer.prepare_graph_data", {"target_task": self.target_task}):
            self.prepare_graph_data(graph)
        self._global_degrees = self._compute_global_degrees()
        if self.prepool_cells:
            with profile_phase("trainer.build_prepool_overlay", {"target_task": self.target_task}):
                self._prepool_overlay = _build_prepool_overlay_cached(
                    graph,
                    row_alpha=self.prepool_row_alpha,
                    col_alpha=self.prepool_col_alpha,
                    chunk_size=self.prepool_chunk_size,
                    enable_row=self.prepool_rows,
                    enable_col=self.prepool_cols,
                )

        with profile_phase("trainer.init_pair_features", {"target_task": self.target_task}):
            if self.target_task == "entity_matching" and self.em_pair_features:
                # Pair-feature extraction only needs nodes that actually appear in
                # EM supervision edges (train/val/test). Keep graph itself full.
                required_node_ids: Optional[Set[int]] = self._collect_em_required_node_ids()
                self.em_pair_feature_store = EntityPairFeatureStore(
                    graph=self.graph,
                    dataset_name=dataset_name,
                    feature_names=self.em_pair_features,
                    table_root_override=self.em_table_root,
                    required_node_ids=required_node_ids,
                    row_stats_mode=self.em_row_stats_mode,
                    pair_cache_mode=self.em_pair_cache_mode,
                    pair_cache_root=self.em_pair_cache_root,
                    generated_feature_specs_path=self.em_generated_feature_specs_path,
                )
            if self.target_task == "joinable_table_search" and self.jts_pair_features:
                self.jts_pair_feature_store = JoinablePairFeatureStore(
                    graph=self.graph,
                    dataset_name=dataset_name,
                    feature_names=self.jts_pair_features,
                    table_root_override=self.jts_table_root,
                    generated_feature_specs_path=self.jts_generated_feature_specs_path,
                )
            if self.target_task == "schema_matching" and self.sm_pair_features:
                self.sm_pair_feature_store = SchemaPairFeatureStore(
                    graph=self.graph,
                    dataset_name=dataset_name,
                    feature_names=self.sm_pair_features,
                    table_root_override=self.sm_table_root,
                    generated_feature_specs_path=self.sm_generated_feature_specs_path,
                )
            if self.target_task == "union_table_search" and self.uts_pair_features:
                self.uts_pair_feature_store = UnionPairFeatureStore(
                    graph=self.graph,
                    dataset_name=dataset_name,
                    feature_names=self.uts_pair_features,
                    table_root_override=self.uts_table_root,
                    generated_feature_specs_path=self.uts_generated_feature_specs_path,
                )
        pair_feature_dim = 0
        if self.em_pair_feature_store is not None:
            pair_feature_dim = int(self.em_pair_feature_store.feature_dim)
        elif self.jts_pair_feature_store is not None:
            pair_feature_dim = int(self.jts_pair_feature_store.feature_dim)
        elif self.sm_pair_feature_store is not None:
            pair_feature_dim = int(self.sm_pair_feature_store.feature_dim)
        elif self.uts_pair_feature_store is not None:
            pair_feature_dim = int(self.uts_pair_feature_store.feature_dim)
        pair_feature_order = self._current_pair_feature_order()
        self._resolve_decoder_static_pair_keep(pair_feature_order=pair_feature_order)
        if self._decoder_pair_keep_indices is not None:
            pair_feature_dim = int(len(self._decoder_pair_keep_indices))
        self._init_online_symbolic(pair_feature_order=pair_feature_order)
        if self._online_symbolic_enabled:
            pair_feature_dim += int(self._online_symbolic_effective_dim)
        if self.rerank_enable and self.rerank_policy_path:
            self.reranker = ResidualReranker(
                task=self.target_task,
                pair_feature_order=pair_feature_order,
                policy_path=self.rerank_policy_path,
                match_mode=self.rerank_match_mode,
                hard_guard_patterns=self.rerank_hard_guard_patterns,
                min_match_ratio=self.rerank_min_match_ratio,
                min_match_ratio_pos=self.rerank_min_match_ratio_pos,
                min_match_ratio_neg=self.rerank_min_match_ratio_neg,
                scale_delta_by_match=self.rerank_scale_delta_by_match,
                shift_s_prior_by_threshold=self.rerank_shift_s_prior_by_threshold,
                policy_relax=self.rerank_policy_relax,
                rule_combination_mode=self.rerank_rule_combination_mode,
                hard_override_eps=self.rerank_hard_override_eps,
                positive_boundary_only=self.rerank_positive_boundary_only,
                positive_boundary_margin=self.rerank_positive_boundary_margin,
                negative_boundary_margin=self.rerank_negative_boundary_margin,
            )

        with profile_phase("trainer.init_model", {"target_task": self.target_task, "pair_feature_dim": pair_feature_dim}):
            self.model = GraphLinkPredictor(
                target_task=target_task,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                temperature=temperature,
                num_gnn_layers=num_gnn_layers,
                gnn_type=gnn_type,
                pair_feature_dim=pair_feature_dim,
                em_use_interactions=self.em_use_interactions,
                em_decoder_width_mult=self.em_decoder_width_mult,
                device=self.device,
            ).to(self.device)

        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.link_loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_neighbors = num_neighbors
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self._train_edges: List[Tuple[int, int]] = []
        self._val_edges: List[Tuple[int, int]] = []
        self._test_edges: List[Tuple[int, int]] = []
        self._record_component_summary(stage="post_init")

    @staticmethod
    def _safe_len(value) -> int:
        try:
            return len(value)
        except Exception:
            return 0

    @staticmethod
    def _subset_sequence(values, indices: Sequence[int]):
        if values is None:
            return None
        if isinstance(values, np.ndarray):
            return values[list(indices)]
        return [values[int(i)] for i in indices]

    def _apply_debug_edge_cap(
        self,
        *,
        split: str,
        edges: List[Tuple[int, int]],
        labels,
        edge_ids,
    ):
        limit_map = {
            "train": int(self.debug_max_train_edges),
            "val": int(self.debug_max_val_edges),
            "test": int(self.debug_max_test_edges),
        }
        limit = int(limit_map.get(str(split), 0))
        total = int(len(edges))
        if limit <= 0 or total <= limit:
            return edges, labels, edge_ids

        seed_offset = {"train": 11, "val": 23, "test": 37}.get(str(split), 0)
        indices = list(range(total))
        rng = random.Random(int(self.seed) + int(seed_offset))
        rng.shuffle(indices)
        keep = sorted(indices[:limit])
        logger.info(
            "[DebugEdgeCap] split=%s kept=%d/%d seed=%d",
            str(split),
            int(len(keep)),
            int(total),
            int(self.seed) + int(seed_offset),
        )
        return (
            [edges[int(i)] for i in keep],
            self._subset_sequence(labels, keep),
            self._subset_sequence(edge_ids, keep),
        )

    def _collect_em_required_node_ids(self) -> Set[int]:
        required: Set[int] = set()
        for getter in (
            self.graph.get_train_edges,
            self.graph.get_val_edges,
            self.graph.get_test_edges,
        ):
            try:
                edges, _, _ = getter("entity_matching")
            except Exception:
                continue
            for src, dst in edges:
                try:
                    required.add(int(src))
                    required.add(int(dst))
                except Exception:
                    continue
        return required

    def _optimizer_state_bytes(self) -> int:
        total = 0
        for state in self.optimizer.state.values():
            for value in state.values():
                total += tensor_nbytes(value)
        return total

    def _compute_global_degrees(self) -> torch.Tensor:
        if self.graph_data is None or getattr(self.graph_data, "edge_index", None) is None:
            return torch.zeros((0,), dtype=torch.float32)
        num_nodes = int(self.graph_data.x.size(0))
        flat_index = self.graph_data.edge_index.detach().cpu().reshape(-1)
        if flat_index.numel() == 0:
            return torch.zeros((num_nodes,), dtype=torch.float32)
        return torch.bincount(flat_index, minlength=num_nodes).to(dtype=torch.float32)

    def _edge_globals_from_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        src_local = batch.edge_label_index[0]
        dst_local = batch.edge_label_index[1]
        src_global = batch.n_id[src_local].detach().cpu().to(dtype=torch.long)
        dst_global = batch.n_id[dst_local].detach().cpu().to(dtype=torch.long)
        return src_global, dst_global

    def _edge_degrees_from_globals(
        self,
        *,
        src_global: torch.Tensor,
        dst_global: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_degree = self._global_degrees.index_select(0, src_global).to(device=device, dtype=dtype)
        dst_degree = self._global_degrees.index_select(0, dst_global).to(device=device, dtype=dtype)
        return src_degree, dst_degree

    def _active_pair_feature_store(self):
        store = self.em_pair_feature_store
        if store is None:
            store = self.jts_pair_feature_store
        if store is None:
            store = self.sm_pair_feature_store
        if store is None:
            store = self.uts_pair_feature_store
        return store

    def _current_pair_feature_order(self) -> List[str]:
        store = self._active_pair_feature_store()
        if store is None:
            return []
        return [str(name) for name in getattr(store, "feature_order", [])]

    def _resolve_decoder_static_pair_keep(
        self,
        *,
        pair_feature_order: Sequence[str],
    ) -> None:
        self._decoder_pair_keep_indices = None
        self._decoder_pair_keep_order = []
        if not self.symbolic_static_decouple:
            return
        task = str(self.target_task)
        requested: List[str] = []
        resolver = None
        if task == "entity_matching":
            requested = [
                str(x).strip()
                for x in (self.em_decoder_pair_features or [])
                if str(x).strip() and str(x).strip() != "none"
            ]
            resolver = EntityPairFeatureStore._resolve_requested_atoms
        elif task == "joinable_table_search":
            requested = [
                str(x).strip()
                for x in (self.jts_decoder_pair_features or [])
                if str(x).strip() and str(x).strip() != "none"
            ]
            resolver = JoinablePairFeatureStore._resolve_requested_atoms
        elif task == "schema_matching":
            requested = [
                str(x).strip()
                for x in (self.sm_decoder_pair_features or [])
                if str(x).strip() and str(x).strip() != "none"
            ]
            resolver = SchemaPairFeatureStore._resolve_requested_atoms
        elif task == "union_table_search":
            requested = [
                str(x).strip()
                for x in (self.uts_decoder_pair_features or [])
                if str(x).strip() and str(x).strip() != "none"
            ]
            resolver = UnionPairFeatureStore._resolve_requested_atoms
        else:
            return

        if not requested:
            # In decoupled mode, empty decoder feature request means no static
            # pair features should be injected into decoder input.
            self._decoder_pair_keep_indices = []
            self._decoder_pair_keep_order = []
            logger.info(
                "[DecoderStaticSelect] task=%s requested=%s resolved_atoms=%s keep_dim=%d source_dim=%d",
                task,
                requested,
                [],
                0,
                int(len(pair_feature_order)),
            )
            return
        assert resolver is not None
        _, requested_atoms = resolver(requested)
        if not requested_atoms:
            self._decoder_pair_keep_indices = []
            self._decoder_pair_keep_order = []
            return

        idx_map: Dict[str, List[int]] = {}
        for idx, name in enumerate([str(x) for x in pair_feature_order]):
            idx_map.setdefault(name, []).append(int(idx))

        used_cnt: Dict[str, int] = {}
        keep_indices: List[int] = []
        keep_order: List[str] = []
        missing: List[str] = []
        for atom in requested_atoms:
            used = int(used_cnt.get(atom, 0))
            candidates = idx_map.get(atom, [])
            if used >= len(candidates):
                missing.append(atom)
                continue
            keep_indices.append(int(candidates[used]))
            keep_order.append(str(atom))
            used_cnt[atom] = used + 1

        if missing:
            raise ValueError(
                f"{task} decoder_static_features cannot be resolved from symbolic_source_features. "
                f"missing_atoms={sorted(set(missing))} source_order={list(pair_feature_order)}"
            )

        self._decoder_pair_keep_indices = keep_indices
        self._decoder_pair_keep_order = keep_order
        logger.info(
            "[DecoderStaticSelect] task=%s requested=%s resolved_atoms=%s keep_dim=%d source_dim=%d",
            task,
            requested,
            keep_order,
            int(len(keep_indices)),
            int(len(pair_feature_order)),
        )

    def _init_online_symbolic(self, *, pair_feature_order: Sequence[str]) -> None:
        self._online_symbolic_enabled = False
        self._online_symbolic_repr_effective = "aggregation"
        self._online_symbolic_executor = None
        self._online_symbolic_pair_feature_order = list(pair_feature_order)
        self._online_symbolic_expected_features = []
        self._online_symbolic_raw_dim = 0
        self._online_symbolic_effective_dim = 0
        self._online_symbolic_spec_id = ""
        self._online_symbolic_norm_mean = None
        self._online_symbolic_norm_std = None

        if not self.online_symbolic_spec_path:
            return
        if not pair_feature_order:
            raise RuntimeError("Online symbolic requires non-empty pair_feature_order.")

        spec = load_symbolic_feature_spec(
            self.online_symbolic_spec_path,
            expected_task=str(self.target_task),
            allowed_features=None,
        )
        self._online_symbolic_executor = SymbolicFeatureExecutor(
            spec=spec,
            fallback_value=0.0,
            strict=False,
        )
        self._online_symbolic_expected_features = [str(x) for x in getattr(spec, "feature_pool_used", [])]
        repr_mode = str(self.online_symbolic_repr).strip().lower()
        if repr_mode == "auto":
            repr_mode = "concat" if str(spec.spec_version) == "v2" else "aggregation"
        if repr_mode not in {"concat", "aggregation"}:
            raise ValueError(
                f"Unsupported effective online_symbolic_repr={repr_mode!r}; expected concat/aggregation"
            )
        if repr_mode == "concat":
            raw_dim = int(len(spec.channels)) if str(spec.spec_version) == "v2" else 1
        else:
            raw_dim = 1
        raw_dim = max(1, int(raw_dim))
        eff_dim = int(raw_dim) * int(self.online_symbolic_tile_repeat)

        self._online_symbolic_enabled = True
        self._online_symbolic_repr_effective = repr_mode
        self._online_symbolic_raw_dim = int(raw_dim)
        self._online_symbolic_effective_dim = int(eff_dim)
        self._online_symbolic_spec_id = str(getattr(spec, "spec_id", ""))
        missing = sorted(set(self._online_symbolic_expected_features) - set(self._online_symbolic_pair_feature_order))
        logger.info(
            "[OnlineSymbolic] enabled task=%s spec=%s repr=%s raw_dim=%d tile_repeat=%d effective_dim=%d normalize=%s missing_expected=%d",
            str(self.target_task),
            self.online_symbolic_spec_path,
            self._online_symbolic_repr_effective,
            int(self._online_symbolic_raw_dim),
            int(self.online_symbolic_tile_repeat),
            int(self._online_symbolic_effective_dim),
            self.online_symbolic_normalize,
            int(len(missing)),
        )

    def _rerank_summary(self) -> Dict[str, object]:
        if self.reranker is None:
            return {
                "rerank_enabled": False,
                "rerank_benefit_enable": bool(self.rerank_benefit_enable),
                "rerank_benefit_threshold_grid": list(self.rerank_benefit_threshold_grid),
                "rerank_benefit_train_epochs": int(self.rerank_benefit_train_epochs),
                "rerank_benefit_lr": float(self.rerank_benefit_lr),
                "rerank_benefit_l2": float(self.rerank_benefit_l2),
                "rerank_rule_combination_mode": self.rerank_rule_combination_mode,
                "rerank_positive_boundary_only": bool(self.rerank_positive_boundary_only),
                "rerank_positive_boundary_margin": float(self.rerank_positive_boundary_margin),
                "rerank_negative_boundary_margin": float(self.rerank_negative_boundary_margin),
                "rerank_em_residual_enable": bool(self.rerank_em_residual_enable),
                "rerank_em_residual_feature_names": list(self.rerank_em_residual_feature_names),
                "rerank_em_residual_margin_grid": list(self.rerank_em_residual_margin_grid),
                "rerank_em_residual_scale_grid": list(self.rerank_em_residual_scale_grid),
                "rerank_em_residual_l2": float(self.rerank_em_residual_l2),
                "rerank_reference_threshold_mode": self.rerank_reference_threshold_mode,
                "rerank_reference_fixed_threshold": float(self.rerank_reference_fixed_threshold),
                "rerank_max_active_ratio": float(self.rerank_max_active_ratio),
                "eval_threshold_mode": self.eval_threshold_mode,
                "eval_fixed_threshold": float(self.eval_fixed_threshold),
            }
        return {
            "rerank_enabled": True,
            "rerank_policy_path": self.rerank_policy_path,
            "rerank_rule_count": int(len(self.reranker.policy.rules)),
            "rerank_selected_feature_count": int(len(self.reranker.selected_features)),
            "rerank_gate_margin_grid": list(self.rerank_gate_margin_grid),
            "rerank_gate_delta_min_grid": list(self.rerank_gate_delta_min_grid),
            "rerank_benefit_enable": bool(self.rerank_benefit_enable),
            "rerank_benefit_threshold_grid": list(self.rerank_benefit_threshold_grid),
            "rerank_benefit_train_epochs": int(self.rerank_benefit_train_epochs),
            "rerank_benefit_lr": float(self.rerank_benefit_lr),
            "rerank_benefit_l2": float(self.rerank_benefit_l2),
            "rerank_match_mode": self.rerank_match_mode,
            "rerank_hard_guard_patterns": list(self.rerank_hard_guard_patterns),
            "rerank_min_match_ratio": float(self.rerank_min_match_ratio),
            "rerank_min_match_ratio_pos": self.rerank_min_match_ratio_pos,
            "rerank_min_match_ratio_neg": self.rerank_min_match_ratio_neg,
            "rerank_scale_delta_by_match": bool(self.rerank_scale_delta_by_match),
            "rerank_shift_s_prior_by_threshold": bool(self.rerank_shift_s_prior_by_threshold),
            "rerank_policy_relax": float(self.rerank_policy_relax),
            "rerank_rule_combination_mode": self.rerank_rule_combination_mode,
            "rerank_hard_override_eps": float(self.rerank_hard_override_eps),
            "rerank_positive_boundary_only": bool(self.rerank_positive_boundary_only),
            "rerank_positive_boundary_margin": float(self.rerank_positive_boundary_margin),
            "rerank_negative_boundary_margin": float(self.rerank_negative_boundary_margin),
            "rerank_em_residual_enable": bool(self.rerank_em_residual_enable),
            "rerank_em_residual_feature_names": list(self.rerank_em_residual_feature_names),
            "rerank_em_residual_margin_grid": list(self.rerank_em_residual_margin_grid),
            "rerank_em_residual_scale_grid": list(self.rerank_em_residual_scale_grid),
            "rerank_em_residual_l2": float(self.rerank_em_residual_l2),
            "rerank_reference_threshold_mode": self.rerank_reference_threshold_mode,
            "rerank_reference_fixed_threshold": float(self.rerank_reference_fixed_threshold),
            "rerank_max_active_ratio": float(self.rerank_max_active_ratio),
            "eval_threshold_mode": self.eval_threshold_mode,
            "eval_fixed_threshold": float(self.eval_fixed_threshold),
        }

    def _select_reference_threshold(self, labels: np.ndarray, scores: np.ndarray) -> float:
        if self.rerank_reference_threshold_mode == "val_best":
            threshold, _ = find_best_threshold(labels, scores)
            return float(threshold)
        return float(self.rerank_reference_fixed_threshold)

    def _compute_rerank_delta(self, batch, s_prior: torch.Tensor, *, reference_threshold: float = 0.5) -> torch.Tensor:
        if self.reranker is None or not self.reranker.enabled:
            return torch.zeros_like(s_prior)
        src_global, dst_global = self._edge_globals_from_batch(batch)
        src_degree, dst_degree = self._edge_degrees_from_globals(
            src_global=src_global,
            dst_global=dst_global,
            device=s_prior.device,
            dtype=s_prior.dtype,
        )
        return self.reranker.compute_raw_delta(
            s_prior=s_prior,
            pair_features=getattr(batch, "edge_pair_features", None),
            src_degree=src_degree,
            dst_degree=dst_degree,
            reference_threshold=float(reference_threshold),
        )

    def _compute_rerank_delta_from_replay_inputs(
        self,
        *,
        base_scores: np.ndarray,
        replay_inputs: Dict[str, np.ndarray],
        reference_threshold: float,
    ) -> np.ndarray:
        if self.reranker is None or not self.reranker.enabled:
            return np.zeros_like(base_scores, dtype=np.float32)
        pair_features = np.asarray(replay_inputs.get("pair_features", []), dtype=np.float32)
        if pair_features.size == 0:
            pair_features = np.zeros((len(base_scores), 0), dtype=np.float32)
        src_degree = np.asarray(replay_inputs.get("src_degree", []), dtype=np.float32)
        dst_degree = np.asarray(replay_inputs.get("dst_degree", []), dtype=np.float32)
        with torch.no_grad():
            delta = self.reranker.compute_raw_delta(
                s_prior=torch.as_tensor(base_scores, dtype=torch.float32),
                pair_features=torch.as_tensor(pair_features, dtype=torch.float32),
                src_degree=torch.as_tensor(src_degree, dtype=torch.float32),
                dst_degree=torch.as_tensor(dst_degree, dtype=torch.float32),
                reference_threshold=float(reference_threshold),
            )
        return delta.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _build_benefit_features_np(
        *,
        base_scores: np.ndarray,
        delta_scores: np.ndarray,
        replay_inputs: Dict[str, np.ndarray],
        reference_threshold: float,
    ) -> np.ndarray:
        base = np.asarray(base_scores, dtype=np.float32).reshape(-1, 1)
        delta = np.asarray(delta_scores, dtype=np.float32).reshape(-1, 1)
        projected = np.clip(base + delta, 0.0, 1.0)
        signed_dist = base - float(reference_threshold)
        abs_dist = np.abs(signed_dist)
        pair_features = np.asarray(replay_inputs.get("pair_features", []), dtype=np.float32)
        if pair_features.size == 0:
            pair_features = np.zeros((len(base), 0), dtype=np.float32)
        src_degree = np.asarray(replay_inputs.get("src_degree", []), dtype=np.float32).reshape(-1, 1)
        dst_degree = np.asarray(replay_inputs.get("dst_degree", []), dtype=np.float32).reshape(-1, 1)
        if src_degree.size == 0:
            src_degree = np.zeros((len(base), 1), dtype=np.float32)
        if dst_degree.size == 0:
            dst_degree = np.zeros((len(base), 1), dtype=np.float32)
        feature_blocks = [
            base,
            signed_dist,
            abs_dist,
            delta,
            np.abs(delta),
            projected,
            projected - float(reference_threshold),
            np.abs(projected - float(reference_threshold)),
            np.log1p(np.maximum(src_degree, 0.0)),
            np.log1p(np.maximum(dst_degree, 0.0)),
            pair_features,
        ]
        return np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _build_benefit_targets_np(
        *,
        base_scores: np.ndarray,
        delta_scores: np.ndarray,
        labels: np.ndarray,
        reference_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        labels_bool = np.asarray(labels, dtype=np.int32).reshape(-1).astype(bool)
        raw_pred = np.asarray(base_scores, dtype=np.float32).reshape(-1) >= float(reference_threshold)
        delta = np.asarray(delta_scores, dtype=np.float32).reshape(-1)
        eps = 1e-9
        move_toward = (labels_bool & (delta > eps)) | ((~labels_bool) & (delta < -eps))
        move_away = (labels_bool & (delta < -eps)) | ((~labels_bool) & (delta > eps))
        raw_wrong = raw_pred != labels_bool
        raw_correct = ~raw_wrong

        targets = np.zeros_like(delta, dtype=np.float32)
        targets[raw_wrong & move_toward] = 1.0

        weights = np.full_like(delta, 0.10, dtype=np.float32)
        weights[raw_wrong & move_toward] = 1.50
        weights[raw_correct & move_away] = 1.25
        weights[raw_wrong & move_away] = 0.75
        weights[raw_correct & move_toward] = 0.25
        weights *= (1.0 + np.maximum(0.0, 0.10 - np.abs(np.asarray(base_scores, dtype=np.float32) - float(reference_threshold))) / 0.10)

        meta = {
            "positive": int((targets > 0.5).sum()),
            "negative": int((targets <= 0.5).sum()),
            "raw_wrong": int(raw_wrong.sum()),
            "raw_correct": int(raw_correct.sum()),
            "move_toward": int(move_toward.sum()),
            "move_away": int(move_away.sum()),
        }
        return targets, weights, meta

    def _fit_benefit_selector(
        self,
        *,
        base_scores: np.ndarray,
        delta_scores: np.ndarray,
        labels: np.ndarray,
        replay_inputs: Dict[str, np.ndarray],
        reference_threshold: float,
    ) -> Tuple[Optional[LinearBenefitSelector], np.ndarray, Dict[str, object]]:
        if not self.rerank_benefit_enable:
            return None, np.ones_like(base_scores, dtype=np.float32), {"enabled": False}

        features = self._build_benefit_features_np(
            base_scores=base_scores,
            delta_scores=delta_scores,
            replay_inputs=replay_inputs,
            reference_threshold=float(reference_threshold),
        )
        targets, weights, target_meta = self._build_benefit_targets_np(
            base_scores=base_scores,
            delta_scores=delta_scores,
            labels=labels,
            reference_threshold=float(reference_threshold),
        )
        selector = LinearBenefitSelector(
            lr=self.rerank_benefit_lr,
            weight_decay=self.rerank_benefit_l2,
            epochs=self.rerank_benefit_train_epochs,
            seed=self.seed + 7919,
        )
        if not selector.fit(features, targets, sample_weight=weights):
            logger.warning("[RerankBenefit] selector skipped: insufficient positives/negatives on val.")
            return None, np.ones_like(base_scores, dtype=np.float32), {
                "enabled": False,
                "reason": "insufficient_labels",
                **target_meta,
            }
        probs = selector.predict_proba(features)
        return selector, probs, {
            "enabled": True,
            **selector.summary(),
            **target_meta,
            "prob_mean": float(np.mean(probs)) if len(probs) > 0 else 0.0,
            "prob_std": float(np.std(probs)) if len(probs) > 0 else 0.0,
        }

    def _predict_benefit_probs(
        self,
        *,
        selector: Optional[LinearBenefitSelector],
        base_scores: np.ndarray,
        delta_scores: np.ndarray,
        replay_inputs: Dict[str, np.ndarray],
        reference_threshold: float,
    ) -> np.ndarray:
        if selector is None:
            return np.ones_like(base_scores, dtype=np.float32)
        features = self._build_benefit_features_np(
            base_scores=base_scores,
            delta_scores=delta_scores,
            replay_inputs=replay_inputs,
            reference_threshold=float(reference_threshold),
        )
        return selector.predict_proba(features)

    def _task_residual_feature_names(self) -> List[str]:
        if self.rerank_em_residual_feature_names:
            return list(self.rerank_em_residual_feature_names)
        if self.target_task == "entity_matching":
            return ["s_prior__boundary", "row_value_jaccard__boundary"]
        return []

    def _task_residual_enabled(self) -> bool:
        if not self.rerank_em_residual_enable:
            return False
        return self.target_task == "entity_matching"

    def _build_task_residual_features_np(
        self,
        *,
        base_scores: np.ndarray,
        replay_inputs: Dict[str, np.ndarray],
        reference_threshold: float,
    ) -> np.ndarray:
        feature_names = self._task_residual_feature_names()
        if not feature_names:
            return np.zeros((len(base_scores), 0), dtype=np.float32)
        pair_features = np.asarray(replay_inputs.get("pair_features", []), dtype=np.float32)
        if pair_features.size == 0:
            pair_features = np.zeros((len(base_scores), len(self._current_pair_feature_order())), dtype=np.float32)
        evidence = build_evidence(
            task=self.target_task,
            s_prior=torch.as_tensor(base_scores, dtype=torch.float32),
            pair_feature_order=self._current_pair_feature_order(),
            pair_features=torch.as_tensor(pair_features, dtype=torch.float32),
            src_degree=torch.as_tensor(np.asarray(replay_inputs.get("src_degree", []), dtype=np.float32), dtype=torch.float32),
            dst_degree=torch.as_tensor(np.asarray(replay_inputs.get("dst_degree", []), dtype=np.float32), dtype=torch.float32),
        )
        boundary_focus = np.clip(
            1.0 - np.abs(np.asarray(base_scores, dtype=np.float32) - float(reference_threshold)) / 0.15,
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        columns: List[np.ndarray] = []
        for feature_name in feature_names:
            use_boundary = feature_name.endswith("__boundary")
            base_name = feature_name[:-10] if use_boundary else feature_name
            if base_name in evidence:
                values = evidence[base_name].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            else:
                values = np.zeros((len(base_scores),), dtype=np.float32)
            if use_boundary:
                values = values * boundary_focus
            columns.append(values.reshape(-1, 1))
        return np.concatenate(columns, axis=1).astype(np.float32, copy=False)

    def _fit_task_residual_head(
        self,
        *,
        base_scores: np.ndarray,
        labels: np.ndarray,
        replay_inputs: Dict[str, np.ndarray],
        reference_threshold: float,
        active_mask: np.ndarray,
    ) -> Tuple[Optional[LinearResidualHead], np.ndarray, Dict[str, object]]:
        if not self._task_residual_enabled():
            return None, np.zeros_like(base_scores, dtype=np.float32), {"enabled": False}
        features = self._build_task_residual_features_np(
            base_scores=base_scores,
            replay_inputs=replay_inputs,
            reference_threshold=float(reference_threshold),
        )
        if features.ndim != 2 or features.shape[1] == 0 or len(features) == 0:
            return None, np.zeros_like(base_scores, dtype=np.float32), {"enabled": False, "reason": "no_features"}
        targets = np.asarray(labels, dtype=np.float32).reshape(-1) - np.asarray(base_scores, dtype=np.float32).reshape(-1)
        untouched_boundary = (~np.asarray(active_mask, dtype=bool)) & (
            np.abs(np.asarray(base_scores, dtype=np.float32) - float(reference_threshold)) <= 0.10
        )
        current_pred = np.asarray(base_scores, dtype=np.float32) >= float(reference_threshold)
        current_wrong = current_pred != np.asarray(labels, dtype=np.float32).astype(bool)
        sample_weight = np.ones((len(base_scores),), dtype=np.float32)
        sample_weight = sample_weight + untouched_boundary.astype(np.float32) * 4.0 + current_wrong.astype(np.float32) * 2.0
        head = LinearResidualHead(l2=self.rerank_em_residual_l2)
        if not head.fit(features, targets, sample_weight=sample_weight):
            return None, np.zeros_like(base_scores, dtype=np.float32), {"enabled": False, "reason": "fit_failed"}
        preds = head.predict(features)
        return head, preds, {
            "enabled": True,
            **head.summary(),
            "feature_names": list(self._task_residual_feature_names()),
            "untouched_boundary_ratio": float(np.mean(untouched_boundary.astype(np.float32))) if len(untouched_boundary) > 0 else 0.0,
        }

    def _predict_task_residual_scores(
        self,
        *,
        head: Optional[LinearResidualHead],
        base_scores: np.ndarray,
        replay_inputs: Dict[str, np.ndarray],
        reference_threshold: float,
    ) -> np.ndarray:
        if head is None:
            return np.zeros_like(base_scores, dtype=np.float32)
        features = self._build_task_residual_features_np(
            base_scores=base_scores,
            replay_inputs=replay_inputs,
            reference_threshold=float(reference_threshold),
        )
        if features.ndim != 2 or features.shape[1] == 0:
            return np.zeros_like(base_scores, dtype=np.float32)
        return head.predict(features)

    @staticmethod
    def _apply_task_residual_np(
        base_scores: np.ndarray,
        residual_scores: np.ndarray,
        *,
        reference_threshold: float,
        active_mask: np.ndarray,
        residual_scale: float,
        residual_margin: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if float(residual_scale) <= 0.0 or float(residual_margin) <= 0.0:
            return np.asarray(base_scores, dtype=np.float32), np.zeros_like(base_scores, dtype=bool)
        residual_mask = (~np.asarray(active_mask, dtype=bool)) & (
            np.abs(np.asarray(base_scores, dtype=np.float32) - float(reference_threshold)) <= float(residual_margin)
        )
        final_scores = np.asarray(base_scores, dtype=np.float32) + np.asarray(residual_scores, dtype=np.float32) * float(
            residual_scale
        ) * residual_mask.astype(np.float32)
        return np.clip(final_scores, 0.0, 1.0), residual_mask

    def _search_best_task_residual_config(
        self,
        *,
        base_scores: np.ndarray,
        residual_scores: np.ndarray,
        labels: np.ndarray,
        reference_threshold: float,
        active_mask: np.ndarray,
    ) -> Dict[str, object]:
        best = {
            "enabled": False,
            "residual_scale": 0.0,
            "residual_margin": 0.0,
            "scores": np.asarray(base_scores, dtype=np.float32),
            "mask": np.zeros_like(base_scores, dtype=bool),
            "val_f1": _binary_classification_metrics(labels, base_scores, threshold=float(reference_threshold)).get("link_f1", 0.0),
            "val_threshold": float(reference_threshold),
            "fix_gap": 0,
            "breaks": 0,
        }
        raw_pred = (np.asarray(base_scores, dtype=np.float32) >= float(reference_threshold)).astype(np.int32)
        labels_int = np.asarray(labels, dtype=np.int32)
        for residual_scale in self.rerank_em_residual_scale_grid:
            for residual_margin in self.rerank_em_residual_margin_grid:
                scores_final, residual_mask = self._apply_task_residual_np(
                    base_scores,
                    residual_scores,
                    reference_threshold=float(reference_threshold),
                    active_mask=active_mask,
                    residual_scale=float(residual_scale),
                    residual_margin=float(residual_margin),
                )
                if self.eval_threshold_mode == "val_best":
                    val_threshold, val_f1 = find_best_threshold(labels, scores_final)
                else:
                    val_threshold = float(self.eval_fixed_threshold)
                    val_f1 = _binary_classification_metrics(labels, scores_final, threshold=val_threshold).get("link_f1", 0.0)
                final_pred = (scores_final >= float(val_threshold)).astype(np.int32)
                fixes = int(((raw_pred != labels_int) & (final_pred == labels_int)).sum())
                breaks = int(((raw_pred == labels_int) & (final_pred != labels_int)).sum())
                fix_gap = fixes - breaks
                current = (float(val_f1), int(fix_gap), -int(breaks), -float(np.mean(residual_mask.astype(np.float32))))
                best_key = (
                    float(best["val_f1"]),
                    int(best["fix_gap"]),
                    -int(best["breaks"]),
                    -float(np.mean(np.asarray(best["mask"], dtype=np.float32))) if len(best["mask"]) > 0 else 0.0,
                )
                if current > best_key:
                    best = {
                        "enabled": float(residual_scale) > 0.0 and float(residual_margin) > 0.0,
                        "residual_scale": float(residual_scale),
                        "residual_margin": float(residual_margin),
                        "scores": np.asarray(scores_final, dtype=np.float32),
                        "mask": np.asarray(residual_mask, dtype=bool),
                        "val_f1": float(val_f1),
                        "val_threshold": float(val_threshold),
                        "fix_gap": int(fix_gap),
                        "breaks": int(breaks),
                    }
        return best

    @staticmethod
    def _compute_benefit_mask_np(*, benefit_probs: Optional[np.ndarray], benefit_min_score: float) -> np.ndarray:
        if benefit_probs is None or float(benefit_min_score) <= 0.0:
            if benefit_probs is None:
                raise ValueError("benefit_probs must be provided when benefit selector is enabled.")
            return np.ones_like(benefit_probs, dtype=bool)
        return np.asarray(benefit_probs, dtype=np.float32) >= float(benefit_min_score)

    @staticmethod
    def _apply_rerank_np(
        base_scores: np.ndarray,
        delta_scores: np.ndarray,
        *,
        alpha: float,
        delta_cap: float,
        gate_margin: float,
        gate_delta_min: float,
        reference_threshold: float,
        benefit_probs: Optional[np.ndarray] = None,
        benefit_min_score: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        clipped = np.clip(delta_scores, -float(delta_cap), float(delta_cap))
        gate_mask = GraphLinkPredictionTrainer._compute_gate_mask_np(
            base_scores=base_scores,
            delta_scores=delta_scores,
            gate_margin=gate_margin,
            gate_delta_min=gate_delta_min,
            reference_threshold=reference_threshold,
        )
        if benefit_probs is None:
            benefit_mask = np.ones_like(base_scores, dtype=bool)
        else:
            benefit_mask = GraphLinkPredictionTrainer._compute_benefit_mask_np(
                benefit_probs=benefit_probs,
                benefit_min_score=float(benefit_min_score),
            )
        active_mask = gate_mask & benefit_mask
        final_scores = np.clip(base_scores + float(alpha) * clipped * active_mask.astype(base_scores.dtype), 0.0, 1.0)
        return final_scores, gate_mask, active_mask

    @staticmethod
    def _compute_gate_mask_np(
        *,
        base_scores: np.ndarray,
        delta_scores: np.ndarray,
        gate_margin: float,
        gate_delta_min: float,
        reference_threshold: float,
    ) -> np.ndarray:
        mask = np.ones_like(base_scores, dtype=bool)
        if float(gate_margin) < 1.0:
            mask &= np.abs(base_scores - float(reference_threshold)) <= float(gate_margin)
        if float(gate_delta_min) > 0.0:
            mask &= np.abs(delta_scores) >= float(gate_delta_min)
        return mask

    def _resolve_replay_archive_dir(self) -> str:
        if not self.replay_archive_root:
            return ""
        root = os.path.abspath(self.replay_archive_root)
        run_tag = _safe_token(self.run_tag, default="run")
        dataset = _safe_token(self.dataset_name, default="dataset")
        task = _safe_token(self.target_task, default="task")
        seed = f"seed_{int(self.seed)}"
        return os.path.join(root, run_tag, dataset, task, seed)

    def _resolve_replay_archive_dir_for_track(self, track: str) -> str:
        if not self.replay_archive_root:
            return ""
        root = os.path.abspath(self.replay_archive_root)
        run_tag = _safe_token(self.run_tag, default="run")
        dataset = _safe_token(self.dataset_name, default="dataset")
        task = _safe_token(self.target_task, default="task")
        seed = f"seed_{int(self.seed)}"
        token = str(track).strip().lower()
        if not token:
            token = "full_replay"
        return os.path.join(root, token, run_tag, dataset, task, seed)

    def _search_best_rerank_config(
        self,
        *,
        base_scores: np.ndarray,
        delta_scores: np.ndarray,
        labels: np.ndarray,
        reference_threshold: float,
        benefit_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        alpha_grid = [float(v) for v in self.rerank_alpha_grid if float(v) > 0.0]
        delta_cap_grid = [float(v) for v in self.rerank_delta_cap_grid]
        gate_margin_grid = [float(v) for v in self.rerank_gate_margin_grid]
        gate_delta_min_grid = [float(v) for v in self.rerank_gate_delta_min_grid]
        benefit_threshold_grid = [0.0]
        if benefit_probs is not None and self.rerank_benefit_enable:
            benefit_threshold_grid = sorted({max(0.0, float(v)) for v in self.rerank_benefit_threshold_grid})
        if not alpha_grid:
            raise ValueError("rerank_alpha_grid must contain at least one positive value when reranking is enabled")
        if not delta_cap_grid:
            raise ValueError("rerank_delta_cap_grid must contain at least one value when reranking is enabled")
        if not gate_margin_grid:
            raise ValueError("rerank_gate_margin_grid must contain at least one value when reranking is enabled")
        if not gate_delta_min_grid:
            raise ValueError("rerank_gate_delta_min_grid must contain at least one value when reranking is enabled")

        best = None
        best_any = None
        for delta_cap in delta_cap_grid:
            for alpha in alpha_grid:
                for gate_margin in gate_margin_grid:
                    for gate_delta_min in gate_delta_min_grid:
                        for benefit_min_score in benefit_threshold_grid:
                            final_scores, gate_mask, active_mask = self._apply_rerank_np(
                                base_scores,
                                delta_scores,
                                alpha=float(alpha),
                                delta_cap=float(delta_cap),
                                gate_margin=float(gate_margin),
                                gate_delta_min=float(gate_delta_min),
                                reference_threshold=float(reference_threshold),
                                benefit_probs=benefit_probs,
                                benefit_min_score=float(benefit_min_score),
                            )
                            gate_ratio = float(np.mean(gate_mask.astype(np.float32))) if len(gate_mask) > 0 else 0.0
                            benefit_ratio = 1.0
                            if benefit_probs is not None:
                                benefit_mask = self._compute_benefit_mask_np(
                                    benefit_probs=benefit_probs,
                                    benefit_min_score=float(benefit_min_score),
                                )
                                benefit_ratio = float(np.mean(benefit_mask.astype(np.float32))) if len(benefit_mask) > 0 else 0.0
                            active_ratio = float(np.mean(active_mask.astype(np.float32))) if len(active_mask) > 0 else 0.0
                            if self.eval_threshold_mode == "val_best":
                                val_threshold, val_f1 = find_best_threshold(labels, final_scores)
                            else:
                                val_threshold = float(self.eval_fixed_threshold)
                                val_f1 = _binary_classification_metrics(labels, final_scores, threshold=val_threshold).get(
                                    "link_f1", 0.0
                                )
                            labels_int = labels.astype(np.int32, copy=False)
                            raw_pred = (base_scores >= float(val_threshold)).astype(np.int32)
                            final_pred = (final_scores >= float(val_threshold)).astype(np.int32)
                            fixes = int(((raw_pred != labels_int) & (final_pred == labels_int)).sum())
                            breaks = int(((raw_pred == labels_int) & (final_pred != labels_int)).sum())
                            fix_gap = int(fixes - breaks)
                            changed_ratio = float(np.mean(raw_pred != final_pred)) if len(final_pred) > 0 else 0.0
                            current = (
                                float(val_f1),
                                fix_gap,
                                -breaks,
                                changed_ratio,
                                active_ratio,
                                -float(benefit_min_score),
                                -float(gate_delta_min),
                                float(gate_margin),
                            )
                            best_key = None
                            if best is not None:
                                best_key = (
                                    float(best["val_f1"]),
                                    int(best["val_fix_gap"]),
                                    -int(best["val_breaks"]),
                                    float(best["val_changed_ratio"]),
                                    float(best["active_ratio"]),
                                    -float(best["benefit_min_score"]),
                                    -float(best["gate_delta_min"]),
                                    float(best["gate_margin"]),
                                )
                            candidate = {
                                "enabled": True,
                                "alpha": float(alpha),
                                "delta_cap": float(delta_cap),
                                "gate_margin": float(gate_margin),
                                "gate_delta_min": float(gate_delta_min),
                                "gate_ratio": float(gate_ratio),
                                "benefit_min_score": float(benefit_min_score),
                                "benefit_ratio": float(benefit_ratio),
                                "active_ratio": float(active_ratio),
                                "reference_threshold": float(reference_threshold),
                                "val_f1": float(val_f1),
                                "val_fix_gap": int(fix_gap),
                                "val_breaks": int(breaks),
                                "val_changed_ratio": float(changed_ratio),
                                "val_threshold": float(val_threshold),
                                "scores": final_scores,
                                "active_ratio_cap_satisfied": float(active_ratio) <= float(self.rerank_max_active_ratio) + 1e-8,
                            }
                            best_any_key = None
                            if best_any is not None:
                                best_any_key = (
                                    float(best_any["val_f1"]),
                                    int(best_any["val_fix_gap"]),
                                    -int(best_any["val_breaks"]),
                                    float(best_any["val_changed_ratio"]),
                                    float(best_any["active_ratio"]),
                                    -float(best_any["benefit_min_score"]),
                                    -float(best_any["gate_delta_min"]),
                                    float(best_any["gate_margin"]),
                                )
                            if best_any is None or current > best_any_key:
                                best_any = candidate
                            if not candidate["active_ratio_cap_satisfied"]:
                                continue
                            if best is None or current > best_key:
                                best = candidate
        if best is not None:
            return best
        assert best_any is not None
        if float(self.rerank_max_active_ratio) < 1.0:
            logger.warning(
                "[RerankGate] no candidate satisfied active_ratio<=%.4f; falling back to unconstrained best with active_ratio=%.4f",
                float(self.rerank_max_active_ratio),
                float(best_any["active_ratio"]),
            )
        return best_any

    def _pair_feature_store_summary(self) -> Dict[str, object]:
        store = self.em_pair_feature_store
        store_type = "em"
        if store is None:
            store = self.jts_pair_feature_store
            store_type = "jts"
        if store is None:
            store = self.sm_pair_feature_store
            store_type = "sm"
        if store is None:
            store = self.uts_pair_feature_store
            store_type = "uts"
        if store is None:
            return {"pair_feature_store": "disabled"}

        summary: Dict[str, object] = {
            "pair_feature_store": store_type,
            "feature_dim": int(getattr(store, "feature_dim", 0)),
            "feature_order_len": self._safe_len(getattr(store, "feature_order", [])),
            "node_id_to_column_count": self._safe_len(getattr(store, "node_id_to_column", {})),
            "table_cache_tables": self._safe_len(getattr(store, "_table_cache", {})),
            "pair_cache_pairs": self._safe_len(getattr(store, "_pair_cache", {})),
            "table_root": getattr(store, "table_root", ""),
            "online_symbolic_enabled": bool(self._online_symbolic_enabled),
            "online_symbolic_spec_id": str(self._online_symbolic_spec_id),
            "online_symbolic_repr": str(self._online_symbolic_repr_effective),
            "online_symbolic_raw_dim": int(self._online_symbolic_raw_dim),
            "online_symbolic_tile_repeat": int(self.online_symbolic_tile_repeat),
            "online_symbolic_effective_dim": int(self._online_symbolic_effective_dim),
            "online_symbolic_normalize": str(self.online_symbolic_normalize),
        }
        return summary

    def _prepare_em_split_feature_cache(
        self,
        *,
        train_edges: List[Tuple[int, int]],
        val_edges: List[Tuple[int, int]],
        test_edges: List[Tuple[int, int]],
    ) -> None:
        self._em_feature_build_sec = 0.0
        if not (
            self.target_task == "entity_matching"
            and self.em_pair_feature_store is not None
            and self.em_pair_cache_mode == "readwrite"
        ):
            return
        start = time.time()
        split_edges = {
            "train": train_edges,
            "val": val_edges,
            "test": test_edges,
        }
        for split, edges in split_edges.items():
            self.em_pair_feature_store.build_or_load_split_matrix(split=split, edges=edges)
        self._em_feature_build_sec = float(time.time() - start)
        stats = self.em_pair_feature_store.get_cache_stats()
        split_hit = float(stats.get("split_cache_hit", 0))
        split_miss = float(stats.get("split_cache_miss", 0))
        pair_hit = float(stats.get("pair_cache_hit", 0))
        pair_miss = float(stats.get("pair_cache_miss", 0))
        split_den = split_hit + split_miss
        pair_den = pair_hit + pair_miss
        split_ratio = float(split_hit / split_den) if split_den > 0 else 0.0
        pair_ratio = float(pair_hit / pair_den) if pair_den > 0 else 0.0
        logger.info(
            "[EM-Diag] feature_build_sec=%.3f cache_hit_ratio=%.4f pair_cache_hit_ratio=%.4f",
            float(self._em_feature_build_sec),
            float(split_ratio),
            float(pair_ratio),
        )

    def _component_summary(self, stage: str) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "stage": stage,
            "target_task": self.target_task,
            "device": str(self.device),
            "batch_size": int(self.batch_size),
            "num_workers": int(self.num_workers),
            "graph_num_nodes": int(getattr(self.graph, "num_nodes", 0)),
            "graph_num_edges": int(getattr(self.graph, "num_edges", 0)),
            "prepool_rows": bool(self.prepool_rows),
            "prepool_cols": bool(self.prepool_cols),
            "graph_x_bytes": tensor_nbytes(getattr(self.graph_data, "x", None)),
            "graph_edge_index_bytes": tensor_nbytes(getattr(self.graph_data, "edge_index", None)),
            "graph_edge_attr_bytes": tensor_nbytes(getattr(self.graph_data, "edge_attr", None)),
            "model_param_bytes": int(sum(tensor_nbytes(param) for param in self.model.parameters())),
            "optimizer_state_bytes": int(self._optimizer_state_bytes()),
            "train_edges": int(len(self._train_edges)),
            "val_edges": int(len(self._val_edges)),
            "test_edges": int(len(self._test_edges)),
        }
        if self._prepool_overlay is not None:
            summary.update(
                {
                    "prepool_row_map_bytes": tensor_nbytes(self._prepool_overlay.get("row_map")),
                    "prepool_row_emb_bytes": tensor_nbytes(self._prepool_overlay.get("row_emb")),
                    "prepool_col_map_bytes": tensor_nbytes(self._prepool_overlay.get("col_map")),
                    "prepool_col_emb_bytes": tensor_nbytes(self._prepool_overlay.get("col_emb")),
                }
            )
        else:
            summary.update(
                {
                    "prepool_row_map_bytes": 0,
                    "prepool_row_emb_bytes": 0,
                    "prepool_col_map_bytes": 0,
                    "prepool_col_emb_bytes": 0,
                }
            )
        summary.update(self._pair_feature_store_summary())
        summary.update(self._rerank_summary())
        return summary

    def _record_component_summary(self, stage: str) -> None:
        record_profile_event("trainer_components", self._component_summary(stage=stage))

    def prepare_graph_data(self, graph):
        """Prepares the PyG Data object."""
        logger.info("Preparing graph data for PyG loaders (cell-prepool + edge filtering)...")

        node_embeddings = _get_node_embeddings_cached(graph)
        if self.prepool_cells:
            logger.info("[Prepool] Using overlay updates on batch.x (row/col only).")

        edge_index, edge_attr = graph.get_edge_index()
        edge_index = edge_index.contiguous()
        edge_attr = edge_attr.contiguous()

        if self.drop_cell_edges:
            drop_ids: List[int] = []
            for name in ("cell_row", "cell_column"):
                if name in graph.edge_features_map:
                    drop_ids.append(int(graph.edge_features_map[name]))
            before = int(edge_index.size(1))
            edge_index, edge_attr = _filter_edges_by_type_ids(edge_index, edge_attr, drop_type_ids=drop_ids)
            after = int(edge_index.size(1))
            logger.info(f"[Graph] drop_cell_edges=1 removed {before - after} edges (kept={after})")

        # JTS fix: avoid message-passing leakage from the target relation itself.
        # Keep supervision pairs in edge_label_index, but remove JTS relation edges
        # from the propagation graph used by LinkNeighborLoader.
        if self.target_task == "joinable_table_search":
            jts_type_id = graph.edge_type_to_id.get("joinable_table_search", None)
            if jts_type_id is not None:
                before = int(edge_index.size(1))
                edge_index, edge_attr = _filter_edges_by_type_ids(
                    edge_index,
                    edge_attr,
                    drop_type_ids=[int(jts_type_id)],
                )
                after = int(edge_index.size(1))
                logger.info(
                    "[Graph] jts_leakage_guard=1 removed_target_edges=%d (kept=%d)",
                    before - after,
                    after,
                )

        self.graph_data = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    def _loader_generator(self, split: str) -> torch.Generator:
        split_offsets = {"train": 101, "val": 202, "test": 303}
        gen = torch.Generator()
        gen.manual_seed(self.seed + split_offsets.get(split, 0))
        return gen

    def _worker_init_fn(self, split: str):
        split_offsets = {"train": 1001, "val": 2001, "test": 3001}
        base_seed = int(self.seed) + split_offsets.get(split, 0)

        def _init(worker_id: int) -> None:
            worker_seed = base_seed + int(worker_id)
            random.seed(worker_seed)
            np.random.seed(worker_seed % (2**32 - 1))
            torch.manual_seed(worker_seed)

        return _init

    def _make_link_loader(
        self,
        *,
        split: str,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        shuffle: bool,
        num_workers: int,
    ) -> LinkNeighborLoader:
        return LinkNeighborLoader(
            self.graph_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            worker_init_fn=self._worker_init_fn(split),
            generator=self._loader_generator(split),
            neg_sampling_ratio=0.0,
        )

    def _fit_em_pair_feature_normalizer(self, train_edges: List[Tuple[int, int]]) -> None:
        self._em_pair_feat_mean = None
        self._em_pair_feat_std = None
        if not (
            self.target_task == "entity_matching"
            and self.em_pair_feat_norm
            and self.em_pair_feature_store is not None
            and len(train_edges) > 0
            and int(self.em_pair_feature_store.feature_dim) > 0
        ):
            return

        chunk_size = 2048
        dim = int(self.em_pair_feature_store.feature_dim)
        feat_sum = torch.zeros((dim,), dtype=torch.float64)
        feat_sumsq = torch.zeros((dim,), dtype=torch.float64)
        feat_count = 0

        for start in range(0, len(train_edges), chunk_size):
            chunk = train_edges[start : start + chunk_size]
            src_ids = torch.tensor([int(pair[0]) for pair in chunk], dtype=torch.long)
            dst_ids = torch.tensor([int(pair[1]) for pair in chunk], dtype=torch.long)
            feats = self.em_pair_feature_store.build_batch_features(
                src_ids,
                dst_ids,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            if feats.numel() == 0:
                continue
            feats64 = feats.to(dtype=torch.float64)
            feat_sum += feats64.sum(dim=0)
            feat_sumsq += (feats64 * feats64).sum(dim=0)
            feat_count += int(feats64.size(0))

        if feat_count <= 0:
            return
        mean = feat_sum / float(feat_count)
        var = feat_sumsq / float(feat_count) - mean * mean
        var = torch.clamp(var, min=1e-12)
        std = torch.sqrt(var)
        self._em_pair_feat_mean = mean.to(dtype=torch.float32)
        self._em_pair_feat_std = std.to(dtype=torch.float32)
        logger.info(
            "[EM] pair feature normalization fitted: dim=%d samples=%d",
            dim,
            feat_count,
        )

    def _normalize_em_pair_features(self, pair_feats: torch.Tensor) -> torch.Tensor:
        if not (
            self.target_task == "entity_matching"
            and self.em_pair_feat_norm
            and self._em_pair_feat_mean is not None
            and self._em_pair_feat_std is not None
            and pair_feats.numel() > 0
        ):
            return pair_feats
        mean = self._em_pair_feat_mean.to(device=pair_feats.device, dtype=pair_feats.dtype)
        std = self._em_pair_feat_std.to(device=pair_feats.device, dtype=pair_feats.dtype)
        return (pair_feats - mean) / std

    def _compute_online_symbolic_from_pair(
        self,
        pair_feats: torch.Tensor,
        *,
        apply_norm: bool,
        apply_tile: bool,
    ) -> torch.Tensor:
        if not self._online_symbolic_enabled or self._online_symbolic_executor is None:
            return torch.zeros((int(pair_feats.shape[0]), 0), dtype=pair_feats.dtype, device=pair_feats.device)
        if pair_feats.ndim != 2:
            raise ValueError(f"pair_feats must be 2D, got shape={tuple(pair_feats.shape)}")
        if pair_feats.shape[0] <= 0:
            return torch.zeros((0, int(self._online_symbolic_effective_dim)), dtype=pair_feats.dtype, device=pair_feats.device)

        feats_np = pair_feats.detach().cpu().numpy().astype(np.float32, copy=False)
        fmap = build_feature_map_from_matrix(
            pair_features=feats_np,
            pair_feature_order=self._online_symbolic_pair_feature_order,
            extras=None,
        )
        for name in self._online_symbolic_expected_features:
            if name not in fmap:
                fmap[name] = np.zeros((int(feats_np.shape[0]),), dtype=np.float64)
        if self._online_symbolic_repr_effective == "concat":
            sym = self._online_symbolic_executor.run_channels(fmap, apply_range_hint=True)
        else:
            sym = self._online_symbolic_executor.run_score(fmap, apply_range_hint=True).reshape(-1, 1)
        sym_np = np.asarray(sym, dtype=np.float32)

        if (
            apply_norm
            and self.online_symbolic_normalize == "zscore"
            and self._online_symbolic_norm_mean is not None
            and self._online_symbolic_norm_std is not None
        ):
            mean = self._online_symbolic_norm_mean.detach().cpu().numpy().reshape(1, -1).astype(np.float32, copy=False)
            std = self._online_symbolic_norm_std.detach().cpu().numpy().reshape(1, -1).astype(np.float32, copy=False)
            sym_np = (sym_np - mean) / std

        if apply_tile and int(self.online_symbolic_tile_repeat) > 1:
            sym_np = np.tile(sym_np, (1, int(self.online_symbolic_tile_repeat)))

        return torch.as_tensor(sym_np, dtype=pair_feats.dtype, device=pair_feats.device)

    def _fit_online_symbolic_normalizer(self, train_edges: List[Tuple[int, int]]) -> None:
        self._online_symbolic_norm_mean = None
        self._online_symbolic_norm_std = None
        if not (
            self._online_symbolic_enabled
            and self.online_symbolic_normalize == "zscore"
            and len(train_edges) > 0
        ):
            return
        store = self._active_pair_feature_store()
        if store is None:
            return

        chunk_size = 2048
        sym_sum: Optional[torch.Tensor] = None
        sym_sumsq: Optional[torch.Tensor] = None
        sym_count = 0

        for start in range(0, len(train_edges), chunk_size):
            chunk = train_edges[start : start + chunk_size]
            src_ids = torch.tensor([int(pair[0]) for pair in chunk], dtype=torch.long)
            dst_ids = torch.tensor([int(pair[1]) for pair in chunk], dtype=torch.long)
            feats = store.build_batch_features(
                src_ids,
                dst_ids,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            if feats.numel() == 0:
                continue
            feats = self._normalize_em_pair_features(feats)
            sym = self._compute_online_symbolic_from_pair(
                feats,
                apply_norm=False,
                apply_tile=False,
            )
            if sym.numel() == 0:
                continue
            sym64 = sym.to(dtype=torch.float64, device=torch.device("cpu"))
            if sym_sum is None:
                sym_sum = torch.zeros((int(sym64.shape[1]),), dtype=torch.float64)
                sym_sumsq = torch.zeros((int(sym64.shape[1]),), dtype=torch.float64)
            sym_sum += sym64.sum(dim=0)
            assert sym_sumsq is not None
            sym_sumsq += (sym64 * sym64).sum(dim=0)
            sym_count += int(sym64.shape[0])

        if sym_sum is None or sym_sumsq is None or sym_count <= 0:
            return

        mean = sym_sum / float(sym_count)
        var = sym_sumsq / float(sym_count) - mean * mean
        var = torch.clamp(var, min=1e-12)
        std = torch.sqrt(var)
        self._online_symbolic_norm_mean = mean.to(dtype=torch.float32)
        self._online_symbolic_norm_std = std.to(dtype=torch.float32)
        logger.info(
            "[OnlineSymbolic] fitted train-only zscore: task=%s dim=%d samples=%d spec_id=%s",
            str(self.target_task),
            int(mean.shape[0]),
            int(sym_count),
            self._online_symbolic_spec_id or "unknown",
        )

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_weight = getattr(self.link_loss, "pos_weight", None)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight,
        )
        prob = torch.sigmoid(logits)
        pt = prob * labels + (1.0 - prob) * (1.0 - labels)
        alpha_t = labels * float(self.em_focal_alpha) + (1.0 - labels) * (1.0 - float(self.em_focal_alpha))
        focal_factor = torch.pow(torch.clamp(1.0 - pt, min=1e-6), float(self.em_focal_gamma))
        return torch.mean(alpha_t * focal_factor * bce)

    def _compute_train_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        logits = outputs["link_logits"]
        if self.target_task != "entity_matching":
            return self.link_loss(logits, labels)

        if self.em_loss == "focal":
            return self._focal_loss(logits, labels)

        bce = self.link_loss(logits, labels)
        if self.em_loss == "bce_contrastive":
            contrastive = self.contrastive_loss(outputs["projections"], labels)
            return bce + float(self.em_contrastive_weight) * contrastive
        return bce

    def _select_hard_negative_edges(self) -> torch.Tensor:
        empty = torch.empty((2, 0), dtype=torch.long)
        if not (
            self.target_task == "entity_matching"
            and self.em_hard_neg_ratio > 0.0
            and self._train_edge_label_index_base is not None
            and self._train_edge_label_base is not None
        ):
            return empty

        labels = self._train_edge_label_base
        pos_cnt = int((labels == 1).sum().item())
        if pos_cnt <= 0:
            return empty
        target_k = int(round(float(pos_cnt) * float(self.em_hard_neg_ratio)))
        if target_k <= 0:
            return empty

        neg_mask = labels == 0
        if not bool(neg_mask.any()):
            return empty
        neg_edge_index = self._train_edge_label_index_base[:, neg_mask].contiguous()
        if neg_edge_index.numel() == 0:
            return empty

        candidate_labels = torch.zeros((neg_edge_index.size(1),), dtype=torch.float32)
        candidate_loader = self._make_link_loader(
            split="train",
            edge_label_index=neg_edge_index,
            edge_label=candidate_labels,
            shuffle=False,
            num_workers=self._loader_num_workers,
        )

        self.model.eval()
        all_probs: List[torch.Tensor] = []
        all_src: List[torch.Tensor] = []
        all_dst: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                candidate_loader,
                desc="Mining hard negatives",
                leave=False,
                disable=TQDM_DISABLE,
            ):
                batch = batch.to(self.device)
                self._apply_prepool_overlay(batch)
                self._attach_task_pair_features(batch)
                outputs = self.model(batch)
                probs = torch.sigmoid(outputs["link_logits"]).view(-1).detach().cpu()
                if probs.numel() == 0:
                    continue
                src_global = batch.n_id[batch.edge_label_index[0]].detach().cpu()
                dst_global = batch.n_id[batch.edge_label_index[1]].detach().cpu()
                all_probs.append(probs)
                all_src.append(src_global)
                all_dst.append(dst_global)

        if not all_probs:
            return empty
        probs = torch.cat(all_probs, dim=0)
        src = torch.cat(all_src, dim=0)
        dst = torch.cat(all_dst, dim=0)
        if probs.numel() == 0:
            return empty

        k = min(int(target_k), int(probs.numel()))
        if k <= 0:
            return empty
        topk_idx = torch.topk(probs, k, largest=True).indices
        selected = torch.stack([src[topk_idx], dst[topk_idx]], dim=0).to(dtype=torch.long)
        logger.info(
            "[EM] hard negatives selected: target=%d selected=%d candidates=%d",
            int(target_k),
            int(k),
            int(probs.numel()),
        )
        return selected.contiguous()

    def _rebuild_train_loader_with_extra_negatives(self, extra_neg_edge_index: Optional[torch.Tensor]) -> None:
        if self._train_edge_label_index_base is None or self._train_edge_label_base is None:
            return
        base_index = self._train_edge_label_index_base
        base_label = self._train_edge_label_base
        if extra_neg_edge_index is None or extra_neg_edge_index.numel() == 0:
            active_index = base_index
            active_label = base_label
            added = 0
        else:
            extra_neg_edge_index = extra_neg_edge_index.to(dtype=torch.long, device=base_index.device).contiguous()
            extra_label = torch.zeros((extra_neg_edge_index.size(1),), dtype=base_label.dtype, device=base_label.device)
            active_index = torch.cat([base_index, extra_neg_edge_index], dim=1).contiguous()
            active_label = torch.cat([base_label, extra_label], dim=0).contiguous()
            added = int(extra_neg_edge_index.size(1))

        self._train_edge_label_index_active = active_index
        self._train_edge_label_active = active_label
        self.train_loader = self._make_link_loader(
            split="train",
            edge_label_index=active_index,
            edge_label=active_label,
            shuffle=True,
            num_workers=self._loader_num_workers,
        )
        logger.info(
            "[EM] train loader refreshed: base_edges=%d added_hard_neg=%d total_edges=%d",
            int(base_index.size(1)),
            int(added),
            int(active_index.size(1)),
        )

    def create_dataloaders(self, num_workers=4):
        """Creates efficient dataloaders using LinkNeighborLoader."""
        logger.info("Creating parallelized dataloaders...")
        self._loader_num_workers = int(num_workers)
        with profile_phase("trainer.collect_supervision_edges", {"target_task": self.target_task}):
            train_edges, train_labels, train_edge_ids = self.graph.get_train_edges(self.target_task)
            val_edges, val_labels, val_edge_ids = self.graph.get_val_edges(self.target_task)
            test_edges, test_labels, test_edge_ids = self.graph.get_test_edges(self.target_task)

        train_edges, train_labels, train_edge_ids = self._apply_debug_edge_cap(
            split="train",
            edges=train_edges,
            labels=train_labels,
            edge_ids=train_edge_ids,
        )
        val_edges, val_labels, val_edge_ids = self._apply_debug_edge_cap(
            split="val",
            edges=val_edges,
            labels=val_labels,
            edge_ids=val_edge_ids,
        )
        test_edges, test_labels, test_edge_ids = self._apply_debug_edge_cap(
            split="test",
            edges=test_edges,
            labels=test_labels,
            edge_ids=test_edge_ids,
        )

        self._train_edges = train_edges
        self._val_edges = val_edges
        self._test_edges = test_edges

        with profile_phase("trainer.materialize_label_tensors", {"target_task": self.target_task}):
            train_edge_label_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
            train_edge_label = torch.tensor(train_labels, dtype=torch.float)

            val_edge_label_index = torch.tensor(val_edges, dtype=torch.long).t().contiguous()
            val_edge_label = torch.tensor(val_labels, dtype=torch.float)

            test_edge_label_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
            test_edge_label = torch.tensor(test_labels, dtype=torch.float)

        self._train_edge_label_index_base = train_edge_label_index
        self._train_edge_label_base = train_edge_label

        if self.target_task == "entity_matching" and self.em_auto_pos_weight and train_edge_label.numel() > 0:
            pos = int((train_edge_label == 1).sum().item())
            neg = int((train_edge_label == 0).sum().item())
            if pos > 0 and neg > 0:
                ratio = float(neg) / float(pos)
                ratio = max(1.0, ratio)
                if self.em_pos_weight_cap > 0.0:
                    ratio = min(ratio, float(self.em_pos_weight_cap))
                self._em_train_pos_weight = float(ratio)
                self.link_loss = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([self._em_train_pos_weight], dtype=torch.float32, device=self.device)
                )
                logger.info(
                    "[EM] auto pos_weight enabled: pos=%d neg=%d pos_weight=%.4f (cap=%.3f)",
                    pos,
                    neg,
                    self._em_train_pos_weight,
                    self.em_pos_weight_cap,
                )

        self._prepare_em_split_feature_cache(
            train_edges=train_edges,
            val_edges=val_edges,
            test_edges=test_edges,
        )
        self._fit_em_pair_feature_normalizer(train_edges)
        self._fit_online_symbolic_normalizer(train_edges)
        if self.target_task == "entity_matching" and self.em_pair_feature_store is not None:
            feat_diag = self.em_pair_feature_store.get_feature_hit_diagnostics(
                train_edges,
                sample_size=2048,
                seed=self.seed + 17,
            )
            logger.info("[EM-PairFeat] diagnostics: %s", json.dumps(feat_diag, ensure_ascii=False, sort_keys=True))

        # Create the loaders
        with profile_phase("trainer.create_loaders", {"target_task": self.target_task, "num_workers": int(num_workers)}):
            self._rebuild_train_loader_with_extra_negatives(None)
            self.val_loader = self._make_link_loader(
                split="val",
                edge_label_index=val_edge_label_index,
                edge_label=val_edge_label,
                shuffle=False,
                num_workers=num_workers,
            )
            self.test_loader = self._make_link_loader(
                split="test",
                edge_label_index=test_edge_label_index,
                edge_label=test_edge_label,
                shuffle=False,
                num_workers=num_workers,
            )
        record_profile_event(
            "loader_stats",
            {
                "target_task": self.target_task,
                "train_edges": int(len(train_edges)),
                "val_edges": int(len(val_edges)),
                "test_edges": int(len(test_edges)),
                "em_train_pos_weight": float(self._em_train_pos_weight),
                "train_index_bytes": tensor_nbytes(train_edge_label_index),
                "val_index_bytes": tensor_nbytes(val_edge_label_index),
                "test_index_bytes": tensor_nbytes(test_edge_label_index),
                "train_label_bytes": tensor_nbytes(train_edge_label),
                "val_label_bytes": tensor_nbytes(val_edge_label),
                "test_label_bytes": tensor_nbytes(test_edge_label),
            },
        )
        self._record_component_summary(stage="post_dataloaders")
        logger.info("Dataloaders created.")

    def _get_joinable_col_col_edges(self, split: str) -> Tuple[List[Tuple[int, int]], List[int], List[str]]:
        if split == "train":
            mask = self.graph.train_mask
        elif split == "val":
            mask = self.graph.val_mask
        elif split == "test":
            mask = self.graph.test_mask
        else:
            raise ValueError(f"Unknown split={split}. Expected one of: train,val,test")

        edges: List[Tuple[int, int]] = []
        labels: List[int] = []
        types: List[str] = []

        for i, keep in enumerate(mask):
            if keep != 1:
                continue
            if self.graph.edge_types_data[i] != "joinable_table_search":
                continue
            nodes = self.graph.edge_lists[i]
            if len(nodes) < 2:
                continue
            edges.append((int(nodes[0]), int(nodes[1])))
            labels.append(int(self.graph.edge_labels[i]))
            types.append("joinable_table_search")

        logger.info(f"[JTS-0223] split={split} supervised_edges(col-col)={len(edges)}")
        return edges, labels, types

    def get_cached_edges(self, split: str) -> List[Tuple[int, int]]:
        if split == "train":
            return self._train_edges
        if split == "val":
            return self._val_edges
        if split == "test":
            return self._test_edges
        raise ValueError(f"Unknown split={split}. Expected one of: train,val,test")

    def mine_positive_edges(
        self,
        split: str,
        *,
        mode: str = "threshold",
        threshold: float = 0.5,
        top_ratio: float = 0.5,
    ) -> List[Tuple[int, int]]:
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown split={split}. Expected one of: train,val,test")

        if loader is None:
            raise RuntimeError("Data loaders are not initialized. Call train() or create_dataloaders() first.")
        if mode not in {"threshold", "top_ratio", "elbow"}:
            raise ValueError(f"Unknown mode={mode}. Expected one of: threshold,top_ratio,elbow")

        self.model.eval()
        positives: List[Tuple[int, int]] = []
        with torch.no_grad():
            if mode == "elbow":
                probs_all: List[torch.Tensor] = []
                src_all: List[torch.Tensor] = []
                dst_all: List[torch.Tensor] = []
                for batch in tqdm(
                    loader,
                    desc=f"Mining predicted edges ({split})",
                    leave=False,
                    disable=TQDM_DISABLE,
                ):
                    batch = batch.to(self.device)
                    self._apply_prepool_overlay(batch)
                    self._attach_task_pair_features(batch)
                    outputs = self.model(batch)
                    probs = torch.sigmoid(outputs["link_logits"]).view(-1).detach().cpu()
                    if probs.numel() == 0:
                        continue
                    src_global = batch.n_id[batch.edge_label_index[0]].detach().cpu()
                    dst_global = batch.n_id[batch.edge_label_index[1]].detach().cpu()
                    probs_all.append(probs)
                    src_all.append(src_global)
                    dst_all.append(dst_global)

                if not probs_all:
                    return []

                probs = torch.cat(probs_all, dim=0)
                src = torch.cat(src_all, dim=0)
                dst = torch.cat(dst_all, dim=0)
                elbow_threshold = _elbow_threshold_from_scores(probs)
                if not math.isfinite(elbow_threshold):
                    logger.warning(f"[Reciprocity] split={split} elbow_threshold invalid; skip.")
                    return []
                mask = probs >= float(elbow_threshold)
                if not bool(mask.any()):
                    return []
                selected_src = src[mask].tolist()
                selected_dst = dst[mask].tolist()
                positives = list(zip(selected_src, selected_dst))
                keep_ratio = float(mask.sum().item()) / float(mask.numel())
                logger.info(
                    f"[Reciprocity] split={split} elbow_threshold={elbow_threshold:.6f} "
                    f"candidates={int(mask.numel())} positives={int(mask.sum().item())} keep_ratio={keep_ratio:.4f}"
                )
                return positives

            for batch in tqdm(
                loader,
                desc=f"Mining predicted edges ({split})",
                leave=False,
                disable=TQDM_DISABLE,
            ):
                batch = batch.to(self.device)
                self._apply_prepool_overlay(batch)
                self._attach_task_pair_features(batch)
                outputs = self.model(batch)
                probs = torch.sigmoid(outputs["link_logits"]).view(-1)

                if probs.numel() == 0:
                    continue

                if mode == "threshold":
                    mask = probs >= threshold
                    if not mask.any():
                        continue
                    selected_src = batch.edge_label_index[0][mask]
                    selected_dst = batch.edge_label_index[1][mask]
                else:
                    if top_ratio <= 0:
                        continue
                    if top_ratio >= 1:
                        selected_src = batch.edge_label_index[0]
                        selected_dst = batch.edge_label_index[1]
                    else:
                        k = int(math.ceil(probs.numel() * float(top_ratio)))
                        k = max(1, min(k, probs.numel()))
                        topk_idx = torch.topk(probs, k, largest=True).indices
                        selected_src = batch.edge_label_index[0][topk_idx]
                        selected_dst = batch.edge_label_index[1][topk_idx]

                src_global = batch.n_id[selected_src].detach().cpu().tolist()
                dst_global = batch.n_id[selected_dst].detach().cpu().tolist()
                positives.extend(list(zip(src_global, dst_global)))

        return positives

    def mine_top_ratio_edges_from_edge_label_index(
        self,
        edge_label_index: torch.Tensor,
        *,
        keep_ratio: float = 0.5,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        return self.mine_edges_from_edge_label_index(
            edge_label_index,
            mode="top_ratio",
            keep_ratio=keep_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def mine_edges_from_edge_label_index(
        self,
        edge_label_index: torch.Tensor,
        *,
        mode: str = "top_ratio",
        threshold: float = 0.5,
        keep_ratio: float = 0.5,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
        if mode not in {"threshold", "top_ratio", "elbow"}:
            raise ValueError(f"Unknown mode={mode}. Expected one of: threshold,top_ratio,elbow")
        if mode == "top_ratio" and keep_ratio <= 0:
            return []

        if not isinstance(edge_label_index, torch.Tensor):
            raise TypeError("edge_label_index must be a torch.Tensor")
        if edge_label_index.dim() != 2 or edge_label_index.size(0) != 2:
            raise ValueError("edge_label_index must have shape [2, num_edges]")
        edge_label_index = edge_label_index.to(dtype=torch.long).contiguous()
        if edge_label_index.numel() == 0:
            return []

        edge_label = torch.zeros((edge_label_index.size(1),), dtype=torch.float)
        loader = LinkNeighborLoader(
            self.graph_data,
            num_neighbors=self.num_neighbors,
            batch_size=int(batch_size),
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            shuffle=False,
            num_workers=int(num_workers),
            neg_sampling_ratio=0.0,
        )

        self.model.eval()
        probs_all: List[torch.Tensor] = []
        src_all: List[torch.Tensor] = []
        dst_all: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Heavy mining (candidates)", leave=False, disable=TQDM_DISABLE):
                batch = batch.to(self.device)
                self._apply_prepool_overlay(batch)
                self._attach_task_pair_features(batch)
                outputs = self.model(batch)
                probs = torch.sigmoid(outputs["link_logits"]).view(-1).detach().cpu()
                if probs.numel() == 0:
                    continue
                src_global = batch.n_id[batch.edge_label_index[0]].detach().cpu()
                dst_global = batch.n_id[batch.edge_label_index[1]].detach().cpu()
                probs_all.append(probs)
                src_all.append(src_global)
                dst_all.append(dst_global)

        if not probs_all:
            return []

        probs = torch.cat(probs_all, dim=0)
        src = torch.cat(src_all, dim=0)
        dst = torch.cat(dst_all, dim=0)

        if mode == "elbow":
            elbow_threshold = _elbow_threshold_from_scores(probs)
            if not math.isfinite(elbow_threshold):
                logger.warning("[Heavy] elbow_threshold invalid; skip.")
                return []
            mask = probs >= float(elbow_threshold)
            if not bool(mask.any()):
                return []
            keep_ratio_effective = float(mask.sum().item()) / float(mask.numel())
            logger.info(
                f"[Heavy] elbow_threshold={elbow_threshold:.6f} candidates={int(mask.numel())} "
                f"selected={int(mask.sum().item())} keep_ratio={keep_ratio_effective:.4f}"
            )
            selected_src = src[mask].tolist()
            selected_dst = dst[mask].tolist()
            return list(zip(selected_src, selected_dst))

        if mode == "threshold":
            mask = probs >= float(threshold)
            if not bool(mask.any()):
                return []
            selected_src = src[mask].tolist()
            selected_dst = dst[mask].tolist()
            return list(zip(selected_src, selected_dst))

        keep_ratio = float(keep_ratio)
        if keep_ratio >= 1.0:
            return list(zip(src.tolist(), dst.tolist()))

        k = int(math.ceil(probs.numel() * keep_ratio))
        k = max(1, min(k, probs.numel()))
        topk_idx = torch.topk(probs, k, largest=True).indices

        selected_src = src[topk_idx].tolist()
        selected_dst = dst[topk_idx].tolist()
        return list(zip(selected_src, selected_dst))

    def _apply_prepool_overlay(self, batch) -> None:
        if not self.prepool_cells or self._prepool_overlay is None:
            return
        n_id = getattr(batch, "n_id", None)
        if n_id is None:
            return
        n_id_cpu = n_id.detach().cpu()
        row_map = self._prepool_overlay["row_map"]
        col_map = self._prepool_overlay["col_map"]
        row_emb = self._prepool_overlay["row_emb"]
        col_emb = self._prepool_overlay["col_emb"]

        if row_map is not None and row_emb is not None:
            row_idx = row_map[n_id_cpu]
            row_mask = row_idx >= 0
            if bool(row_mask.any()):
                batch.x[row_mask] = row_emb[row_idx[row_mask]].to(
                    device=batch.x.device,
                    dtype=batch.x.dtype,
                )

        if col_map is not None and col_emb is not None:
            col_idx = col_map[n_id_cpu]
            col_mask = col_idx >= 0
            if bool(col_mask.any()):
                batch.x[col_mask] = col_emb[col_idx[col_mask]].to(
                    device=batch.x.device,
                    dtype=batch.x.dtype,
                )

    def _attach_task_pair_features(self, batch) -> None:
        pair_feature_store = self.em_pair_feature_store
        if pair_feature_store is None:
            pair_feature_store = self.jts_pair_feature_store
        if pair_feature_store is None:
            pair_feature_store = self.sm_pair_feature_store
        if pair_feature_store is None:
            pair_feature_store = self.uts_pair_feature_store
        if pair_feature_store is None:
            return
        src_local = batch.edge_label_index[0]
        dst_local = batch.edge_label_index[1]
        src_global = batch.n_id[src_local]
        dst_global = batch.n_id[dst_local]
        pair_feats = pair_feature_store.build_batch_features(
            src_global,
            dst_global,
            device=batch.x.device,
            dtype=batch.x.dtype,
        )
        if self.target_task == "entity_matching":
            pair_feats = self._normalize_em_pair_features(pair_feats)
        pair_feats_source = pair_feats
        pair_feats_base = pair_feats_source
        if self._decoder_pair_keep_indices is not None:
            if self._decoder_pair_keep_indices:
                keep_idx = torch.as_tensor(
                    self._decoder_pair_keep_indices,
                    dtype=torch.long,
                    device=pair_feats_source.device,
                )
                pair_feats_base = pair_feats_source.index_select(1, keep_idx)
            else:
                pair_feats_base = torch.zeros(
                    (int(pair_feats_source.shape[0]), 0),
                    dtype=pair_feats_source.dtype,
                    device=pair_feats_source.device,
                )
        if self._online_symbolic_enabled:
            sym_feats = self._compute_online_symbolic_from_pair(
                pair_feats_source,
                apply_norm=True,
                apply_tile=True,
            )
            if sym_feats.numel() > 0:
                pair_feats = torch.cat([pair_feats_base, sym_feats], dim=1)
            else:
                pair_feats = pair_feats_base
        else:
            pair_feats = pair_feats_base
        batch.edge_pair_features = pair_feats

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False, disable=TQDM_DISABLE)):
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            self._apply_prepool_overlay(batch)
            self._attach_task_pair_features(batch)

            if batch_idx == 0:
                record_profile_event(
                    "train_first_batch",
                    {
                        "target_task": self.target_task,
                        "batch_num_nodes": int(batch.x.size(0)),
                        "batch_num_edges": int(batch.edge_index.size(1)),
                        "edge_label_count": int(batch.edge_label.numel()),
                        "batch_x_bytes": tensor_nbytes(batch.x),
                        "batch_edge_index_bytes": tensor_nbytes(batch.edge_index),
                        "batch_edge_attr_bytes": tensor_nbytes(getattr(batch, "edge_attr", None)),
                        "batch_pair_feature_bytes": tensor_nbytes(getattr(batch, "edge_pair_features", None)),
                    },
                )

            outputs = self.model(batch)

            # Use ground truth labels from the loader
            ground_truth = batch.edge_label
            loss = self._compute_train_loss(outputs, ground_truth)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(
        self,
        data_loader,
        ratio=0.5,
        block_size=6,
        threshold: float = 0.5,
        return_raw: bool = False,
        return_delta_raw: bool = False,
        return_replay_inputs: bool = False,
    ):
        """
        Evaluate model performance.

        Args:
            data_loader: Data loader.
            ratio: (legacy, unused in 0210) joinable_table_search block voting threshold
            block_size: (legacy, unused in 0210) joinable_table_search block size
            threshold: Decision threshold used for F1 and related metrics.
            return_raw: If True, also return raw scores and labels.

        Returns:
            If return_raw=False: metrics dict.
            If return_raw=True: (metrics dict, raw scores array, raw labels array).
        """
        self.model.eval()
        all_pred_chunks, all_label_chunks = [], []
        all_pred_raw_chunks = []
        all_delta_raw_chunks = []
        all_pair_features = []
        all_decoder_input_chunks = []
        all_edge_hidden_chunks = []
        all_src_degree_chunks = []
        all_dst_degree_chunks = []
        all_edge_src_chunks = []
        all_edge_dst_chunks = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False, disable=TQDM_DISABLE):
                batch = batch.to(self.device)
                self._apply_prepool_overlay(batch)
                self._attach_task_pair_features(batch)
                outputs = self.model(batch, return_replay_tensors=bool(return_replay_inputs))
                preds = torch.sigmoid(outputs["link_logits"]).view(-1)
                preds_raw = torch.sigmoid(outputs.get("link_logits_raw", outputs["link_logits"])).view(-1)
                labels = batch.edge_label.view(-1)
                src_global, dst_global = self._edge_globals_from_batch(batch)
                src_degree, dst_degree = self._edge_degrees_from_globals(
                    src_global=src_global,
                    dst_global=dst_global,
                    device=preds.device,
                    dtype=preds.dtype,
                )
                if return_delta_raw and self.reranker is not None and self.reranker.enabled:
                    delta_raw = self._compute_rerank_delta(batch, preds)
                    all_delta_raw_chunks.append(delta_raw.detach().cpu().numpy())
                if return_replay_inputs:
                    pair_features = getattr(batch, "edge_pair_features", None)
                    if pair_features is None:
                        pair_features = torch.zeros((preds.numel(), 0), dtype=preds.dtype, device=preds.device)
                    decoder_input = outputs.get("decoder_input", None)
                    if decoder_input is None:
                        decoder_input = torch.zeros((preds.numel(), 0), dtype=preds.dtype, device=preds.device)
                    edge_hidden = outputs.get("edge_hidden", None)
                    if edge_hidden is None:
                        edge_hidden = torch.zeros((preds.numel(), 0), dtype=preds.dtype, device=preds.device)
                    all_pair_features.append(pair_features.detach().cpu().numpy())
                    all_decoder_input_chunks.append(decoder_input.detach().cpu().numpy())
                    all_edge_hidden_chunks.append(edge_hidden.detach().cpu().numpy())
                    all_src_degree_chunks.append(src_degree.detach().cpu().numpy())
                    all_dst_degree_chunks.append(dst_degree.detach().cpu().numpy())
                    all_edge_src_chunks.append(src_global.detach().cpu().numpy())
                    all_edge_dst_chunks.append(dst_global.detach().cpu().numpy())
                all_pred_chunks.append(preds.detach().cpu().numpy())
                all_pred_raw_chunks.append(preds_raw.detach().cpu().numpy())
                all_label_chunks.append(labels.detach().cpu().numpy())

        if not all_label_chunks:
            if return_raw:
                if return_delta_raw:
                    if return_replay_inputs:
                        return {}, np.array([]), np.array([]), np.array([]), {}
                    return {}, np.array([]), np.array([]), np.array([])
                if return_replay_inputs:
                    return {}, np.array([]), np.array([]), {}
                return {}, np.array([]), np.array([])
            return {}

        all_preds = np.concatenate(all_pred_chunks, axis=0)
        all_preds_raw = np.concatenate(all_pred_raw_chunks, axis=0) if all_pred_raw_chunks else np.asarray(all_preds)
        all_labels = np.concatenate(all_label_chunks, axis=0)

        metrics = _binary_classification_metrics(all_labels, all_preds, threshold=threshold)

        replay_inputs = None
        if return_replay_inputs:
            pair_feature_dim = len(self._current_pair_feature_order())
            if all_pair_features:
                pair_features_arr = np.concatenate(all_pair_features, axis=0).astype(np.float32, copy=False)
            else:
                pair_features_arr = np.zeros((all_preds.shape[0], pair_feature_dim), dtype=np.float32)
            if all_decoder_input_chunks:
                decoder_input_arr = np.concatenate(all_decoder_input_chunks, axis=0).astype(np.float32, copy=False)
            else:
                decoder_input_arr = np.zeros((all_preds.shape[0], 0), dtype=np.float32)
            if all_edge_hidden_chunks:
                edge_hidden_arr = np.concatenate(all_edge_hidden_chunks, axis=0).astype(np.float32, copy=False)
            else:
                edge_hidden_arr = np.zeros((all_preds.shape[0], 0), dtype=np.float32)
            replay_inputs = {
                "pair_features": pair_features_arr,
                "decoder_input": decoder_input_arr,
                "edge_hidden": edge_hidden_arr,
                "src_degree": np.concatenate(all_src_degree_chunks, axis=0).astype(np.float32, copy=False),
                "dst_degree": np.concatenate(all_dst_degree_chunks, axis=0).astype(np.float32, copy=False),
                "edge_src": np.concatenate(all_edge_src_chunks, axis=0).astype(np.int64, copy=False),
                "edge_dst": np.concatenate(all_edge_dst_chunks, axis=0).astype(np.int64, copy=False),
                "scores_final": np.asarray(all_preds, dtype=np.float32),
                "scores_gnn_only": np.asarray(all_preds_raw, dtype=np.float32),
            }

        if return_raw:
            if return_delta_raw:
                delta_raw_arr = (
                    np.concatenate(all_delta_raw_chunks, axis=0).astype(np.float32, copy=False)
                    if all_delta_raw_chunks
                    else np.array([], dtype=np.float32)
                )
                if return_replay_inputs:
                    return metrics, all_preds, all_labels, delta_raw_arr, replay_inputs
                return metrics, all_preds, all_labels, delta_raw_arr
            if return_replay_inputs:
                return metrics, all_preds, all_labels, replay_inputs
            return metrics, all_preds, all_labels
        return metrics

    def train(self, num_epochs: int = 100, early_stopping_patience: int = 10):
        train_start_time = time.time()
        with profile_phase("trainer.create_dataloaders", {"target_task": self.target_task}):
            self.create_dataloaders(num_workers=self.num_workers)

        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(num_epochs):
            epoch_start = time.time()
            with profile_phase(
                "trainer.epoch.train",
                {"target_task": self.target_task, "epoch": int(epoch + 1)},
            ):
                train_loss = self.train_epoch()
            train_duration = time.time() - epoch_start

            val_start = time.time()
            with profile_phase(
                "trainer.epoch.val",
                {"target_task": self.target_task, "epoch": int(epoch + 1)},
            ):
                if self.target_task == "entity_matching":
                    val_metrics_default, val_scores, val_labels = self.evaluate(
                        self.val_loader,
                        threshold=0.5,
                        return_raw=True,
                    )
                else:
                    val_metrics_default = self.evaluate(self.val_loader)
                    val_scores = np.array([])
                    val_labels = np.array([])
            val_duration = time.time() - val_start

            val_f1 = float(val_metrics_default.get("link_f1", 0.0))
            val_best_threshold = 0.5
            val_best_f1 = val_f1
            if self.target_task == "entity_matching" and len(val_scores) > 0:
                val_best_threshold, val_best_f1 = find_best_threshold(val_labels, val_scores)
                logger.info(
                    "Epoch %d/%d | Train Loss: %.4f | Val F1@0.5: %.4f | Val BestThr: %.3f (F1=%.4f)",
                    int(epoch + 1),
                    int(num_epochs),
                    float(train_loss),
                    float(val_f1),
                    float(val_best_threshold),
                    float(val_best_f1),
                )
            else:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.4f | Val F1: %.4f",
                    int(epoch + 1),
                    int(num_epochs),
                    float(train_loss),
                    float(val_f1),
                )
            record_profile_event(
                "epoch_summary",
                {
                    "target_task": self.target_task,
                    "epoch": int(epoch + 1),
                    "train_loss": float(train_loss),
                    "val_f1": float(val_f1),
                    "val_f1_default": float(val_f1),
                    "val_best_threshold": float(val_best_threshold),
                    "val_best_f1": float(val_best_f1),
                    "train_duration_sec": float(train_duration),
                    "val_duration_sec": float(val_duration),
                    "epoch_duration_sec": float(time.time() - epoch_start),
                    "optimizer_state_bytes": int(self._optimizer_state_bytes()),
                },
            )
            if self.target_task == "entity_matching" and epoch == 0:
                start_to_epoch1 = float(time.time() - train_start_time)
                logger.info("[EM-Diag] start_to_epoch1_sec=%.3f", start_to_epoch1)
                record_profile_event(
                    "em_start_to_epoch1",
                    {
                        "target_task": self.target_task,
                        "start_to_epoch1_sec": start_to_epoch1,
                        "feature_build_sec": float(self._em_feature_build_sec),
                    },
                )

            if (
                self.target_task == "entity_matching"
                and self.em_hard_neg_ratio > 0.0
                and (epoch + 1) >= self.em_hard_neg_warmup_epochs
                and (epoch + 1) < num_epochs
            ):
                with profile_phase(
                    "trainer.epoch.hard_negative_update",
                    {"target_task": self.target_task, "epoch": int(epoch + 1)},
                ):
                    selected_hard_neg = self._select_hard_negative_edges()
                    self._rebuild_train_loader_with_extra_negatives(selected_hard_neg)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save the best model state
                best_model_state = {"model": copy.deepcopy(self.model.state_dict())}
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("\nFinal evaluation on test set:")
        if best_model_state:
            logger.info("Loading best model for final evaluation...")
            if isinstance(best_model_state, dict) and "model" in best_model_state:
                self.model.load_state_dict(best_model_state["model"])
            else:
                self.model.load_state_dict(best_model_state)

        rerank_collect = bool(self.reranker is not None and self.reranker.enabled)
        rerank_cfg: Dict[str, object] = {
            "enabled": False,
            "alpha": 0.0,
            "delta_cap": 0.0,
            "gate_margin": 1.0,
            "gate_delta_min": 0.0,
            "gate_ratio": 1.0,
            "reference_threshold": float(self.eval_fixed_threshold),
            "val_f1": 0.0,
            "val_threshold": float(self.eval_fixed_threshold),
            "task_residual_enabled": False,
            "task_residual_scale": 0.0,
            "task_residual_margin": 0.0,
        }

        # Step 1: collect replay tensors; if rerank is enabled, fit/search rerank on validation.
        if rerank_collect:
            logger.info(
                "Collecting train replay for benefit selector and searching rerank config on validation set (threshold_mode=%s)...",
                self.eval_threshold_mode,
            )
        else:
            logger.info(
                "Collecting train/val replay tensors for offline policy sweep (threshold_mode=%s)...",
                self.eval_threshold_mode,
            )
        train_scores = np.array([], dtype=np.float32)
        train_labels = np.array([], dtype=np.float32)
        train_delta_scores = np.array([], dtype=np.float32)
        train_replay_inputs: Dict[str, np.ndarray] = {}
        with profile_phase("trainer.final_train_selector", {"target_task": self.target_task}):
            train_out = self.evaluate(
                self.train_loader,
                return_raw=True,
                return_replay_inputs=True,
            )
        _, train_scores, train_labels, train_replay_inputs = train_out
        with profile_phase("trainer.final_val_search", {"target_task": self.target_task}):
            val_out = self.evaluate(
                self.val_loader,
                return_raw=True,
                return_replay_inputs=True,
            )

        val_metrics, val_scores, val_labels, val_replay_inputs = val_out
        train_scores_gnn_only = np.asarray(
            train_replay_inputs.get("scores_gnn_only", train_scores),
            dtype=np.float32,
        ).reshape(-1)
        val_scores_gnn_only = np.asarray(
            val_replay_inputs.get("scores_gnn_only", val_scores),
            dtype=np.float32,
        ).reshape(-1)
        val_delta_scores = np.array([], dtype=np.float32)
        val_gate_mask = np.ones_like(val_scores, dtype=np.int8)
        val_active_mask = np.ones_like(val_scores, dtype=np.int8)
        val_benefit_probs = np.ones_like(val_scores, dtype=np.float32)
        train_task_residual_scores = np.zeros_like(train_scores, dtype=np.float32)
        val_task_residual_scores = np.zeros_like(val_scores, dtype=np.float32)
        val_task_residual_mask = np.zeros_like(val_scores, dtype=np.int8)
        task_residual_summary: Dict[str, object] = {"enabled": False}
        benefit_summary: Dict[str, object] = {"enabled": False}

        if len(val_scores) > 0:
            reference_threshold = self._select_reference_threshold(val_labels, val_scores)
            if rerank_collect and len(train_scores) > 0:
                train_delta_scores = self._compute_rerank_delta_from_replay_inputs(
                    base_scores=train_scores,
                    replay_inputs=train_replay_inputs,
                    reference_threshold=float(reference_threshold),
                )
            if rerank_collect:
                val_delta_scores = self._compute_rerank_delta_from_replay_inputs(
                    base_scores=val_scores,
                    replay_inputs=val_replay_inputs,
                    reference_threshold=float(reference_threshold),
                )
            if rerank_collect:
                if len(val_delta_scores) == len(val_scores):
                    if len(train_delta_scores) == len(train_scores) and len(train_scores) > 0:
                        selector, _, benefit_summary = self._fit_benefit_selector(
                            base_scores=train_scores,
                            delta_scores=train_delta_scores,
                            labels=train_labels,
                            replay_inputs=train_replay_inputs,
                            reference_threshold=float(reference_threshold),
                        )
                        if selector is not None:
                            val_benefit_probs = self._predict_benefit_probs(
                                selector=selector,
                                base_scores=val_scores,
                                delta_scores=val_delta_scores,
                                replay_inputs=val_replay_inputs,
                                reference_threshold=float(reference_threshold),
                            )
                            benefit_summary = {
                                **benefit_summary,
                                "train_samples": int(len(train_scores)),
                                "source_split": "train",
                                "val_prob_mean": float(np.mean(val_benefit_probs)) if len(val_benefit_probs) > 0 else 0.0,
                                "val_prob_std": float(np.std(val_benefit_probs)) if len(val_benefit_probs) > 0 else 0.0,
                            }
                        else:
                            val_benefit_probs = np.ones_like(val_scores, dtype=np.float32)
                    else:
                        logger.warning("[RerankBenefit] train replay unavailable; selector disabled for this run.")
                        selector = None
                    self.benefit_selector = selector
                    rerank_cfg = self._search_best_rerank_config(
                        base_scores=val_scores,
                        delta_scores=val_delta_scores,
                        labels=val_labels,
                        reference_threshold=float(reference_threshold),
                        benefit_probs=val_benefit_probs,
                    )
                    val_scores_final = np.asarray(rerank_cfg["scores"], dtype=np.float32)
                    logger.info(
                        "[Rerank] val selected alpha=%.3f delta_cap=%.3f val_f1=%.4f val_threshold=%.3f",
                        float(rerank_cfg["alpha"]),
                        float(rerank_cfg["delta_cap"]),
                        float(rerank_cfg["val_f1"]),
                        float(rerank_cfg["val_threshold"]),
                    )
                    logger.info(
                        "[RerankGate] val selected gate_margin=%.3f gate_delta_min=%.3f gate_ratio=%.4f benefit_min_score=%.3f benefit_ratio=%.4f active_ratio=%.4f cap_ok=%s ref_threshold=%.3f",
                        float(rerank_cfg["gate_margin"]),
                        float(rerank_cfg["gate_delta_min"]),
                        float(rerank_cfg["gate_ratio"]),
                        float(rerank_cfg.get("benefit_min_score", 0.0)),
                        float(rerank_cfg.get("benefit_ratio", 1.0)),
                        float(rerank_cfg.get("active_ratio", 1.0)),
                        str(bool(rerank_cfg.get("active_ratio_cap_satisfied", True))).lower(),
                        float(rerank_cfg["reference_threshold"]),
                    )
                    if bool(benefit_summary.get("enabled", False)):
                        logger.info(
                            "[RerankBenefit] fitted on %s feature_dim=%d pos=%d neg=%d prob_mean=%.4f prob_std=%.4f val_prob_mean=%.4f val_prob_std=%.4f",
                            str(benefit_summary.get("source_split", "train")),
                            int(benefit_summary.get("feature_dim", 0)),
                            int(benefit_summary.get("pos_count", 0)),
                            int(benefit_summary.get("neg_count", 0)),
                            float(benefit_summary.get("prob_mean", 0.0)),
                            float(benefit_summary.get("prob_std", 0.0)),
                            float(benefit_summary.get("val_prob_mean", 0.0)),
                            float(benefit_summary.get("val_prob_std", 0.0)),
                        )
                    val_gate_mask = self._compute_gate_mask_np(
                        base_scores=val_scores,
                        delta_scores=val_delta_scores,
                        gate_margin=float(rerank_cfg["gate_margin"]),
                        gate_delta_min=float(rerank_cfg["gate_delta_min"]),
                        reference_threshold=float(rerank_cfg["reference_threshold"]),
                    ).astype(np.int8, copy=False)
                    _, _, val_active_mask_bool = self._apply_rerank_np(
                        val_scores,
                        val_delta_scores,
                        alpha=float(rerank_cfg["alpha"]),
                        delta_cap=float(rerank_cfg["delta_cap"]),
                        gate_margin=float(rerank_cfg["gate_margin"]),
                        gate_delta_min=float(rerank_cfg["gate_delta_min"]),
                        reference_threshold=float(rerank_cfg["reference_threshold"]),
                        benefit_probs=val_benefit_probs,
                        benefit_min_score=float(rerank_cfg.get("benefit_min_score", 0.0)),
                    )
                    val_active_mask = np.asarray(val_active_mask_bool, dtype=np.int8)
                    train_active_mask_bool = np.ones_like(train_scores, dtype=bool)
                    train_scores_rule = train_scores
                    if len(train_delta_scores) == len(train_scores) and len(train_scores) > 0:
                        train_benefit_probs = np.ones_like(train_scores, dtype=np.float32)
                        if self.benefit_selector is not None:
                            train_benefit_probs = self._predict_benefit_probs(
                                selector=self.benefit_selector,
                                base_scores=train_scores,
                                delta_scores=train_delta_scores,
                                replay_inputs=train_replay_inputs,
                                reference_threshold=float(rerank_cfg["reference_threshold"]),
                            )
                        train_scores_rule, _, train_active_mask_bool = self._apply_rerank_np(
                            train_scores,
                            train_delta_scores,
                            alpha=float(rerank_cfg["alpha"]),
                            delta_cap=float(rerank_cfg["delta_cap"]),
                            gate_margin=float(rerank_cfg["gate_margin"]),
                            gate_delta_min=float(rerank_cfg["gate_delta_min"]),
                            reference_threshold=float(rerank_cfg["reference_threshold"]),
                            benefit_probs=train_benefit_probs,
                            benefit_min_score=float(rerank_cfg.get("benefit_min_score", 0.0)),
                        )
                    if self._task_residual_enabled() and len(train_scores_rule) > 0:
                        self.task_residual_head, train_task_residual_scores, task_residual_summary = self._fit_task_residual_head(
                            base_scores=train_scores_rule,
                            labels=train_labels,
                            replay_inputs=train_replay_inputs,
                            reference_threshold=float(rerank_cfg["reference_threshold"]),
                            active_mask=np.asarray(train_active_mask_bool, dtype=np.int8),
                        )
                        if self.task_residual_head is not None:
                            val_task_residual_scores = self._predict_task_residual_scores(
                                head=self.task_residual_head,
                                base_scores=val_scores_final,
                                replay_inputs=val_replay_inputs,
                                reference_threshold=float(rerank_cfg["reference_threshold"]),
                            )
                            residual_cfg = self._search_best_task_residual_config(
                                base_scores=val_scores_final,
                                residual_scores=val_task_residual_scores,
                                labels=val_labels,
                                reference_threshold=float(rerank_cfg["reference_threshold"]),
                                active_mask=val_active_mask_bool,
                            )
                            if residual_cfg.get("enabled", False):
                                val_scores_final = np.asarray(residual_cfg["scores"], dtype=np.float32)
                                val_task_residual_mask = np.asarray(residual_cfg["mask"], dtype=np.int8)
                                rerank_cfg["val_threshold"] = float(residual_cfg["val_threshold"])
                                rerank_cfg["val_f1"] = float(residual_cfg["val_f1"])
                            rerank_cfg["task_residual_enabled"] = bool(residual_cfg.get("enabled", False))
                            rerank_cfg["task_residual_scale"] = float(residual_cfg.get("residual_scale", 0.0))
                            rerank_cfg["task_residual_margin"] = float(residual_cfg.get("residual_margin", 0.0))
                            rerank_cfg["task_residual_fix_gap"] = int(residual_cfg.get("fix_gap", 0))
                            rerank_cfg["task_residual_breaks"] = int(residual_cfg.get("breaks", 0))
                            task_residual_summary = {**task_residual_summary, **residual_cfg}
                else:
                    val_scores_final = val_scores
                    logger.warning("Reranking requested but validation deltas are unavailable; using raw validation scores.")
            else:
                val_scores_final = val_scores

            if rerank_collect and len(val_delta_scores) == len(val_scores):
                best_threshold = float(rerank_cfg["val_threshold"])
                best_val_f1 = float(rerank_cfg["val_f1"])
            else:
                if self.eval_threshold_mode == "val_best":
                    best_threshold, best_val_f1 = find_best_threshold(val_labels, val_scores_final)
                else:
                    best_threshold = float(self.eval_fixed_threshold)
                    best_val_f1 = _binary_classification_metrics(
                        val_labels, val_scores_final, threshold=best_threshold
                    ).get("link_f1", 0.0)

            val_metrics_raw = _binary_classification_metrics(val_labels, val_scores, threshold=best_threshold)
            val_metrics_final = _binary_classification_metrics(val_labels, val_scores_final, threshold=best_threshold)
            val_metrics_raw_05 = _binary_classification_metrics(val_labels, val_scores, threshold=0.5)
            val_metrics_final_05 = _binary_classification_metrics(val_labels, val_scores_final, threshold=0.5)
            if rerank_collect:
                logger.info(
                    "[Rerank] val threshold=%.3f raw_f1=%.4f reranked_f1=%.4f",
                    float(best_threshold),
                    float(val_metrics_raw.get("link_f1", 0.0)),
                    float(val_metrics_final.get("link_f1", 0.0)),
                )
                logger.info(
                    "[Rerank] val @0.5 raw_f1=%.4f reranked_f1=%.4f",
                    float(val_metrics_raw_05.get("link_f1", 0.0)),
                    float(val_metrics_final_05.get("link_f1", 0.0)),
                )
                logger.info(
                    "[Rerank] selected validation objective f1=%.4f (mode=%s)",
                    float(best_val_f1),
                    self.eval_threshold_mode,
                )
            else:
                logger.info(
                    "[Validation] selected threshold=%.3f f1=%.4f (mode=%s)",
                    float(best_threshold),
                    float(best_val_f1),
                    self.eval_threshold_mode,
                )
                logger.info(
                    "[Validation] @0.5 f1=%.4f",
                    float(val_metrics_final_05.get("link_f1", 0.0)),
                )
        else:
            val_scores_final = val_scores
            best_threshold = float(self.eval_fixed_threshold)
            logger.warning("No validation data available.")

        if not rerank_collect:
            rerank_cfg["reference_threshold"] = float(best_threshold)

        # Step 2: evaluate on test set using validation-selected (or fixed) threshold.
        logger.info("\nEvaluating on test set with threshold %.3f (mode=%s)...", best_threshold, self.eval_threshold_mode)
        with profile_phase("trainer.final_test_eval", {"target_task": self.target_task}):
            test_out = self.evaluate(
                self.test_loader,
                threshold=best_threshold,
                return_raw=True,
                return_replay_inputs=True,
            )

        test_metrics_raw, test_scores, test_labels, test_replay_inputs = test_out
        test_scores_gnn_only = np.asarray(
            test_replay_inputs.get("scores_gnn_only", test_scores),
            dtype=np.float32,
        ).reshape(-1)
        test_delta_scores = np.array([], dtype=np.float32)
        test_benefit_probs = np.ones_like(test_scores, dtype=np.float32)
        test_task_residual_scores = np.zeros_like(test_scores, dtype=np.float32)
        test_task_residual_mask_bool = np.zeros_like(test_scores, dtype=bool)

        raw_test_metrics_selected = {"link_f1": 0.0}
        raw_test_metrics_reference = {"link_f1": 0.0}
        raw_test_metrics_default_05 = {"link_f1": 0.0}
        if len(test_scores) > 0:
            if rerank_collect:
                test_delta_scores = self._compute_rerank_delta_from_replay_inputs(
                    base_scores=test_scores,
                    replay_inputs=test_replay_inputs,
                    reference_threshold=float(rerank_cfg["reference_threshold"]),
                )
            if rerank_collect and len(test_delta_scores) == len(test_scores):
                test_benefit_probs = self._predict_benefit_probs(
                    selector=self.benefit_selector,
                    base_scores=test_scores,
                    delta_scores=test_delta_scores,
                    replay_inputs=test_replay_inputs,
                    reference_threshold=float(rerank_cfg["reference_threshold"]),
                )
                test_scores_final, _, test_active_mask_bool = self._apply_rerank_np(
                    test_scores,
                    test_delta_scores,
                    alpha=float(rerank_cfg["alpha"]),
                    delta_cap=float(rerank_cfg["delta_cap"]),
                    gate_margin=float(rerank_cfg["gate_margin"]),
                    gate_delta_min=float(rerank_cfg["gate_delta_min"]),
                    reference_threshold=float(rerank_cfg["reference_threshold"]),
                    benefit_probs=test_benefit_probs,
                    benefit_min_score=float(rerank_cfg.get("benefit_min_score", 0.0)),
                )
                if bool(rerank_cfg.get("task_residual_enabled", False)) and self.task_residual_head is not None:
                    test_task_residual_scores = self._predict_task_residual_scores(
                        head=self.task_residual_head,
                        base_scores=test_scores_final,
                        replay_inputs=test_replay_inputs,
                        reference_threshold=float(rerank_cfg["reference_threshold"]),
                    )
                    test_scores_final, test_task_residual_mask_bool = self._apply_task_residual_np(
                        test_scores_final,
                        test_task_residual_scores,
                        reference_threshold=float(rerank_cfg["reference_threshold"]),
                        active_mask=test_active_mask_bool,
                        residual_scale=float(rerank_cfg.get("task_residual_scale", 0.0)),
                        residual_margin=float(rerank_cfg.get("task_residual_margin", 0.0)),
                    )
            else:
                test_scores_final = test_scores
                test_active_mask_bool = np.ones_like(test_scores, dtype=bool)

            test_metrics = _binary_classification_metrics(test_labels, test_scores_final, threshold=best_threshold)
            raw_test_metrics_selected = _binary_classification_metrics(test_labels, test_scores, threshold=best_threshold)
            raw_test_metrics_reference = _binary_classification_metrics(
                test_labels, test_scores, threshold=float(rerank_cfg["reference_threshold"])
            )
            raw_test_metrics_default_05 = _binary_classification_metrics(test_labels, test_scores, threshold=0.5)
            test_metrics_default_05 = _binary_classification_metrics(test_labels, test_scores_final, threshold=0.5)
            if rerank_collect:
                logger.info(f"Test Metrics (reranked, threshold={best_threshold:.3f}): {test_metrics}")
                logger.info(f"Test Metrics (raw, threshold={best_threshold:.3f}): {raw_test_metrics_selected}")
                logger.info(
                    "Test Metrics (raw, reference_threshold=%.3f): %s",
                    float(rerank_cfg["reference_threshold"]),
                    raw_test_metrics_reference,
                )
                logger.info(f"Test Metrics (reranked, threshold=0.5): {test_metrics_default_05}")
                logger.info(f"Test Metrics (raw, threshold=0.5): {raw_test_metrics_default_05}")
                logger.info(
                    f">>> Test F1 delta (threshold={best_threshold:.3f}) raw->reranked: "
                    f"{test_metrics['link_f1'] - raw_test_metrics_selected['link_f1']:.4f} "
                    f"({raw_test_metrics_selected['link_f1']:.4f} -> {test_metrics['link_f1']:.4f})"
                )
                logger.info(
                    ">>> Test F1 delta (rerank@eval_threshold - raw@reference_threshold): %.4f (%.4f - %.4f)",
                    float(test_metrics["link_f1"]) - float(raw_test_metrics_reference["link_f1"]),
                    float(test_metrics["link_f1"]),
                    float(raw_test_metrics_reference["link_f1"]),
                )
            else:
                logger.info(f"Test Metrics (threshold={best_threshold:.3f}): {test_metrics}")
                logger.info(f"Test Metrics (threshold=0.5): {test_metrics_default_05}")
        else:
            test_metrics = test_metrics_raw
            test_scores_final = test_scores
            logger.info(f"Test Metrics: {test_metrics}")

        test_metrics["best_threshold"] = float(best_threshold)
        test_metrics["eval_threshold_mode"] = self.eval_threshold_mode
        test_metrics["raw_test_f1_at_best_threshold"] = (
            raw_test_metrics_selected["link_f1"] if len(test_scores) > 0 else 0.0
        )
        test_metrics["raw_test_f1_reference_threshold"] = (
            raw_test_metrics_reference["link_f1"] if len(test_scores) > 0 else 0.0
        )
        test_metrics["test_delta_same_threshold"] = (
            float(test_metrics.get("link_f1", 0.0)) - float(test_metrics["raw_test_f1_at_best_threshold"])
        )
        test_metrics["test_delta_reference_threshold"] = (
            float(test_metrics.get("link_f1", 0.0)) - float(test_metrics["raw_test_f1_reference_threshold"])
        )
        test_metrics["raw_test_f1_at_0.5"] = raw_test_metrics_default_05["link_f1"] if len(test_scores) > 0 else 0.0
        test_metrics["rerank_enabled"] = bool(rerank_collect)
        test_metrics["rerank_alpha"] = float(rerank_cfg["alpha"])
        test_metrics["rerank_delta_cap"] = float(rerank_cfg["delta_cap"])
        test_metrics["rerank_gate_margin"] = float(rerank_cfg["gate_margin"])
        test_metrics["rerank_gate_delta_min"] = float(rerank_cfg["gate_delta_min"])
        test_metrics["rerank_gate_ratio"] = float(rerank_cfg["gate_ratio"])
        test_metrics["rerank_benefit_enabled"] = bool(self.benefit_selector is not None and self.rerank_benefit_enable)
        test_metrics["rerank_benefit_min_score"] = float(rerank_cfg.get("benefit_min_score", 0.0))
        test_metrics["rerank_benefit_ratio"] = float(rerank_cfg.get("benefit_ratio", 1.0))
        test_metrics["rerank_active_ratio"] = float(rerank_cfg.get("active_ratio", 1.0))
        test_metrics["rerank_reference_threshold"] = float(rerank_cfg["reference_threshold"])
        test_metrics["rerank_task_residual_enabled"] = bool(rerank_cfg.get("task_residual_enabled", False))
        test_metrics["rerank_task_residual_scale"] = float(rerank_cfg.get("task_residual_scale", 0.0))
        test_metrics["rerank_task_residual_margin"] = float(rerank_cfg.get("task_residual_margin", 0.0))

        scores_dir = "scores"
        os.makedirs(scores_dir, exist_ok=True)
        pair_feature_order = self._current_pair_feature_order()
        scores_path = os.path.join(scores_dir, f"{self.target_task}_scores.npz")
        if rerank_collect and len(test_delta_scores) == len(test_scores):
            test_gate_mask = self._compute_gate_mask_np(
                base_scores=test_scores,
                delta_scores=test_delta_scores,
                gate_margin=float(rerank_cfg["gate_margin"]),
                gate_delta_min=float(rerank_cfg["gate_delta_min"]),
                reference_threshold=float(rerank_cfg["reference_threshold"]),
            )
        else:
            test_gate_mask = np.ones_like(test_scores, dtype=np.int8)
            test_active_mask_bool = np.ones_like(test_scores, dtype=bool)
        scores_payload = {
            "dataset_name": np.asarray([self.dataset_name], dtype="<U64"),
            "run_tag": np.asarray([self.run_tag], dtype="<U128"),
            "trainer_seed": np.asarray([int(self.seed)], dtype=np.int64),
            "task": np.asarray([self.target_task], dtype="<U64"),
            "val_scores": val_scores,
            "val_scores_gnn_only": val_scores_gnn_only.astype(np.float32, copy=False),
            "val_scores_final": val_scores_final,
            "val_labels": val_labels,
            "val_delta_scores": val_delta_scores,
            "val_gate_mask": np.asarray(val_gate_mask, dtype=np.int8),
            "val_active_mask": np.asarray(val_active_mask, dtype=np.int8),
            "val_benefit_probs": np.asarray(val_benefit_probs, dtype=np.float32),
            "val_task_residual_scores": np.asarray(val_task_residual_scores, dtype=np.float32),
            "val_task_residual_mask": np.asarray(val_task_residual_mask, dtype=np.int8),
            "test_scores": test_scores,
            "test_scores_gnn_only": test_scores_gnn_only.astype(np.float32, copy=False),
            "test_scores_final": test_scores_final,
            "test_labels": test_labels,
            "test_delta_scores": test_delta_scores,
            "test_benefit_probs": np.asarray(test_benefit_probs, dtype=np.float32),
            "test_task_residual_scores": np.asarray(test_task_residual_scores, dtype=np.float32),
            "test_task_residual_mask": np.asarray(test_task_residual_mask_bool, dtype=np.int8),
            "rerank_alpha": np.asarray([float(rerank_cfg["alpha"])], dtype=np.float32),
            "rerank_delta_cap": np.asarray([float(rerank_cfg["delta_cap"])], dtype=np.float32),
            "rerank_gate_margin": np.asarray([float(rerank_cfg["gate_margin"])], dtype=np.float32),
            "rerank_gate_delta_min": np.asarray([float(rerank_cfg["gate_delta_min"])], dtype=np.float32),
            "rerank_gate_ratio": np.asarray([float(rerank_cfg["gate_ratio"])], dtype=np.float32),
            "rerank_benefit_min_score": np.asarray([float(rerank_cfg.get("benefit_min_score", 0.0))], dtype=np.float32),
            "rerank_benefit_ratio": np.asarray([float(rerank_cfg.get("benefit_ratio", 1.0))], dtype=np.float32),
            "rerank_active_ratio": np.asarray([float(rerank_cfg.get("active_ratio", 1.0))], dtype=np.float32),
            "rerank_reference_threshold": np.asarray([float(rerank_cfg["reference_threshold"])], dtype=np.float32),
            "rerank_task_residual_enabled": np.asarray(
                [1 if bool(rerank_cfg.get("task_residual_enabled", False)) else 0], dtype=np.int64
            ),
            "rerank_task_residual_scale": np.asarray(
                [float(rerank_cfg.get("task_residual_scale", 0.0))], dtype=np.float32
            ),
            "rerank_task_residual_margin": np.asarray(
                [float(rerank_cfg.get("task_residual_margin", 0.0))], dtype=np.float32
            ),
            "rerank_enabled": np.asarray([1 if bool(rerank_collect) else 0], dtype=np.int64),
            "rerank_benefit_enabled": np.asarray(
                [1 if bool(self.benefit_selector is not None and self.rerank_benefit_enable) else 0], dtype=np.int64
            ),
            "eval_threshold": np.asarray([float(best_threshold)], dtype=np.float32),
            "eval_threshold_mode": np.asarray([self.eval_threshold_mode], dtype="<U16"),
            "test_gate_mask": np.asarray(test_gate_mask, dtype=np.int8),
            "test_active_mask": np.asarray(test_active_mask_bool, dtype=np.int8),
        }
        np.savez(scores_path, **scores_payload)
        logger.info(f"Raw scores saved to {scores_path} (reload with np.load to re-search thresholds)")

        replay_path = os.path.join(scores_dir, f"{self.target_task}_replay_inputs.npz")
        train_scores_out = np.asarray(train_scores, dtype=np.float32)
        train_labels_out = np.asarray(train_labels, dtype=np.float32)
        train_replay_inputs_out = dict(train_replay_inputs)

        selection_replay_payload = {
            "dataset_name": np.asarray([self.dataset_name], dtype="<U64"),
            "run_tag": np.asarray([self.run_tag], dtype="<U128"),
            "trainer_seed": np.asarray([int(self.seed)], dtype=np.int64),
            "task": np.asarray([self.target_task], dtype="<U64"),
            "online_symbolic_enabled": np.asarray([int(self._online_symbolic_enabled)], dtype=np.int64),
            "online_symbolic_spec_id": np.asarray([self._online_symbolic_spec_id], dtype="<U128"),
            "online_symbolic_repr": np.asarray([self._online_symbolic_repr_effective], dtype="<U32"),
            "online_symbolic_raw_dim": np.asarray([int(self._online_symbolic_raw_dim)], dtype=np.int64),
            "online_symbolic_tile_repeat": np.asarray([int(self.online_symbolic_tile_repeat)], dtype=np.int64),
            "online_symbolic_effective_dim": np.asarray([int(self._online_symbolic_effective_dim)], dtype=np.int64),
            "online_symbolic_normalize": np.asarray([self.online_symbolic_normalize], dtype="<U32"),
            "pair_feature_order": np.asarray(pair_feature_order, dtype="<U64"),
            "train_scores": train_scores_out.astype(np.float32, copy=False),
            "train_scores_gnn_only": np.asarray(train_replay_inputs_out.get("scores_gnn_only", []), dtype=np.float32),
            "train_labels": train_labels_out.astype(np.float32, copy=False),
            "train_pair_features": np.asarray(train_replay_inputs_out.get("pair_features", []), dtype=np.float32),
            "train_decoder_input": np.asarray(train_replay_inputs_out.get("decoder_input", []), dtype=np.float32),
            "train_edge_hidden": np.asarray(train_replay_inputs_out.get("edge_hidden", []), dtype=np.float32),
            "train_src_degree": np.asarray(train_replay_inputs_out.get("src_degree", []), dtype=np.float32),
            "train_dst_degree": np.asarray(train_replay_inputs_out.get("dst_degree", []), dtype=np.float32),
            "train_edge_src": np.asarray(train_replay_inputs_out.get("edge_src", []), dtype=np.int64),
            "train_edge_dst": np.asarray(train_replay_inputs_out.get("edge_dst", []), dtype=np.int64),
            "val_scores": val_scores.astype(np.float32, copy=False),
            "val_scores_gnn_only": val_scores_gnn_only.astype(np.float32, copy=False),
            "val_labels": val_labels.astype(np.float32, copy=False),
            "val_pair_features": np.asarray(val_replay_inputs.get("pair_features", []), dtype=np.float32),
            "val_decoder_input": np.asarray(val_replay_inputs.get("decoder_input", []), dtype=np.float32),
            "val_edge_hidden": np.asarray(val_replay_inputs.get("edge_hidden", []), dtype=np.float32),
            "val_src_degree": np.asarray(val_replay_inputs.get("src_degree", []), dtype=np.float32),
            "val_dst_degree": np.asarray(val_replay_inputs.get("dst_degree", []), dtype=np.float32),
            "val_edge_src": np.asarray(val_replay_inputs.get("edge_src", []), dtype=np.int64),
            "val_edge_dst": np.asarray(val_replay_inputs.get("edge_dst", []), dtype=np.int64),
        }

        eval_replay_payload = {
            "dataset_name": np.asarray([self.dataset_name], dtype="<U64"),
            "run_tag": np.asarray([self.run_tag], dtype="<U128"),
            "trainer_seed": np.asarray([int(self.seed)], dtype=np.int64),
            "task": np.asarray([self.target_task], dtype="<U64"),
            "online_symbolic_enabled": np.asarray([int(self._online_symbolic_enabled)], dtype=np.int64),
            "online_symbolic_spec_id": np.asarray([self._online_symbolic_spec_id], dtype="<U128"),
            "online_symbolic_repr": np.asarray([self._online_symbolic_repr_effective], dtype="<U32"),
            "online_symbolic_raw_dim": np.asarray([int(self._online_symbolic_raw_dim)], dtype=np.int64),
            "online_symbolic_tile_repeat": np.asarray([int(self.online_symbolic_tile_repeat)], dtype=np.int64),
            "online_symbolic_effective_dim": np.asarray([int(self._online_symbolic_effective_dim)], dtype=np.int64),
            "online_symbolic_normalize": np.asarray([self.online_symbolic_normalize], dtype="<U32"),
            "pair_feature_order": np.asarray(pair_feature_order, dtype="<U64"),
            "val_scores": val_scores.astype(np.float32, copy=False),
            "val_scores_gnn_only": val_scores_gnn_only.astype(np.float32, copy=False),
            "val_labels": val_labels.astype(np.float32, copy=False),
            "val_pair_features": np.asarray(val_replay_inputs.get("pair_features", []), dtype=np.float32),
            "val_decoder_input": np.asarray(val_replay_inputs.get("decoder_input", []), dtype=np.float32),
            "val_edge_hidden": np.asarray(val_replay_inputs.get("edge_hidden", []), dtype=np.float32),
            "val_src_degree": np.asarray(val_replay_inputs.get("src_degree", []), dtype=np.float32),
            "val_dst_degree": np.asarray(val_replay_inputs.get("dst_degree", []), dtype=np.float32),
            "val_edge_src": np.asarray(val_replay_inputs.get("edge_src", []), dtype=np.int64),
            "val_edge_dst": np.asarray(val_replay_inputs.get("edge_dst", []), dtype=np.int64),
            "test_scores": test_scores.astype(np.float32, copy=False),
            "test_scores_gnn_only": test_scores_gnn_only.astype(np.float32, copy=False),
            "test_labels": test_labels.astype(np.float32, copy=False),
            "test_pair_features": np.asarray(test_replay_inputs.get("pair_features", []), dtype=np.float32),
            "test_decoder_input": np.asarray(test_replay_inputs.get("decoder_input", []), dtype=np.float32),
            "test_edge_hidden": np.asarray(test_replay_inputs.get("edge_hidden", []), dtype=np.float32),
            "test_src_degree": np.asarray(test_replay_inputs.get("src_degree", []), dtype=np.float32),
            "test_dst_degree": np.asarray(test_replay_inputs.get("dst_degree", []), dtype=np.float32),
            "test_edge_src": np.asarray(test_replay_inputs.get("edge_src", []), dtype=np.int64),
            "test_edge_dst": np.asarray(test_replay_inputs.get("edge_dst", []), dtype=np.int64),
        }
        replay_payload = {**selection_replay_payload, **eval_replay_payload}
        np.savez(replay_path, **replay_payload)
        logger.info("Replay inputs saved to %s", replay_path)

        archive_dir = self._resolve_replay_archive_dir()
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)
            archive_scores_path = os.path.join(archive_dir, f"{self.target_task}_scores.npz")
            archive_replay_path = os.path.join(archive_dir, f"{self.target_task}_replay_inputs.npz")
            selection_archive_dir = self._resolve_replay_archive_dir_for_track("selection_replay")
            eval_archive_dir = self._resolve_replay_archive_dir_for_track("eval_replay")
            os.makedirs(selection_archive_dir, exist_ok=True)
            os.makedirs(eval_archive_dir, exist_ok=True)
            selection_archive_replay_path = os.path.join(selection_archive_dir, f"{self.target_task}_replay_inputs.npz")
            eval_archive_replay_path = os.path.join(eval_archive_dir, f"{self.target_task}_replay_inputs.npz")
            np.savez(archive_scores_path, **scores_payload)
            np.savez(archive_replay_path, **replay_payload)
            np.savez(selection_archive_replay_path, **selection_replay_payload)
            np.savez(eval_archive_replay_path, **eval_replay_payload)
            logger.info(
                "Replay artifacts archived to %s (scores=%s replay=%s selection=%s eval=%s)",
                archive_dir,
                archive_scores_path,
                archive_replay_path,
                selection_archive_replay_path,
                eval_archive_replay_path,
            )
        record_profile_event(
            "trainer_final_metrics",
            {
                "target_task": self.target_task,
                "best_threshold": float(best_threshold),
                "eval_threshold_mode": self.eval_threshold_mode,
                "rerank_enabled": bool(rerank_collect),
                "rerank_alpha": float(rerank_cfg["alpha"]),
                "rerank_delta_cap": float(rerank_cfg["delta_cap"]),
                "rerank_gate_margin": float(rerank_cfg["gate_margin"]),
                "rerank_gate_delta_min": float(rerank_cfg["gate_delta_min"]),
                "rerank_gate_ratio": float(rerank_cfg["gate_ratio"]),
                "rerank_benefit_enabled": bool(self.benefit_selector is not None and self.rerank_benefit_enable),
                "rerank_benefit_min_score": float(rerank_cfg.get("benefit_min_score", 0.0)),
                "rerank_benefit_ratio": float(rerank_cfg.get("benefit_ratio", 1.0)),
                "rerank_active_ratio": float(rerank_cfg.get("active_ratio", 1.0)),
                "rerank_reference_threshold": float(rerank_cfg["reference_threshold"]),
                "rerank_task_residual_enabled": bool(rerank_cfg.get("task_residual_enabled", False)),
                "rerank_task_residual_scale": float(rerank_cfg.get("task_residual_scale", 0.0)),
                "rerank_task_residual_margin": float(rerank_cfg.get("task_residual_margin", 0.0)),
                "task_residual_summary": task_residual_summary,
                "test_metrics": test_metrics,
            },
        )

        return test_metrics
