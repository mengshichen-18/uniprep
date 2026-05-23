from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from generated_feature_runtime import GeneratedFeatureRegistry, load_generated_feature_registry


logger = logging.getLogger(__name__)


class JoinablePairFeatureStore:
    """Compute and cache OmniMatch-style pair features for JTS col-col pairs."""

    _REPO_DIR = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_TABLE_ROOT_BASE = os.path.normpath(
        os.getenv(
            "TABLE_ROOT_BASE",
            os.path.join(_REPO_DIR, "..", "..", "datasets_joint_discovery_integration_split_work"),
        )
    )
    _DATASET_TABLE_ROOTS: Dict[str, str] = {
        "wikidbs": os.path.join(_DEFAULT_TABLE_ROOT_BASE, "wikidbs_040303", "datalake_plus"),
        "santos_benchmark": os.path.join(_DEFAULT_TABLE_ROOT_BASE, "santos_benchmark_040303", "datalake_plus"),
        "magellan": os.path.join(_DEFAULT_TABLE_ROOT_BASE, "magellan_040303", "datalake_plus"),
    }
    _ATOM_GROUPS: Dict[str, Tuple[str, ...]] = {
        "jaccard_containment": ("jaccard", "containment_max"),
        "value_distribution": ("value_distribution",),
        "overlap_coverage": ("coverage_a", "coverage_b", "coverage_max"),
        "value_profile": ("unique_ratio_sim", "numeric_ratio_sim", "avg_len_ratio_sim"),
        "header_similarity": ("header_token_jaccard", "header_edit_similarity"),
    }
    _CANONICAL_ATOM_ORDER: Tuple[str, ...] = (
        "jaccard",
        "containment_max",
        "value_distribution",
        "coverage_a",
        "coverage_b",
        "coverage_max",
        "unique_ratio_sim",
        "numeric_ratio_sim",
        "avg_len_ratio_sim",
        "header_token_jaccard",
        "header_edit_similarity",
    )
    _SUPPORTED_ATOMS: Set[str] = set(_CANONICAL_ATOM_ORDER)
    _VALUE_BASED_ATOMS: Set[str] = {
        "jaccard",
        "containment_max",
        "coverage_a",
        "coverage_b",
        "coverage_max",
        "unique_ratio_sim",
        "numeric_ratio_sim",
        "avg_len_ratio_sim",
    }

    @classmethod
    def _resolve_requested_atoms(
        cls,
        feature_names: List[str],
        *,
        generated_feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        requested_tokens = [name.strip() for name in feature_names if name and name.strip()]
        requested_tokens = [name for name in requested_tokens if name != "none"]

        expanded_atoms: List[str] = []
        generated_atoms: List[str] = []
        unknown: List[str] = []
        generated_set = set(str(name).strip() for name in (generated_feature_names or []) if str(name).strip())
        for token in requested_tokens:
            if token in cls._ATOM_GROUPS:
                expanded_atoms.extend(list(cls._ATOM_GROUPS[token]))
            elif token in cls._SUPPORTED_ATOMS:
                expanded_atoms.append(token)
            elif token in generated_set:
                generated_atoms.append(token)
            else:
                unknown.append(token)

        if unknown:
            supported = sorted(set(cls._ATOM_GROUPS.keys()).union(cls._SUPPORTED_ATOMS).union(generated_set))
            raise ValueError(
                f"Unsupported JTS pair features/atoms: {unknown}. Supported: {supported}"
            )

        atom_set = set(expanded_atoms)
        atom_order = [name for name in cls._CANONICAL_ATOM_ORDER if name in atom_set] + list(generated_atoms)
        return requested_tokens, atom_order

    def __init__(
        self,
        *,
        graph,
        dataset_name: str,
        feature_names: List[str],
        table_root_override: str = "",
        required_node_ids: Optional[Iterable[int]] = None,
        generated_feature_specs_path: str = "",
    ) -> None:
        self.generated_feature_specs_path = str(generated_feature_specs_path).strip()
        self.generated_feature_registry: Optional[GeneratedFeatureRegistry] = None
        generated_feature_names: List[str] = []
        if self.generated_feature_specs_path:
            self.generated_feature_registry = load_generated_feature_registry(
                self.generated_feature_specs_path,
                expected_task="joinable_table_search",
                expected_scope="column_pair",
            )
            generated_feature_names = list(self.generated_feature_registry.feature_names)

        requested_tokens, atom_order = self._resolve_requested_atoms(
            feature_names,
            generated_feature_names=generated_feature_names,
        )
        self.requested_feature_tokens = requested_tokens
        self.feature_names = requested_tokens
        self.feature_order = atom_order
        self.feature_dim = len(self.feature_order)
        self._feature_set: Set[str] = set(self.feature_order)

        self.table_root = self._resolve_table_root(dataset_name=dataset_name, override=table_root_override)
        self.node_id_to_column = self._load_node_id_to_column(graph)
        self._table_cache: Dict[str, Dict[str, List[str]]] = {}
        self._pair_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._warned_missing_tables: Set[str] = set()

        logger.info(
            f"[JTS-PairFeat] enabled={self.feature_names} resolved_atoms={self.feature_order} dim={self.feature_dim} "
            f"table_root={self.table_root} mapped_columns={len(self.node_id_to_column)} "
            f"generated_features={0 if self.generated_feature_registry is None else len(self.generated_feature_registry.feature_names)}"
        )

    @staticmethod
    def _resolve_table_root(*, dataset_name: str, override: str) -> str:
        if override:
            if not os.path.isdir(override):
                raise FileNotFoundError(f"--jts_table_root does not exist: {override}")
            return override
        if dataset_name not in JoinablePairFeatureStore._DATASET_TABLE_ROOTS:
            raise ValueError(
                f"Unknown dataset_name={dataset_name} for JTS pair features. "
                f"Expected one of: {sorted(JoinablePairFeatureStore._DATASET_TABLE_ROOTS.keys())}"
            )
        table_root = JoinablePairFeatureStore._DATASET_TABLE_ROOTS[dataset_name]
        if not os.path.isdir(table_root):
            raise FileNotFoundError(f"Auto-resolved table root does not exist: {table_root}")
        return table_root

    @staticmethod
    def _load_node_id_to_column(graph) -> Dict[int, Tuple[str, str]]:
        mapping_path = os.path.join(graph.data_dir, "node_id_mapping.json")
        with open(mapping_path, "r") as handle:
            node_map = json.load(handle)
        raw_column_map = node_map.get("column", {})

        out: Dict[int, Tuple[str, str]] = {}
        for key, node_id in raw_column_map.items():
            try:
                parsed = ast.literal_eval(key)
            except Exception:
                continue
            if not (isinstance(parsed, tuple) and len(parsed) == 2):
                continue
            table_name, column_name = parsed
            out[int(node_id)] = (str(table_name), str(column_name))
        return out

    @staticmethod
    def _normalize_cell_value(raw: str) -> str:
        value = str(raw).lower()
        if value.endswith(".0"):
            value = value[:-2]
        return value

    @staticmethod
    def _camel_case_split(token: str) -> List[str]:
        if not token:
            return []
        pattern = r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)"
        return [m.group(0).lower() for m in re.finditer(pattern, token)]

    @classmethod
    def _tokenize_column_name(cls, column_name: str) -> List[str]:
        initial_tokens = [tok for tok in re.split(r"_| |/|(\d+)", column_name) if tok]
        tokens: List[str] = []
        for tok in initial_tokens:
            tokens.extend(cls._camel_case_split(tok))
        return tokens

    @staticmethod
    def _jensen_shannon_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        p_a = {token: token_count / len(tokens_a) for token, token_count in Counter(tokens_a).items()}
        p_b = {token: token_count / len(tokens_b) for token, token_count in Counter(tokens_b).items()}

        kl_a = 0.0
        kl_b = 0.0
        for token, freq in p_a.items():
            freq_b = p_b.get(token, 0.0)
            kl_a += freq * math.log(freq / (0.5 * (freq + freq_b)))
        for token, freq in p_b.items():
            freq_a = p_a.get(token, 0.0)
            kl_b += freq * math.log(freq / (0.5 * (freq + freq_a)))

        js = 0.5 * (kl_a + kl_b)
        return float(1.0 - js)

    @staticmethod
    def _safe_ratio_float(a: float, b: float) -> float:
        a = float(a)
        b = float(b)
        if a <= 0.0 or b <= 0.0:
            return 0.0
        lo = min(a, b)
        hi = max(a, b)
        return float(lo) / float(hi)

    @staticmethod
    def _token_jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return float(inter) / float(union) if union > 0 else 0.0

    @staticmethod
    def _normalized_edit_similarity(text_a: str, text_b: str) -> float:
        a = str(text_a)
        b = str(text_b)
        if a == b:
            return 1.0
        n = len(a)
        m = len(b)
        if n == 0 or m == 0:
            return 0.0
        # For long strings, exact Levenshtein DP becomes prohibitively expensive on large EM splits.
        # Use a linear-time char n-gram Jaccard proxy to keep runtime stable.
        if max(n, m) > 128:
            k = 3

            def _ngrams(text: str) -> Set[str]:
                if len(text) <= k:
                    return {text} if text else set()
                return {text[i : i + k] for i in range(0, len(text) - k + 1)}

            grams_a = _ngrams(a)
            grams_b = _ngrams(b)
            if not grams_a or not grams_b:
                return 0.0
            inter = len(grams_a.intersection(grams_b))
            union = len(grams_a.union(grams_b))
            return float(inter) / float(union) if union > 0 else 0.0
        prev = list(range(m + 1))
        for i, ch_a in enumerate(a, start=1):
            curr = [i] + [0] * m
            for j, ch_b in enumerate(b, start=1):
                cost = 0 if ch_a == ch_b else 1
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            prev = curr
        dist = prev[m]
        return max(0.0, 1.0 - float(dist) / float(max(n, m)))

    @staticmethod
    def _is_numeric_like(value: str) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if text == "":
            return False
        try:
            float(text)
            return True
        except Exception:
            return False

    @classmethod
    def _numeric_ratio(cls, values: List[str]) -> float:
        if not values:
            return 0.0
        numeric_cnt = sum(1 for value in values if cls._is_numeric_like(value))
        return float(numeric_cnt) / float(len(values))

    @staticmethod
    def _avg_str_len(values: List[str]) -> float:
        if not values:
            return 0.0
        return float(sum(len(str(v)) for v in values)) / float(len(values))

    def _load_table_columns(self, table_name: str) -> Dict[str, List[str]]:
        cached = self._table_cache.get(table_name)
        if cached is not None:
            return cached

        table_path = os.path.join(self.table_root, table_name)
        if not os.path.isfile(table_path):
            table_path_csv = table_path + ".csv"
            if os.path.isfile(table_path_csv):
                table_path = table_path_csv
        if not os.path.isfile(table_path):
            if table_name not in self._warned_missing_tables:
                logger.warning(f"[JTS-PairFeat] missing table file: {table_path}")
                self._warned_missing_tables.add(table_name)
            self._table_cache[table_name] = {}
            return self._table_cache[table_name]

        with open(table_path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            columns: Dict[str, List[str]] = {name: [] for name in headers}
            for row in reader:
                for name in headers:
                    raw = row.get(name, "")
                    if raw is None:
                        continue
                    raw_str = str(raw)
                    if raw_str == "":
                        continue
                    norm = self._normalize_cell_value(raw_str)
                    if norm == "":
                        continue
                    columns[name].append(norm)

        alias_map: Dict[str, List[str]] = {}
        for name, values in columns.items():
            alias_map[name] = values
            rstrip_name = name.rstrip()
            strip_name = name.strip()
            if rstrip_name not in alias_map:
                alias_map[rstrip_name] = values
            if strip_name not in alias_map:
                alias_map[strip_name] = values

        self._table_cache[table_name] = alias_map
        return alias_map

    def _column_values(self, table_name: str, column_name: str) -> List[str]:
        table_cols = self._load_table_columns(table_name)
        if column_name in table_cols:
            return table_cols[column_name]
        if column_name.rstrip() in table_cols:
            return table_cols[column_name.rstrip()]
        if column_name.strip() in table_cols:
            return table_cols[column_name.strip()]
        return []

    @staticmethod
    def _canonical_pair(src_id: int, dst_id: int) -> Tuple[int, int]:
        if src_id <= dst_id:
            return src_id, dst_id
        return dst_id, src_id

    @staticmethod
    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, float(value))))

    @classmethod
    def _set_jaccard(cls, values_a: Set[str], values_b: Set[str]) -> float:
        return float(cls._token_jaccard(list(values_a), list(values_b)))

    def _generated_feature_helpers(self) -> Dict[str, object]:
        return {
            "safe_ratio_float": self._safe_ratio_float,
            "token_jaccard": self._token_jaccard,
            "normalized_edit_similarity": self._normalized_edit_similarity,
            "clip01": self._clip01,
            "set_jaccard": self._set_jaccard,
        }

    def _compute_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        if self.feature_dim == 0:
            return np.zeros((0,), dtype=np.float32)

        col_a = self.node_id_to_column.get(int(src_id))
        col_b = self.node_id_to_column.get(int(dst_id))
        if col_a is None or col_b is None:
            return np.zeros((self.feature_dim,), dtype=np.float32)

        table_a, name_a = col_a
        table_b, name_b = col_b

        atom_values: Dict[str, float] = {name: 0.0 for name in self.feature_order}
        need_value_based = bool(self._feature_set.intersection(self._VALUE_BASED_ATOMS))
        values_a: List[str] = []
        values_b: List[str] = []
        set_a: Set[str] = set()
        set_b: Set[str] = set()
        inter: Set[str] = set()
        if need_value_based:
            values_a = self._column_values(table_a, name_a)
            values_b = self._column_values(table_b, name_b)
            set_a = {v for v in values_a if v is not None}
            set_b = {v for v in values_b if v is not None}
            inter = set_a.intersection(set_b)

        if "jaccard" in self._feature_set or "containment_max" in self._feature_set:
            diff_a = set_a.difference(set_b)
            diff_b = set_b.difference(set_a)

            sig_a = len(inter) == 0 and len(diff_b) == 0
            sig_b = len(inter) == 0 and len(diff_a) == 0
            if sig_a and sig_b:
                jaccard = 0.0
            else:
                denom = float(len(diff_a) + len(diff_b) + len(inter))
                jaccard = float(len(inter)) / denom if denom > 0 else 0.0

            if sig_a:
                containment_a = 0.0
            else:
                denom_a = float(len(diff_b) + len(inter))
                containment_a = float(len(inter)) / denom_a if denom_a > 0 else 0.0
            if sig_b:
                containment_b = 0.0
            else:
                denom_b = float(len(diff_a) + len(inter))
                containment_b = float(len(inter)) / denom_b if denom_b > 0 else 0.0

            containment_max = max(containment_a, containment_b)
            if "jaccard" in atom_values:
                atom_values["jaccard"] = float(jaccard)
            if "containment_max" in atom_values:
                atom_values["containment_max"] = float(containment_max)

        if "value_distribution" in self._feature_set:
            tokens_a = self._tokenize_column_name(name_a)
            tokens_b = self._tokenize_column_name(name_b)
            atom_values["value_distribution"] = float(self._jensen_shannon_similarity(tokens_a, tokens_b))

        if (
            "coverage_a" in self._feature_set
            or "coverage_b" in self._feature_set
            or "coverage_max" in self._feature_set
        ):
            coverage_a = float(len(inter)) / float(len(set_a)) if set_a else 0.0
            coverage_b = float(len(inter)) / float(len(set_b)) if set_b else 0.0
            coverage_max = max(coverage_a, coverage_b)
            if "coverage_a" in atom_values:
                atom_values["coverage_a"] = float(coverage_a)
            if "coverage_b" in atom_values:
                atom_values["coverage_b"] = float(coverage_b)
            if "coverage_max" in atom_values:
                atom_values["coverage_max"] = float(coverage_max)

        if (
            "unique_ratio_sim" in self._feature_set
            or "numeric_ratio_sim" in self._feature_set
            or "avg_len_ratio_sim" in self._feature_set
        ):
            unique_ratio_a = float(len(set_a)) / float(len(values_a)) if values_a else 0.0
            unique_ratio_b = float(len(set_b)) / float(len(values_b)) if values_b else 0.0
            numeric_ratio_a = self._numeric_ratio(values_a)
            numeric_ratio_b = self._numeric_ratio(values_b)
            avg_len_a = self._avg_str_len(values_a)
            avg_len_b = self._avg_str_len(values_b)
            if "unique_ratio_sim" in atom_values:
                atom_values["unique_ratio_sim"] = float(
                    self._safe_ratio_float(unique_ratio_a, unique_ratio_b)
                )
            if "numeric_ratio_sim" in atom_values:
                atom_values["numeric_ratio_sim"] = float(1.0 - abs(numeric_ratio_a - numeric_ratio_b))
            if "avg_len_ratio_sim" in atom_values:
                atom_values["avg_len_ratio_sim"] = float(self._safe_ratio_float(avg_len_a, avg_len_b))

        if "header_token_jaccard" in self._feature_set or "header_edit_similarity" in self._feature_set:
            tokens_a = self._tokenize_column_name(name_a)
            tokens_b = self._tokenize_column_name(name_b)
            token_jaccard = self._token_jaccard(tokens_a, tokens_b)
            edit_sim = self._normalized_edit_similarity(name_a.strip().lower(), name_b.strip().lower())
            if "header_token_jaccard" in atom_values:
                atom_values["header_token_jaccard"] = float(token_jaccard)
            if "header_edit_similarity" in atom_values:
                atom_values["header_edit_similarity"] = float(edit_sim)

        if self.generated_feature_registry is not None:
            unique_ratio_a = float(len(set_a)) / float(len(values_a)) if values_a else 0.0
            unique_ratio_b = float(len(set_b)) / float(len(values_b)) if values_b else 0.0
            stats_a = {
                "value_set": set_a,
                "values": list(values_a),
                "unique_ratio": unique_ratio_a,
                "numeric_ratio": self._numeric_ratio(values_a),
                "avg_len": self._avg_str_len(values_a),
                "header_tokens": set(self._tokenize_column_name(name_a)),
                "header_text": name_a.strip().lower(),
            }
            stats_b = {
                "value_set": set_b,
                "values": list(values_b),
                "unique_ratio": unique_ratio_b,
                "numeric_ratio": self._numeric_ratio(values_b),
                "avg_len": self._avg_str_len(values_b),
                "header_tokens": set(self._tokenize_column_name(name_b)),
                "header_text": name_b.strip().lower(),
            }
            ctx = {
                "stats_a": stats_a,
                "stats_b": stats_b,
                "helpers": self._generated_feature_helpers(),
            }
            generated_values = self.generated_feature_registry.compute(ctx)
            for name, value in generated_values.items():
                if name in atom_values:
                    atom_values[name] = float(value)

        feats = [float(atom_values[name]) for name in self.feature_order]
        if not feats:
            return np.zeros((self.feature_dim,), dtype=np.float32)
        return np.asarray(feats, dtype=np.float32)

    def get_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        key = self._canonical_pair(int(src_id), int(dst_id))
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        feats = self._compute_pair_features(key[0], key[1])
        self._pair_cache[key] = feats
        return feats

    def build_batch_features(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.feature_dim == 0:
            return torch.zeros((int(src_ids.numel()), 0), dtype=dtype, device=device)
        src = src_ids.detach().cpu().tolist()
        dst = dst_ids.detach().cpu().tolist()
        rows = [self.get_pair_features(int(u), int(v)) for u, v in zip(src, dst)]
        if not rows:
            return torch.zeros((0, self.feature_dim), dtype=dtype, device=device)
        arr = np.stack(rows, axis=0)
        return torch.tensor(arr, dtype=dtype, device=device)


class EntityPairFeatureStore:
    """Compute lightweight row-row pair features for entity matching without prepool."""

    _DATASET_TABLE_ROOTS: Dict[str, str] = JoinablePairFeatureStore._DATASET_TABLE_ROOTS
    _NULL_LIKE: Set[str] = {"", "nan", "none", "null", "na", "n/a", "unknown", "-", "unknown_cell"}
    _MAX_ROW_VALUES = 64
    _MAX_ROW_TOKENS = 128
    _MAX_SERIAL_FIELDS = 48
    _MAX_SERIAL_CHARS = 512
    _MAX_IDF_ROWS = 500000
    _ATOM_GROUPS: Dict[str, Tuple[str, ...]] = {
        "embedding_similarity": ("row_emb_cosine", "row_emb_l1_sim"),
        "row_value_overlap": ("row_value_jaccard", "row_value_containment_max", "row_token_jaccard"),
        "row_profile": ("row_nonempty_ratio", "row_numeric_ratio_sim", "row_avg_len_ratio"),
        # Row-serialization alignment family: lexical serialization + numeric value overlap.
        "serial_value_alignment": ("row_serial_token_jaccard", "row_serial_edit_similarity", "row_numeric_value_overlap"),
        # Backward-compat alias; prefer `serial_value_alignment` in new configs/prompts.
        "ditto_proxy": ("row_serial_token_jaccard", "row_serial_edit_similarity", "row_numeric_value_overlap"),
        # Extra lexical separability features for long/heterogeneous rows.
        "serial_lexical_plus": (
            "row_serial_char3_jaccard",
            "row_serial_char4_jaccard",
            "row_token_idf_jaccard",
            "row_numeric_rel_diff_sim",
        ),
        # PromptEM-like schema-aware alignment signal for row pairs.
        "header_alignment": (
            "row_header_jaccard",
            "row_header_value_exact_ratio",
            "row_header_token_jaccard",
        ),
    }
    _CANONICAL_ATOM_ORDER: Tuple[str, ...] = (
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
        "row_header_jaccard",
        "row_header_value_exact_ratio",
        "row_header_token_jaccard",
    )
    _SUPPORTED_ATOMS: Set[str] = set(_CANONICAL_ATOM_ORDER)

    @classmethod
    def _resolve_requested_atoms(
        cls,
        feature_names: List[str],
        *,
        generated_feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        requested_tokens = [name.strip() for name in feature_names if name and name.strip()]
        requested_tokens = [name for name in requested_tokens if name != "none"]

        expanded_atoms: List[str] = []
        generated_atoms: List[str] = []
        unknown: List[str] = []
        generated_set = set(str(name).strip() for name in (generated_feature_names or []) if str(name).strip())
        for token in requested_tokens:
            if token in cls._ATOM_GROUPS:
                expanded_atoms.extend(list(cls._ATOM_GROUPS[token]))
            elif token in cls._SUPPORTED_ATOMS:
                expanded_atoms.append(token)
            elif token in generated_set:
                generated_atoms.append(token)
            else:
                unknown.append(token)

        if unknown:
            supported = sorted(set(cls._ATOM_GROUPS.keys()).union(cls._SUPPORTED_ATOMS).union(generated_set))
            raise ValueError(
                f"Unsupported EM pair features/atoms: {unknown}. Supported: {supported}"
            )

        # Keep caller-specified order and multiplicity to support explicit
        # compositions such as static6 + pair_tail15 (static21 with duplicates).
        atom_order = list(expanded_atoms) + list(generated_atoms)
        return requested_tokens, atom_order

    def __init__(
        self,
        *,
        graph,
        dataset_name: str,
        feature_names: List[str],
        table_root_override: str = "",
        required_node_ids: Optional[Iterable[int]] = None,
        row_stats_mode: str = "full",
        pair_cache_mode: str = "off",
        pair_cache_root: str = "",
        generated_feature_specs_path: str = "",
    ) -> None:
        self.generated_feature_specs_path = str(generated_feature_specs_path).strip()
        self.generated_feature_registry: Optional[GeneratedFeatureRegistry] = None
        generated_feature_names: List[str] = []
        if self.generated_feature_specs_path:
            self.generated_feature_registry = load_generated_feature_registry(
                self.generated_feature_specs_path,
                expected_task="entity_matching",
            )
            generated_feature_names = list(self.generated_feature_registry.feature_names)

        self.feature_names, self.feature_order = self._resolve_requested_atoms(
            feature_names,
            generated_feature_names=generated_feature_names,
        )
        self.feature_dim = len(self.feature_order)
        self._feature_set = set(self.feature_order)
        self._row_stats_mode = str(row_stats_mode).strip().lower()
        if self._row_stats_mode not in {"required", "full"}:
            raise ValueError(
                f"Unsupported row_stats_mode={self._row_stats_mode}. Expected one of: required,full"
            )
        self._pair_cache_mode = str(pair_cache_mode).strip().lower()
        if self._pair_cache_mode not in {"off", "readwrite"}:
            raise ValueError(
                f"Unsupported pair_cache_mode={self._pair_cache_mode}. Expected one of: off,readwrite"
            )

        self.table_root = JoinablePairFeatureStore._resolve_table_root(
            dataset_name=dataset_name,
            override=table_root_override,
        )
        required_nodes: Optional[Set[int]] = None
        if required_node_ids is not None:
            required_nodes = set()
            for item in required_node_ids:
                try:
                    required_nodes.add(int(item))
                except Exception:
                    continue
        self.node_id_to_row = self._load_node_id_to_row(graph, required_node_ids=required_nodes)
        if self._row_stats_mode == "required":
            self._required_rows_by_table = self._build_required_rows_by_table()
        else:
            # full mode does not consume required-row index; avoid an extra large pass
            self._required_rows_by_table = {}
        self._table_cache: Dict[str, object] = {}
        self._pair_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._warned_missing_tables: Set[str] = set()
        self._node_embeddings = np.load(os.path.join(graph.data_dir, "node_embeddings.npy"), mmap_mode="r")
        self._pair_cache_hit = 0
        self._pair_cache_miss = 0
        self._split_cache_hit = 0
        self._split_cache_miss = 0
        default_cache_root = os.path.join(graph.data_dir, ".em_pair_cache")
        self._pair_cache_root = pair_cache_root.strip() if pair_cache_root else default_cache_root
        if self._pair_cache_mode == "readwrite":
            os.makedirs(self._pair_cache_root, exist_ok=True)
        cfg_payload = {
            "table_root": os.path.abspath(self.table_root),
            "feature_order": list(self.feature_order),
            "row_stats_mode": self._row_stats_mode,
            "generated_feature_fingerprint": (
                str(self.generated_feature_registry.fingerprint)
                if self.generated_feature_registry is not None
                else ""
            ),
        }
        cfg_hash = hashlib.sha1(json.dumps(cfg_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        self._split_cache_prefix = f"{cfg_hash}_d{int(self.feature_dim)}"
        self._idf_token_weights: Dict[str, float] = {}
        self._idf_num_rows = 0
        self._idf_train_csv_path = ""
        if "row_token_idf_jaccard" in self._feature_set:
            self._fit_token_idf_from_train_split()

        logger.info(
            f"[EM-PairFeat] enabled={self.feature_names} dim={self.feature_dim} "
            f"table_root={self.table_root} mapped_rows={len(self.node_id_to_row)} "
            f"row_stats_mode={self._row_stats_mode} pair_cache_mode={self._pair_cache_mode} "
            f"generated_features={0 if self.generated_feature_registry is None else len(self.generated_feature_registry.feature_names)}"
        )

    def _build_required_rows_by_table(self) -> Dict[str, Set[int]]:
        out: Dict[str, Set[int]] = {}
        for table_name, row_idx in self.node_id_to_row.values():
            table = str(table_name).strip()
            if table == "":
                continue
            out.setdefault(table, set()).add(int(row_idx))
        return out

    @staticmethod
    def _load_node_id_to_row(
        graph,
        *,
        required_node_ids: Optional[Set[int]] = None,
    ) -> Dict[int, Tuple[str, int]]:
        mapping_path = os.path.join(graph.data_dir, "node_id_mapping.json")
        with open(mapping_path, "r") as handle:
            node_map = json.load(handle)
        raw_row_map = node_map.get("row", {})

        out: Dict[int, Tuple[str, int]] = {}
        target_size = len(required_node_ids) if required_node_ids is not None else None
        for key, node_id in raw_row_map.items():
            try:
                node_id_int = int(node_id)
            except Exception:
                continue
            if required_node_ids is not None and node_id_int not in required_node_ids:
                continue
            try:
                parsed = ast.literal_eval(key)
            except Exception:
                continue
            if not (isinstance(parsed, tuple) and len(parsed) == 2):
                continue
            table_name, row_idx = parsed
            try:
                out[node_id_int] = (str(table_name), int(row_idx))
            except Exception:
                continue
            if target_size is not None and len(out) >= target_size:
                break
        return out

    @staticmethod
    def _canonical_pair(src_id: int, dst_id: int) -> Tuple[int, int]:
        if src_id <= dst_id:
            return src_id, dst_id
        return dst_id, src_id

    def _resolve_table_path(self, table_name: str) -> str:
        table_path = os.path.join(self.table_root, table_name)
        if not os.path.isfile(table_path):
            table_path_csv = table_path + ".csv"
            if os.path.isfile(table_path_csv):
                table_path = table_path_csv
        if os.path.isfile(table_path):
            return table_path
        if table_name not in self._warned_missing_tables:
            logger.warning(f"[EM-PairFeat] missing table file: {table_path}")
            self._warned_missing_tables.add(table_name)
        return ""

    @staticmethod
    def _tokenize_value(text: str) -> List[str]:
        return [tok for tok in re.split(r"[^a-z0-9]+", str(text).lower()) if tok]

    @classmethod
    def _tokenize_header(cls, header: str) -> List[str]:
        tokens = JoinablePairFeatureStore._tokenize_column_name(str(header))
        if tokens:
            return tokens
        return cls._tokenize_value(header)

    @staticmethod
    def _char_ngram_set(text: str, n: int) -> Set[str]:
        value = str(text or "").strip().lower()
        if value == "":
            return set()
        if len(value) <= n:
            return {value}
        return {value[i : i + n] for i in range(0, len(value) - n + 1)}

    @classmethod
    def _char_ngram_jaccard(cls, text_a: str, text_b: str, n: int) -> float:
        grams_a = cls._char_ngram_set(text_a, n)
        grams_b = cls._char_ngram_set(text_b, n)
        if not grams_a or not grams_b:
            return 0.0
        inter = len(grams_a.intersection(grams_b))
        union = len(grams_a.union(grams_b))
        return float(inter) / float(union) if union > 0 else 0.0

    def _idf_weight(self, token: str) -> float:
        return float(self._idf_token_weights.get(token, 1.0))

    def _weighted_jaccard(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        inter = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        if not union:
            return 0.0
        inter_w = sum(self._idf_weight(token) for token in inter)
        union_w = sum(self._idf_weight(token) for token in union)
        if union_w <= 0.0:
            return 0.0
        return float(inter_w) / float(union_w)

    @staticmethod
    def _coerce_row_index(raw: object) -> Optional[int]:
        try:
            return int(raw)
        except Exception:
            try:
                return int(float(str(raw).strip()))
            except Exception:
                return None

    def _resolve_em_train_csv_path(self) -> str:
        table_root = str(self.table_root).rstrip(os.sep)
        dataset_root = os.path.dirname(table_root)
        candidates = [
            os.path.join(dataset_root, "label_plus", "entity_matching", "train.csv"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return ""

    @staticmethod
    def _table_name_candidates(table_name: str) -> List[str]:
        name = str(table_name).strip()
        if name == "":
            return []
        candidates = [name]
        if name.endswith(".csv"):
            candidates.append(name[:-4])
        else:
            candidates.append(name + ".csv")
        out: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return out

    def _resolve_required_rows_for_table(self, table_name: str) -> Set[int]:
        rows: Set[int] = set()
        for candidate in self._table_name_candidates(table_name):
            rows.update(self._required_rows_by_table.get(candidate, set()))
        return rows

    def _register_required_rows(self, refs: Iterable[Tuple[str, int]]) -> None:
        for table_name, row_idx in refs:
            table = str(table_name).strip()
            if table == "":
                continue
            try:
                idx = int(row_idx)
            except Exception:
                continue
            self._required_rows_by_table.setdefault(table, set()).add(idx)

    def _register_rows_from_edges(self, edges: List[Tuple[int, int]]) -> None:
        refs: List[Tuple[str, int]] = []
        for src_id, dst_id in edges:
            row_a = self.node_id_to_row.get(int(src_id))
            row_b = self.node_id_to_row.get(int(dst_id))
            if row_a is not None:
                refs.append(row_a)
            if row_b is not None:
                refs.append(row_b)
        if refs:
            self._register_required_rows(refs)

    def _row_stats_with_table_fallback(self, table_name: str, row_idx: int) -> Optional[Dict[str, object]]:
        for candidate in self._table_name_candidates(table_name):
            stats = self._row_stats(candidate, row_idx)
            if stats is not None:
                return stats
        return None

    def _fit_token_idf_from_train_split(self) -> None:
        train_csv = self._resolve_em_train_csv_path()
        self._idf_train_csv_path = train_csv
        if train_csv == "":
            logger.warning("[EM-PairFeat] row_token_idf_jaccard enabled but train.csv not found; fallback to uniform weights.")
            return

        refs: Set[Tuple[str, int]] = set()
        with open(train_csv, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pairs = (
                    (row.get("ltable_name"), row.get("l_id")),
                    (row.get("rtable_name"), row.get("r_id")),
                )
                for table_name_raw, row_id_raw in pairs:
                    row_idx = self._coerce_row_index(row_id_raw)
                    table_name = str(table_name_raw).strip() if table_name_raw is not None else ""
                    if row_idx is None or table_name == "":
                        continue
                    refs.add((table_name, row_idx))
                    if len(refs) >= int(self._MAX_IDF_ROWS):
                        break
                if len(refs) >= int(self._MAX_IDF_ROWS):
                    break

        if not refs:
            logger.warning("[EM-PairFeat] row_token_idf_jaccard has no train rows from %s; fallback to uniform weights.", train_csv)
            return
        self._register_required_rows(refs)

        df_counter: Counter = Counter()
        usable_docs = 0
        for table_name, row_idx in refs:
            stats = self._row_stats_with_table_fallback(table_name, row_idx)
            if stats is None:
                continue
            token_set = set(stats.get("token_set", set()))
            if not token_set:
                continue
            usable_docs += 1
            df_counter.update(token_set)

        if usable_docs <= 0:
            logger.warning(
                "[EM-PairFeat] row_token_idf_jaccard could not build token DF from train rows; fallback to uniform weights."
            )
            return

        self._idf_num_rows = int(usable_docs)
        denom = float(usable_docs + 1)
        self._idf_token_weights = {
            str(token): float(math.log(denom / float(freq + 1)) + 1.0)
            for token, freq in df_counter.items()
        }
        logger.info(
            "[EM-PairFeat] fitted train-only IDF: docs=%d vocab=%d source=%s",
            int(self._idf_num_rows),
            int(len(self._idf_token_weights)),
            train_csv,
        )

    @classmethod
    def _numeric_overlap_max(cls, values_a: Set[str], values_b: Set[str]) -> float:
        if not values_a or not values_b:
            return 0.0
        inter = len(values_a.intersection(values_b))
        contain_a = float(inter) / float(len(values_a)) if values_a else 0.0
        contain_b = float(inter) / float(len(values_b)) if values_b else 0.0
        return max(contain_a, contain_b)

    @classmethod
    def _build_row_stats(cls, row: Dict[str, object]) -> Dict[str, object]:
        value_set: Set[str] = set()
        token_set: Set[str] = set()
        header_token_set: Set[str] = set()
        header_value_token_set: Set[str] = set()
        serial_token_set: Set[str] = set()
        numeric_value_set: Set[str] = set()
        numeric_values_float: List[float] = []
        header_to_value: Dict[str, str] = {}
        serial_parts: List[str] = []
        nonempty_count = 0
        numeric_count = 0
        total_len = 0

        for header, raw in row.items():
            header_raw = "" if header is None else str(header).strip().lower()
            header_tokens = cls._tokenize_header(header_raw)
            for tok in header_tokens:
                header_token_set.add(tok)

            raw_str = "" if raw is None else str(raw).strip()
            if raw_str == "":
                continue
            norm = JoinablePairFeatureStore._normalize_cell_value(raw_str)
            if norm in cls._NULL_LIKE:
                continue
            nonempty_count += 1
            total_len += len(norm)
            if JoinablePairFeatureStore._is_numeric_like(norm):
                numeric_count += 1
                if len(numeric_value_set) < cls._MAX_ROW_VALUES:
                    numeric_value_set.add(norm)
                try:
                    numeric_float = float(norm)
                    if np.isfinite(numeric_float):
                        numeric_values_float.append(float(numeric_float))
                except Exception:
                    pass
            if len(value_set) < cls._MAX_ROW_VALUES:
                value_set.add(norm)
            if len(token_set) < cls._MAX_ROW_TOKENS:
                for token in cls._tokenize_value(norm):
                    token_set.add(token)
                    if len(token_set) >= cls._MAX_ROW_TOKENS:
                        break
            if len(header_to_value) < cls._MAX_ROW_VALUES and header_raw and header_raw not in header_to_value:
                header_to_value[header_raw] = norm
            if len(serial_parts) < cls._MAX_SERIAL_FIELDS and header_raw:
                serial_parts.append(f"col {header_raw} val {norm}")
            if len(serial_token_set) < cls._MAX_ROW_TOKENS:
                for token in header_tokens:
                    serial_token_set.add(token)
                    if len(serial_token_set) >= cls._MAX_ROW_TOKENS:
                        break
                if len(serial_token_set) < cls._MAX_ROW_TOKENS:
                    for token in cls._tokenize_value(norm):
                        serial_token_set.add(token)
                        if len(serial_token_set) >= cls._MAX_ROW_TOKENS:
                            break
            if len(header_value_token_set) < cls._MAX_ROW_TOKENS:
                value_tokens = cls._tokenize_value(norm)
                for ht in header_tokens:
                    for vt in value_tokens:
                        header_value_token_set.add(f"{ht}:{vt}")
                        if len(header_value_token_set) >= cls._MAX_ROW_TOKENS:
                            break
                    if len(header_value_token_set) >= cls._MAX_ROW_TOKENS:
                        break

        numeric_ratio = float(numeric_count) / float(nonempty_count) if nonempty_count > 0 else 0.0
        avg_len = float(total_len) / float(nonempty_count) if nonempty_count > 0 else 0.0
        serial_text = " ; ".join(serial_parts)
        if len(serial_text) > cls._MAX_SERIAL_CHARS:
            serial_text = serial_text[: cls._MAX_SERIAL_CHARS]
        numeric_median = 0.0
        if numeric_values_float:
            numeric_median = float(np.median(np.asarray(numeric_values_float, dtype=np.float64)))
        return {
            "value_set": value_set,
            "token_set": token_set,
            "header_token_set": header_token_set,
            "header_value_token_set": header_value_token_set,
            "serial_token_set": serial_token_set,
            "serial_text": serial_text,
            "numeric_value_set": numeric_value_set,
            "header_to_value": header_to_value,
            "nonempty_count": nonempty_count,
            "numeric_ratio": numeric_ratio,
            "avg_len": avg_len,
            "numeric_median": numeric_median,
        }

    def _load_table_rows_full(self, table_name: str) -> List[Dict[str, object]]:
        cached = self._table_cache.get(table_name)
        if isinstance(cached, list):
            return cached

        table_path = self._resolve_table_path(table_name)
        if table_path == "":
            self._table_cache[table_name] = []
            return self._table_cache[table_name]

        rows: List[Dict[str, object]] = []
        with open(table_path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(self._build_row_stats(row))
        self._table_cache[table_name] = rows
        return rows

    def _load_table_rows_required(self, table_name: str) -> Dict[int, Dict[str, object]]:
        cached = self._table_cache.get(table_name)
        if isinstance(cached, dict):
            return cached

        required_rows = self._resolve_required_rows_for_table(table_name)
        if not required_rows:
            self._table_cache[table_name] = {}
            return self._table_cache[table_name]

        table_path = self._resolve_table_path(table_name)
        if table_path == "":
            self._table_cache[table_name] = {}
            return self._table_cache[table_name]

        target_count = len(required_rows)
        max_required = max(required_rows)
        selected: Dict[int, Dict[str, object]] = {}
        with open(table_path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                if row_idx in required_rows:
                    selected[int(row_idx)] = self._build_row_stats(row)
                    if len(selected) >= target_count and row_idx >= max_required:
                        break
                elif row_idx >= max_required and len(selected) >= target_count:
                    break
        self._table_cache[table_name] = selected
        return selected

    def _row_stats(self, table_name: str, row_idx: int) -> Optional[Dict[str, object]]:
        if self._row_stats_mode == "required":
            rows = self._load_table_rows_required(table_name)
            return rows.get(int(row_idx))
        # full mode: first try sparse rows collected from requested edges.
        sparse_rows = self._load_table_rows_required(table_name)
        cached = sparse_rows.get(int(row_idx))
        if cached is not None:
            return cached
        rows = self._load_table_rows_full(table_name)
        if row_idx < 0 or row_idx >= len(rows):
            return None
        return rows[row_idx]

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def _l1_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        mean_abs = float(np.mean(np.abs(vec_a - vec_b)))
        return 1.0 / (1.0 + mean_abs)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, float(value))))

    @classmethod
    def _set_jaccard(cls, values_a: Set[str], values_b: Set[str]) -> float:
        return float(JoinablePairFeatureStore._token_jaccard(list(values_a), list(values_b)))

    def _generated_feature_helpers(self) -> Dict[str, object]:
        return {
            "safe_ratio_float": JoinablePairFeatureStore._safe_ratio_float,
            "token_jaccard": JoinablePairFeatureStore._token_jaccard,
            "weighted_jaccard": self._weighted_jaccard,
            "normalized_edit_similarity": JoinablePairFeatureStore._normalized_edit_similarity,
            "char_ngram_jaccard": self._char_ngram_jaccard,
            "cosine_similarity": self._cosine_similarity,
            "l1_similarity": self._l1_similarity,
            "numeric_overlap_max": self._numeric_overlap_max,
            "clip01": self._clip01,
            "set_jaccard": self._set_jaccard,
        }

    def _compute_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        if self.feature_dim == 0:
            return np.zeros((0,), dtype=np.float32)

        row_a = self.node_id_to_row.get(int(src_id))
        row_b = self.node_id_to_row.get(int(dst_id))
        if row_a is None or row_b is None:
            return np.zeros((self.feature_dim,), dtype=np.float32)

        atom_values: Dict[str, float] = {name: 0.0 for name in self.feature_order}

        need_generated = self.generated_feature_registry is not None
        need_embedding = need_generated or any(name in self._feature_set for name in ("row_emb_cosine", "row_emb_l1_sim"))
        need_row_stats = any(
            name in self._feature_set
            for name in (
                "row_value_jaccard",
                "row_value_containment_max",
                "row_token_jaccard",
                "row_nonempty_ratio",
                "row_numeric_ratio_sim",
                "row_avg_len_ratio",
                "row_serial_token_jaccard",
                "row_serial_edit_similarity",
                "row_numeric_value_overlap",
                "row_header_jaccard",
                "row_header_value_exact_ratio",
                "row_header_token_jaccard",
            )
        ) or need_generated

        emb_a = None
        emb_b = None
        if need_embedding:
            emb_a = np.asarray(self._node_embeddings[int(src_id)], dtype=np.float32)
            emb_b = np.asarray(self._node_embeddings[int(dst_id)], dtype=np.float32)
            if "row_emb_cosine" in atom_values:
                atom_values["row_emb_cosine"] = self._cosine_similarity(emb_a, emb_b)
            if "row_emb_l1_sim" in atom_values:
                atom_values["row_emb_l1_sim"] = self._l1_similarity(emb_a, emb_b)

        stats_a = None
        stats_b = None
        if need_row_stats:
            table_a, row_idx_a = row_a
            table_b, row_idx_b = row_b
            stats_a = self._row_stats_with_table_fallback(table_a, row_idx_a)
            stats_b = self._row_stats_with_table_fallback(table_b, row_idx_b)
            if stats_a is None or stats_b is None:
                return np.zeros((self.feature_dim,), dtype=np.float32)

        if stats_a is not None and stats_b is not None and any(
            name in self._feature_set for name in ("row_value_jaccard", "row_value_containment_max", "row_token_jaccard")
        ):
            set_a = stats_a["value_set"]
            set_b = stats_b["value_set"]
            inter_size = len(set_a.intersection(set_b))
            union_size = len(set_a.union(set_b))
            value_jaccard = float(inter_size) / float(union_size) if union_size > 0 else 0.0
            containment_a = float(inter_size) / float(len(set_a)) if set_a else 0.0
            containment_b = float(inter_size) / float(len(set_b)) if set_b else 0.0
            token_jaccard = JoinablePairFeatureStore._token_jaccard(
                list(stats_a["token_set"]),
                list(stats_b["token_set"]),
            )
            if "row_value_jaccard" in atom_values:
                atom_values["row_value_jaccard"] = value_jaccard
            if "row_value_containment_max" in atom_values:
                atom_values["row_value_containment_max"] = max(containment_a, containment_b)
            if "row_token_jaccard" in atom_values:
                atom_values["row_token_jaccard"] = token_jaccard

        if stats_a is not None and stats_b is not None and any(
            name in self._feature_set for name in ("row_nonempty_ratio", "row_numeric_ratio_sim", "row_avg_len_ratio")
        ):
            nonempty_ratio = JoinablePairFeatureStore._safe_ratio_float(
                float(stats_a["nonempty_count"]),
                float(stats_b["nonempty_count"]),
            )
            numeric_ratio_sim = 1.0 - abs(float(stats_a["numeric_ratio"]) - float(stats_b["numeric_ratio"]))
            avg_len_ratio = JoinablePairFeatureStore._safe_ratio_float(
                float(stats_a["avg_len"]),
                float(stats_b["avg_len"]),
            )
            if "row_nonempty_ratio" in atom_values:
                atom_values["row_nonempty_ratio"] = nonempty_ratio
            if "row_numeric_ratio_sim" in atom_values:
                atom_values["row_numeric_ratio_sim"] = numeric_ratio_sim
            if "row_avg_len_ratio" in atom_values:
                atom_values["row_avg_len_ratio"] = avg_len_ratio

        if stats_a is not None and stats_b is not None and any(
            name in self._feature_set
            for name in ("row_serial_token_jaccard", "row_serial_edit_similarity", "row_numeric_value_overlap")
        ):
            serial_token_jaccard = JoinablePairFeatureStore._token_jaccard(
                list(stats_a["serial_token_set"]),
                list(stats_b["serial_token_set"]),
            )
            serial_edit_similarity = JoinablePairFeatureStore._normalized_edit_similarity(
                str(stats_a["serial_text"]),
                str(stats_b["serial_text"]),
            )
            numeric_overlap = self._numeric_overlap_max(
                set(stats_a["numeric_value_set"]),
                set(stats_b["numeric_value_set"]),
            )
            if "row_serial_token_jaccard" in atom_values:
                atom_values["row_serial_token_jaccard"] = serial_token_jaccard
            if "row_serial_edit_similarity" in atom_values:
                atom_values["row_serial_edit_similarity"] = serial_edit_similarity
            if "row_numeric_value_overlap" in atom_values:
                atom_values["row_numeric_value_overlap"] = numeric_overlap

        if stats_a is not None and stats_b is not None and any(
            name in self._feature_set
            for name in (
                "row_serial_char3_jaccard",
                "row_serial_char4_jaccard",
                "row_token_idf_jaccard",
                "row_numeric_rel_diff_sim",
            )
        ):
            if "row_serial_char3_jaccard" in atom_values:
                atom_values["row_serial_char3_jaccard"] = self._char_ngram_jaccard(
                    str(stats_a["serial_text"]),
                    str(stats_b["serial_text"]),
                    3,
                )
            if "row_serial_char4_jaccard" in atom_values:
                atom_values["row_serial_char4_jaccard"] = self._char_ngram_jaccard(
                    str(stats_a["serial_text"]),
                    str(stats_b["serial_text"]),
                    4,
                )
            if "row_token_idf_jaccard" in atom_values:
                atom_values["row_token_idf_jaccard"] = self._weighted_jaccard(
                    set(stats_a["token_set"]),
                    set(stats_b["token_set"]),
                )
            if "row_numeric_rel_diff_sim" in atom_values:
                numeric_set_a = set(stats_a["numeric_value_set"])
                numeric_set_b = set(stats_b["numeric_value_set"])
                if numeric_set_a and numeric_set_b:
                    med_a = float(stats_a.get("numeric_median", 0.0))
                    med_b = float(stats_b.get("numeric_median", 0.0))
                    denom = max(abs(med_a), abs(med_b), 1.0)
                    rel_diff = abs(med_a - med_b) / float(denom)
                    atom_values["row_numeric_rel_diff_sim"] = 1.0 / (1.0 + rel_diff)
                else:
                    atom_values["row_numeric_rel_diff_sim"] = 0.0

        if stats_a is not None and stats_b is not None and any(
            name in self._feature_set
            for name in ("row_header_jaccard", "row_header_value_exact_ratio", "row_header_token_jaccard")
        ):
            header_jaccard = JoinablePairFeatureStore._token_jaccard(
                list(stats_a["header_token_set"]),
                list(stats_b["header_token_set"]),
            )
            header_value_token_jaccard = JoinablePairFeatureStore._token_jaccard(
                list(stats_a["header_value_token_set"]),
                list(stats_b["header_value_token_set"]),
            )
            header_to_value_a: Dict[str, str] = stats_a["header_to_value"]
            header_to_value_b: Dict[str, str] = stats_b["header_to_value"]
            shared_headers = set(header_to_value_a.keys()).intersection(set(header_to_value_b.keys()))
            if shared_headers:
                exact_hits = sum(1 for key in shared_headers if header_to_value_a.get(key) == header_to_value_b.get(key))
                header_value_exact_ratio = float(exact_hits) / float(len(shared_headers))
            else:
                header_value_exact_ratio = 0.0
            if "row_header_jaccard" in atom_values:
                atom_values["row_header_jaccard"] = header_jaccard
            if "row_header_value_exact_ratio" in atom_values:
                atom_values["row_header_value_exact_ratio"] = header_value_exact_ratio
            if "row_header_token_jaccard" in atom_values:
                atom_values["row_header_token_jaccard"] = header_value_token_jaccard

        if self.generated_feature_registry is not None and stats_a is not None and stats_b is not None:
            ctx = {
                "stats_a": stats_a,
                "stats_b": stats_b,
                "emb_a": emb_a,
                "emb_b": emb_b,
                "helpers": self._generated_feature_helpers(),
            }
            generated_values = self.generated_feature_registry.compute(ctx)
            for name, value in generated_values.items():
                if name in atom_values:
                    atom_values[name] = float(value)

        feats = [float(atom_values.get(name, 0.0)) for name in self.feature_order]
        return np.asarray(feats, dtype=np.float32)

    def get_feature_hit_diagnostics(
        self,
        edges: List[Tuple[int, int]],
        *,
        sample_size: int = 2048,
        seed: int = 0,
    ) -> Dict[str, object]:
        diagnostics: Dict[str, object] = {
            "feature_dim": int(self.feature_dim),
            "sample_size": 0,
            "overall_nonzero_ratio": 0.0,
            "feature_nonzero_ratio": {},
        }
        edge_count = len(edges) if edges is not None else 0
        if self.feature_dim <= 0 or edge_count <= 0:
            return diagnostics

        count = min(int(sample_size), int(edge_count))
        if count <= 0:
            return diagnostics
        if count < int(edge_count):
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(int(edge_count), size=count, replace=False)
            sampled = [edges[int(i)] for i in idx.tolist()]
        else:
            sampled = [tuple(pair) for pair in edges]

        rows: List[np.ndarray] = [self.get_pair_features(int(src), int(dst)) for src, dst in sampled]
        if not rows:
            return diagnostics
        arr = np.stack(rows, axis=0)
        nonzero = np.abs(arr) > 1e-12
        per_feature = nonzero.mean(axis=0)
        overall = (nonzero.sum(axis=1) > 0).mean()
        diagnostics["sample_size"] = int(arr.shape[0])
        diagnostics["overall_nonzero_ratio"] = float(overall)
        diagnostics["feature_nonzero_ratio"] = {
            name: float(per_feature[idx]) for idx, name in enumerate(self.feature_order)
        }
        return diagnostics

    def get_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        key = self._canonical_pair(int(src_id), int(dst_id))
        if self._pair_cache_mode == "off":
            return self._compute_pair_features(key[0], key[1])

        cached = self._pair_cache.get(key)
        if cached is not None:
            self._pair_cache_hit += 1
            return cached
        self._pair_cache_miss += 1
        feats = self._compute_pair_features(key[0], key[1])
        self._pair_cache[key] = feats
        return feats

    @staticmethod
    def _edge_hash(edges: List[Tuple[int, int]]) -> str:
        hasher = hashlib.sha1()
        for src, dst in edges:
            s = int(src)
            d = int(dst)
            if s > d:
                s, d = d, s
            hasher.update(f"{s}:{d};".encode("utf-8"))
        return hasher.hexdigest()[:16]

    def _split_cache_path(self, *, split: str, edges: List[Tuple[int, int]]) -> str:
        edge_hash = self._edge_hash(edges)
        fname = (
            f"{split}_{self._split_cache_prefix}_n{int(len(edges))}_{edge_hash}.npy"
        )
        return os.path.join(self._pair_cache_root, fname)

    def _warm_pair_cache_from_matrix(self, *, edges: List[Tuple[int, int]], matrix: np.ndarray) -> None:
        if self._pair_cache_mode != "readwrite":
            return
        if matrix.ndim != 2 or matrix.shape[0] != len(edges):
            return
        for idx, (src, dst) in enumerate(edges):
            key = self._canonical_pair(int(src), int(dst))
            self._pair_cache[key] = np.asarray(matrix[idx], dtype=np.float32)

    def build_or_load_split_matrix(self, *, split: str, edges: List[Tuple[int, int]]) -> np.ndarray:
        if self.feature_dim == 0:
            return np.zeros((int(len(edges)), 0), dtype=np.float32)
        if len(edges) == 0:
            return np.zeros((0, int(self.feature_dim)), dtype=np.float32)
        # Register row references seen in this split so full mode can use sparse table loading.
        self._register_rows_from_edges(edges)

        cache_path = ""
        if self._pair_cache_mode == "readwrite":
            cache_path = self._split_cache_path(split=split, edges=edges)
            if os.path.isfile(cache_path):
                try:
                    arr = np.load(cache_path)
                    if arr.shape == (len(edges), int(self.feature_dim)):
                        self._split_cache_hit += 1
                        self._warm_pair_cache_from_matrix(edges=edges, matrix=arr)
                        return np.asarray(arr, dtype=np.float32)
                except Exception:
                    pass
            self._split_cache_miss += 1

        src = torch.tensor([int(u) for u, _ in edges], dtype=torch.long)
        dst = torch.tensor([int(v) for _, v in edges], dtype=torch.long)
        arr = self.build_batch_features(
            src,
            dst,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).detach().cpu().numpy()
        if self._pair_cache_mode == "readwrite" and cache_path:
            tmp_path = cache_path + ".tmp.npy"
            try:
                np.save(tmp_path, arr.astype(np.float32, copy=False))
                os.replace(tmp_path, cache_path)
            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        return arr.astype(np.float32, copy=False)

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "pair_cache_hit": int(self._pair_cache_hit),
            "pair_cache_miss": int(self._pair_cache_miss),
            "pair_cache_size": int(len(self._pair_cache)),
            "split_cache_hit": int(self._split_cache_hit),
            "split_cache_miss": int(self._split_cache_miss),
            "table_cache_tables": int(len(self._table_cache)),
        }

    def build_batch_features(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.feature_dim == 0:
            return torch.zeros((int(src_ids.numel()), 0), dtype=dtype, device=device)
        src = src_ids.detach().cpu().tolist()
        dst = dst_ids.detach().cpu().tolist()
        rows = [self.get_pair_features(int(u), int(v)) for u, v in zip(src, dst)]
        if not rows:
            return torch.zeros((0, self.feature_dim), dtype=dtype, device=device)
        arr = np.stack(rows, axis=0)
        return torch.tensor(arr, dtype=dtype, device=device)


class SchemaPairFeatureStore:
    """Compute and cache SM column-column pair features."""

    _DATASET_TABLE_ROOTS: Dict[str, str] = JoinablePairFeatureStore._DATASET_TABLE_ROOTS
    _NULL_LIKE: Set[str] = {"", "nan", "none", "null", "na", "n/a", "unknown", "-", "unknown_cell"}
    _ATOM_GROUPS: Dict[str, Tuple[str, ...]] = {
        "header_similarity": ("header_token_jaccard", "header_edit_similarity"),
        "value_stats": ("unique_ratio_sim", "missing_ratio_sim", "numeric_ratio_sim", "avg_len_ratio_sim"),
        "value_overlap": ("value_jaccard", "value_containment_max"),
    }
    _CANONICAL_ATOM_ORDER: Tuple[str, ...] = (
        "header_token_jaccard",
        "header_edit_similarity",
        "unique_ratio_sim",
        "missing_ratio_sim",
        "numeric_ratio_sim",
        "avg_len_ratio_sim",
        "value_jaccard",
        "value_containment_max",
    )
    _SUPPORTED_ATOMS: Set[str] = set(_CANONICAL_ATOM_ORDER)

    @classmethod
    def _resolve_requested_atoms(
        cls,
        feature_names: List[str],
        *,
        generated_feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        requested_tokens = [name.strip() for name in feature_names if name and name.strip()]
        requested_tokens = [name for name in requested_tokens if name != "none"]

        expanded_atoms: List[str] = []
        generated_atoms: List[str] = []
        unknown: List[str] = []
        generated_set = set(str(name).strip() for name in (generated_feature_names or []) if str(name).strip())
        for token in requested_tokens:
            if token in cls._ATOM_GROUPS:
                expanded_atoms.extend(list(cls._ATOM_GROUPS[token]))
            elif token in cls._SUPPORTED_ATOMS:
                expanded_atoms.append(token)
            elif token in generated_set:
                generated_atoms.append(token)
            else:
                unknown.append(token)

        if unknown:
            supported = sorted(set(cls._ATOM_GROUPS.keys()).union(cls._SUPPORTED_ATOMS).union(generated_set))
            raise ValueError(
                f"Unsupported SM pair features/atoms: {unknown}. Supported: {supported}"
            )

        atom_set = set(expanded_atoms)
        atom_order = [name for name in cls._CANONICAL_ATOM_ORDER if name in atom_set] + list(generated_atoms)
        return requested_tokens, atom_order

    def __init__(
        self,
        *,
        graph,
        dataset_name: str,
        feature_names: List[str],
        table_root_override: str = "",
        generated_feature_specs_path: str = "",
    ) -> None:
        self.generated_feature_specs_path = str(generated_feature_specs_path).strip()
        self.generated_feature_registry: Optional[GeneratedFeatureRegistry] = None
        generated_feature_names: List[str] = []
        if self.generated_feature_specs_path:
            self.generated_feature_registry = load_generated_feature_registry(
                self.generated_feature_specs_path,
                expected_task="schema_matching",
                expected_scope="column_pair",
            )
            generated_feature_names = list(self.generated_feature_registry.feature_names)

        requested_tokens, atom_order = self._resolve_requested_atoms(
            feature_names,
            generated_feature_names=generated_feature_names,
        )
        self.requested_feature_tokens = requested_tokens
        self.feature_names = requested_tokens
        self.feature_order = atom_order
        self._feature_set: Set[str] = set(self.feature_order)
        self.feature_dim = len(self.feature_order)

        self.table_root = JoinablePairFeatureStore._resolve_table_root(
            dataset_name=dataset_name,
            override=table_root_override,
        )
        self.node_id_to_column = JoinablePairFeatureStore._load_node_id_to_column(graph)
        self._table_cache: Dict[str, Dict[str, Dict[str, object]]] = {}
        self._pair_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._warned_missing_tables: Set[str] = set()

        logger.info(
            f"[SM-PairFeat] enabled={self.feature_names} resolved_atoms={self.feature_order} dim={self.feature_dim} "
            f"table_root={self.table_root} mapped_columns={len(self.node_id_to_column)} "
            f"generated_features={0 if self.generated_feature_registry is None else len(self.generated_feature_registry.feature_names)}"
        )

    @staticmethod
    def _canonical_pair(src_id: int, dst_id: int) -> Tuple[int, int]:
        if src_id <= dst_id:
            return src_id, dst_id
        return dst_id, src_id

    def _resolve_table_path(self, table_name: str) -> str:
        table_path = os.path.join(self.table_root, table_name)
        if not os.path.isfile(table_path):
            table_path_csv = table_path + ".csv"
            if os.path.isfile(table_path_csv):
                table_path = table_path_csv
        if os.path.isfile(table_path):
            return table_path
        if table_name not in self._warned_missing_tables:
            logger.warning(f"[SM-PairFeat] missing table file: {table_path}")
            self._warned_missing_tables.add(table_name)
        return ""

    def _load_table_column_stats(self, table_name: str) -> Dict[str, Dict[str, object]]:
        cached = self._table_cache.get(table_name)
        if cached is not None:
            return cached

        table_path = self._resolve_table_path(table_name)
        if table_path == "":
            self._table_cache[table_name] = {}
            return self._table_cache[table_name]

        with open(table_path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            values_map: Dict[str, List[str]] = {name: [] for name in headers}
            missing_map: Dict[str, int] = {name: 0 for name in headers}
            num_rows = 0
            for row in reader:
                num_rows += 1
                for name in headers:
                    raw = row.get(name, "")
                    raw_str = "" if raw is None else str(raw).strip()
                    if raw_str == "":
                        missing_map[name] += 1
                        continue
                    norm = JoinablePairFeatureStore._normalize_cell_value(raw_str)
                    if norm in self._NULL_LIKE:
                        missing_map[name] += 1
                        continue
                    values_map[name].append(norm)

        alias_map: Dict[str, Dict[str, object]] = {}
        for name in headers:
            values = values_map[name]
            value_set = set(values)
            unique_ratio = float(len(value_set)) / float(len(values)) if values else 0.0
            missing_ratio = float(missing_map[name]) / float(num_rows) if num_rows > 0 else 1.0
            numeric_ratio = JoinablePairFeatureStore._numeric_ratio(values)
            avg_len = JoinablePairFeatureStore._avg_str_len(values)
            header_tokens = set(JoinablePairFeatureStore._tokenize_column_name(name))
            stats = {
                "value_set": value_set,
                "unique_ratio": unique_ratio,
                "missing_ratio": missing_ratio,
                "numeric_ratio": numeric_ratio,
                "avg_len": avg_len,
                "header_tokens": header_tokens,
                "header_text": name.strip().lower(),
            }
            alias_map[name] = stats
            rstrip_name = name.rstrip()
            strip_name = name.strip()
            if rstrip_name not in alias_map:
                alias_map[rstrip_name] = stats
            if strip_name not in alias_map:
                alias_map[strip_name] = stats

        self._table_cache[table_name] = alias_map
        return alias_map

    def _column_stats(self, table_name: str, column_name: str) -> Optional[Dict[str, object]]:
        col_map = self._load_table_column_stats(table_name)
        if column_name in col_map:
            return col_map[column_name]
        if column_name.rstrip() in col_map:
            return col_map[column_name.rstrip()]
        if column_name.strip() in col_map:
            return col_map[column_name.strip()]
        return None

    @staticmethod
    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, float(value))))

    @classmethod
    def _set_jaccard(cls, values_a: Set[str], values_b: Set[str]) -> float:
        return float(JoinablePairFeatureStore._token_jaccard(list(values_a), list(values_b)))

    def _generated_feature_helpers(self) -> Dict[str, object]:
        return {
            "safe_ratio_float": JoinablePairFeatureStore._safe_ratio_float,
            "token_jaccard": JoinablePairFeatureStore._token_jaccard,
            "normalized_edit_similarity": JoinablePairFeatureStore._normalized_edit_similarity,
            "clip01": self._clip01,
            "set_jaccard": self._set_jaccard,
        }

    def _compute_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        if self.feature_dim == 0:
            return np.zeros((0,), dtype=np.float32)

        col_a = self.node_id_to_column.get(int(src_id))
        col_b = self.node_id_to_column.get(int(dst_id))
        if col_a is None or col_b is None:
            return np.zeros((self.feature_dim,), dtype=np.float32)

        table_a, name_a = col_a
        table_b, name_b = col_b
        stats_a = self._column_stats(table_a, name_a)
        stats_b = self._column_stats(table_b, name_b)
        if stats_a is None or stats_b is None:
            return np.zeros((self.feature_dim,), dtype=np.float32)

        atom_values: Dict[str, float] = {name: 0.0 for name in self.feature_order}
        if "header_token_jaccard" in self._feature_set or "header_edit_similarity" in self._feature_set:
            token_jaccard = JoinablePairFeatureStore._token_jaccard(
                list(stats_a["header_tokens"]),
                list(stats_b["header_tokens"]),
            )
            edit_sim = JoinablePairFeatureStore._normalized_edit_similarity(
                str(stats_a["header_text"]),
                str(stats_b["header_text"]),
            )
            if "header_token_jaccard" in atom_values:
                atom_values["header_token_jaccard"] = float(token_jaccard)
            if "header_edit_similarity" in atom_values:
                atom_values["header_edit_similarity"] = float(edit_sim)

        if (
            "unique_ratio_sim" in self._feature_set
            or "missing_ratio_sim" in self._feature_set
            or "numeric_ratio_sim" in self._feature_set
            or "avg_len_ratio_sim" in self._feature_set
        ):
            unique_ratio_sim = JoinablePairFeatureStore._safe_ratio_float(
                float(stats_a["unique_ratio"]),
                float(stats_b["unique_ratio"]),
            )
            missing_ratio_sim = 1.0 - abs(float(stats_a["missing_ratio"]) - float(stats_b["missing_ratio"]))
            numeric_ratio_sim = 1.0 - abs(float(stats_a["numeric_ratio"]) - float(stats_b["numeric_ratio"]))
            avg_len_ratio_sim = JoinablePairFeatureStore._safe_ratio_float(
                float(stats_a["avg_len"]),
                float(stats_b["avg_len"]),
            )
            if "unique_ratio_sim" in atom_values:
                atom_values["unique_ratio_sim"] = float(unique_ratio_sim)
            if "missing_ratio_sim" in atom_values:
                atom_values["missing_ratio_sim"] = float(missing_ratio_sim)
            if "numeric_ratio_sim" in atom_values:
                atom_values["numeric_ratio_sim"] = float(numeric_ratio_sim)
            if "avg_len_ratio_sim" in atom_values:
                atom_values["avg_len_ratio_sim"] = float(avg_len_ratio_sim)

        if "value_jaccard" in self._feature_set or "value_containment_max" in self._feature_set:
            set_a = stats_a["value_set"]
            set_b = stats_b["value_set"]
            inter_size = len(set_a.intersection(set_b))
            union_size = len(set_a.union(set_b))
            jaccard = float(inter_size) / float(union_size) if union_size > 0 else 0.0
            containment_a = float(inter_size) / float(len(set_a)) if set_a else 0.0
            containment_b = float(inter_size) / float(len(set_b)) if set_b else 0.0
            containment_max = max(containment_a, containment_b)
            if "value_jaccard" in atom_values:
                atom_values["value_jaccard"] = float(jaccard)
            if "value_containment_max" in atom_values:
                atom_values["value_containment_max"] = float(containment_max)

        if self.generated_feature_registry is not None:
            ctx = {
                "stats_a": stats_a,
                "stats_b": stats_b,
                "helpers": self._generated_feature_helpers(),
            }
            generated_values = self.generated_feature_registry.compute(ctx)
            for name, value in generated_values.items():
                if name in atom_values:
                    atom_values[name] = float(value)

        feats = [float(atom_values[name]) for name in self.feature_order]
        if not feats:
            return np.zeros((self.feature_dim,), dtype=np.float32)
        return np.asarray(feats, dtype=np.float32)

    def get_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        key = self._canonical_pair(int(src_id), int(dst_id))
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        feats = self._compute_pair_features(key[0], key[1])
        self._pair_cache[key] = feats
        return feats

    def build_batch_features(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.feature_dim == 0:
            return torch.zeros((int(src_ids.numel()), 0), dtype=dtype, device=device)
        src = src_ids.detach().cpu().tolist()
        dst = dst_ids.detach().cpu().tolist()
        rows = [self.get_pair_features(int(u), int(v)) for u, v in zip(src, dst)]
        if not rows:
            return torch.zeros((0, self.feature_dim), dtype=dtype, device=device)
        arr = np.stack(rows, axis=0)
        return torch.tensor(arr, dtype=dtype, device=device)


class UnionPairFeatureStore:
    """Compute and cache UTS table-table pair features."""

    _DATASET_TABLE_ROOTS: Dict[str, str] = JoinablePairFeatureStore._DATASET_TABLE_ROOTS
    _NULL_LIKE: Set[str] = {"", "nan", "none", "null", "na", "n/a", "unknown", "-", "unknown_cell"}
    _ATOM_GROUPS: Dict[str, Tuple[str, ...]] = {
        "column_overlap": (
            "col_overlap_a2b_mean",
            "col_overlap_b2a_mean",
            "col_overlap_a2b_cov",
            "col_overlap_b2a_cov",
        ),
        "header_jaccard": ("header_jaccard",),
        "table_size_ratio": ("col_count_ratio", "row_count_ratio"),
    }
    _CANONICAL_ATOM_ORDER: Tuple[str, ...] = (
        "col_overlap_a2b_mean",
        "col_overlap_b2a_mean",
        "col_overlap_a2b_cov",
        "col_overlap_b2a_cov",
        "header_jaccard",
        "col_count_ratio",
        "row_count_ratio",
    )
    _SUPPORTED_ATOMS: Set[str] = set(_CANONICAL_ATOM_ORDER)

    @classmethod
    def _resolve_requested_atoms(
        cls,
        feature_names: List[str],
        *,
        generated_feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        requested_tokens = [name.strip() for name in feature_names if name and name.strip()]
        requested_tokens = [name for name in requested_tokens if name != "none"]

        expanded_atoms: List[str] = []
        generated_atoms: List[str] = []
        unknown: List[str] = []
        generated_set = set(str(name).strip() for name in (generated_feature_names or []) if str(name).strip())
        for token in requested_tokens:
            if token in cls._ATOM_GROUPS:
                expanded_atoms.extend(list(cls._ATOM_GROUPS[token]))
            elif token in cls._SUPPORTED_ATOMS:
                expanded_atoms.append(token)
            elif token in generated_set:
                generated_atoms.append(token)
            else:
                unknown.append(token)

        if unknown:
            supported = sorted(set(cls._ATOM_GROUPS.keys()).union(cls._SUPPORTED_ATOMS).union(generated_set))
            raise ValueError(
                f"Unsupported UTS pair features/atoms: {unknown}. Supported: {supported}"
            )

        atom_set = set(expanded_atoms)
        atom_order = [name for name in cls._CANONICAL_ATOM_ORDER if name in atom_set] + list(generated_atoms)
        return requested_tokens, atom_order

    def __init__(
        self,
        *,
        graph,
        dataset_name: str,
        feature_names: List[str],
        table_root_override: str = "",
        overlap_coverage_threshold: float = 0.1,
        max_unique_values_per_column: int = 1024,
        generated_feature_specs_path: str = "",
    ) -> None:
        self.generated_feature_specs_path = str(generated_feature_specs_path).strip()
        self.generated_feature_registry: Optional[GeneratedFeatureRegistry] = None
        generated_feature_names: List[str] = []
        if self.generated_feature_specs_path:
            self.generated_feature_registry = load_generated_feature_registry(
                self.generated_feature_specs_path,
                expected_task="union_table_search",
                expected_scope="table_pair",
            )
            generated_feature_names = list(self.generated_feature_registry.feature_names)

        requested_tokens, atom_order = self._resolve_requested_atoms(
            feature_names,
            generated_feature_names=generated_feature_names,
        )
        self.requested_feature_tokens = requested_tokens
        self.feature_names = requested_tokens
        self.feature_order = atom_order
        self._feature_set: Set[str] = set(self.feature_order)
        self.feature_dim = len(self.feature_order)

        self.coverage_threshold = float(overlap_coverage_threshold)
        self.max_unique_values_per_column = int(max_unique_values_per_column)
        self.table_root = self._resolve_table_root(dataset_name=dataset_name, override=table_root_override)
        self.node_id_to_table = self._load_node_id_to_table(graph)
        self._table_cache: Dict[str, Dict[str, object]] = {}
        self._pair_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._warned_missing_tables: Set[str] = set()

        logger.info(
            f"[UTS-PairFeat] enabled={self.feature_names} resolved_atoms={self.feature_order} dim={self.feature_dim} "
            f"table_root={self.table_root} mapped_tables={len(self.node_id_to_table)} "
            f"coverage_threshold={self.coverage_threshold:.3f} "
            f"generated_features={0 if self.generated_feature_registry is None else len(self.generated_feature_registry.feature_names)}"
        )

    @staticmethod
    def _resolve_table_root(*, dataset_name: str, override: str) -> str:
        if override:
            if not os.path.isdir(override):
                raise FileNotFoundError(f"--uts_table_root does not exist: {override}")
            return override
        if dataset_name not in UnionPairFeatureStore._DATASET_TABLE_ROOTS:
            raise ValueError(
                f"Unknown dataset_name={dataset_name} for UTS pair features. "
                f"Expected one of: {sorted(UnionPairFeatureStore._DATASET_TABLE_ROOTS.keys())}"
            )
        table_root = UnionPairFeatureStore._DATASET_TABLE_ROOTS[dataset_name]
        if not os.path.isdir(table_root):
            raise FileNotFoundError(f"Auto-resolved table root does not exist: {table_root}")
        return table_root

    @staticmethod
    def _load_node_id_to_table(graph) -> Dict[int, str]:
        mapping_path = os.path.join(graph.data_dir, "node_id_mapping.json")
        with open(mapping_path, "r") as handle:
            node_map = json.load(handle)
        raw_table_map = node_map.get("table", {})

        out: Dict[int, str] = {}
        for table_name, node_id in raw_table_map.items():
            out[int(node_id)] = str(table_name)
        return out

    @staticmethod
    def _canonical_pair(src_id: int, dst_id: int) -> Tuple[int, int]:
        if src_id <= dst_id:
            return src_id, dst_id
        return dst_id, src_id

    def _resolve_table_path(self, table_name: str) -> str:
        candidates: List[str] = [os.path.join(self.table_root, table_name)]
        if table_name.endswith(".csv"):
            candidates.append(os.path.join(self.table_root, table_name[:-4]))
        else:
            candidates.append(os.path.join(self.table_root, table_name + ".csv"))

        for path in candidates:
            if os.path.isfile(path):
                return path

        if table_name not in self._warned_missing_tables:
            logger.warning(
                f"[UTS-PairFeat] missing table file for table={table_name} candidates={candidates}"
            )
            self._warned_missing_tables.add(table_name)
        return ""

    def _load_table_stats(self, table_name: str) -> Dict[str, object]:
        cached = self._table_cache.get(table_name)
        if cached is not None:
            return cached

        table_path = self._resolve_table_path(table_name)
        if table_path == "":
            empty = {
                "column_value_sets": [],
                "header_tokens": set(),
                "num_rows": 0,
                "num_cols": 0,
            }
            self._table_cache[table_name] = empty
            return empty

        with open(table_path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            column_sets: Dict[str, Set[str]] = {name: set() for name in headers}
            num_rows = 0
            for row in reader:
                num_rows += 1
                for name in headers:
                    raw = row.get(name, "")
                    if raw is None:
                        continue
                    raw_str = str(raw).strip()
                    if raw_str == "":
                        continue
                    norm = JoinablePairFeatureStore._normalize_cell_value(raw_str)
                    if norm in self._NULL_LIKE:
                        continue
                    col_set = column_sets[name]
                    if len(col_set) < self.max_unique_values_per_column:
                        col_set.add(norm)

        header_tokens: Set[str] = set()
        for name in headers:
            header_tokens.update(JoinablePairFeatureStore._tokenize_column_name(name))

        stats = {
            "column_value_sets": [column_sets[name] for name in headers],
            "header_tokens": header_tokens,
            "num_rows": int(num_rows),
            "num_cols": int(len(headers)),
        }
        self._table_cache[table_name] = stats
        return stats

    @staticmethod
    def _overlap_score(values_a: Set[str], values_b: Set[str]) -> float:
        if not values_a or not values_b:
            return 0.0
        inter_size = len(values_a.intersection(values_b))
        if inter_size == 0:
            return 0.0
        union_size = len(values_a) + len(values_b) - inter_size
        jaccard = float(inter_size) / float(union_size) if union_size > 0 else 0.0
        containment_a = float(inter_size) / float(len(values_a)) if values_a else 0.0
        containment_b = float(inter_size) / float(len(values_b)) if values_b else 0.0
        return max(jaccard, containment_a, containment_b)

    def _directional_overlap_stats(
        self,
        src_cols: List[Set[str]],
        dst_cols: List[Set[str]],
    ) -> Tuple[float, float]:
        if not src_cols or not dst_cols:
            return 0.0, 0.0

        best_scores: List[float] = []
        for src_values in src_cols:
            best = 0.0
            if src_values:
                for dst_values in dst_cols:
                    score = self._overlap_score(src_values, dst_values)
                    if score > best:
                        best = score
            best_scores.append(best)

        if not best_scores:
            return 0.0, 0.0
        mean_score = float(np.mean(best_scores))
        coverage = float(sum(score >= self.coverage_threshold for score in best_scores)) / float(
            len(best_scores)
        )
        return mean_score, coverage

    @staticmethod
    def _safe_ratio(a: int, b: int) -> float:
        a = int(a)
        b = int(b)
        if a <= 0 or b <= 0:
            return 0.0
        lo = min(a, b)
        hi = max(a, b)
        return float(lo) / float(hi)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, float(value))))

    @classmethod
    def _set_jaccard(cls, values_a: Set[str], values_b: Set[str]) -> float:
        return float(JoinablePairFeatureStore._token_jaccard(list(values_a), list(values_b)))

    def _generated_feature_helpers(self) -> Dict[str, object]:
        return {
            "safe_ratio_float": JoinablePairFeatureStore._safe_ratio_float,
            "token_jaccard": JoinablePairFeatureStore._token_jaccard,
            "normalized_edit_similarity": JoinablePairFeatureStore._normalized_edit_similarity,
            "clip01": self._clip01,
            "set_jaccard": self._set_jaccard,
        }

    def _compute_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        if self.feature_dim == 0:
            return np.zeros((0,), dtype=np.float32)

        table_a = self.node_id_to_table.get(int(src_id))
        table_b = self.node_id_to_table.get(int(dst_id))
        if table_a is None or table_b is None:
            return np.zeros((self.feature_dim,), dtype=np.float32)

        stats_a = self._load_table_stats(table_a)
        stats_b = self._load_table_stats(table_b)

        cols_a = stats_a["column_value_sets"]
        cols_b = stats_b["column_value_sets"]
        header_tokens_a = stats_a["header_tokens"]
        header_tokens_b = stats_b["header_tokens"]
        num_rows_a = int(stats_a["num_rows"])
        num_rows_b = int(stats_b["num_rows"])
        num_cols_a = int(stats_a["num_cols"])
        num_cols_b = int(stats_b["num_cols"])

        atom_values: Dict[str, float] = {name: 0.0 for name in self.feature_order}
        if (
            "col_overlap_a2b_mean" in self._feature_set
            or "col_overlap_b2a_mean" in self._feature_set
            or "col_overlap_a2b_cov" in self._feature_set
            or "col_overlap_b2a_cov" in self._feature_set
        ):
            a2b_mean, a2b_cov = self._directional_overlap_stats(cols_a, cols_b)
            b2a_mean, b2a_cov = self._directional_overlap_stats(cols_b, cols_a)
            if "col_overlap_a2b_mean" in atom_values:
                atom_values["col_overlap_a2b_mean"] = float(a2b_mean)
            if "col_overlap_b2a_mean" in atom_values:
                atom_values["col_overlap_b2a_mean"] = float(b2a_mean)
            if "col_overlap_a2b_cov" in atom_values:
                atom_values["col_overlap_a2b_cov"] = float(a2b_cov)
            if "col_overlap_b2a_cov" in atom_values:
                atom_values["col_overlap_b2a_cov"] = float(b2a_cov)

        if "header_jaccard" in self._feature_set:
            if not header_tokens_a or not header_tokens_b:
                atom_values["header_jaccard"] = 0.0
            else:
                inter = len(header_tokens_a.intersection(header_tokens_b))
                union = len(header_tokens_a.union(header_tokens_b))
                atom_values["header_jaccard"] = float(inter) / float(union) if union > 0 else 0.0

        if "col_count_ratio" in self._feature_set:
            atom_values["col_count_ratio"] = float(self._safe_ratio(num_cols_a, num_cols_b))
        if "row_count_ratio" in self._feature_set:
            atom_values["row_count_ratio"] = float(self._safe_ratio(num_rows_a, num_rows_b))

        if self.generated_feature_registry is not None:
            stats_ctx_a = {
                "column_value_sets": cols_a,
                "header_tokens": header_tokens_a,
                "num_rows": num_rows_a,
                "num_cols": num_cols_a,
            }
            stats_ctx_b = {
                "column_value_sets": cols_b,
                "header_tokens": header_tokens_b,
                "num_rows": num_rows_b,
                "num_cols": num_cols_b,
            }
            ctx = {
                "stats_a": stats_ctx_a,
                "stats_b": stats_ctx_b,
                "helpers": self._generated_feature_helpers(),
            }
            generated_values = self.generated_feature_registry.compute(ctx)
            for name, value in generated_values.items():
                if name in atom_values:
                    atom_values[name] = float(value)

        feats = [float(atom_values[name]) for name in self.feature_order]
        if not feats:
            return np.zeros((self.feature_dim,), dtype=np.float32)
        return np.asarray(feats, dtype=np.float32)

    def get_pair_features(self, src_id: int, dst_id: int) -> np.ndarray:
        key = self._canonical_pair(int(src_id), int(dst_id))
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        feats = self._compute_pair_features(key[0], key[1])
        self._pair_cache[key] = feats
        return feats

    def build_batch_features(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.feature_dim == 0:
            return torch.zeros((int(src_ids.numel()), 0), dtype=dtype, device=device)
        src = src_ids.detach().cpu().tolist()
        dst = dst_ids.detach().cpu().tolist()
        rows = [self.get_pair_features(int(u), int(v)) for u, v in zip(src, dst)]
        if not rows:
            return torch.zeros((0, self.feature_dim), dtype=dtype, device=device)
        arr = np.stack(rows, axis=0)
        return torch.tensor(arr, dtype=dtype, device=device)
