from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from runtime_profile import profile_phase, record_profile_event


logger = logging.getLogger(__name__)


class HierGraph:
    """Graph loader for 0308_base using binary metadata caches."""

    CACHE_VERSION = 1
    DEFAULT_CACHE_FALLBACK_NAMESPACES = ("0308_base_em", "0308_base")

    def __init__(self, data_dir: str, *, cache_namespace: str = "0308_base", include_cell_edges: bool = False):
        self.data_dir = data_dir
        self.cache_namespace = cache_namespace
        self.include_cell_edges = bool(include_cell_edges)
        self.cache_dir = os.path.join(self.data_dir, f".{self.cache_namespace}_cache")

        self.edge_types = [
            "token_cell",
            "cell_row",
            "cell_column",
            "row_table",
            "column_table",
            "entity_matching",
            "joinable_table_search",
            "schema_matching",
            "union_table_search",
        ]
        self.target_classes = ["entity_matching", "joinable_table_search", "schema_matching", "union_table_search"]

        self.edge_features_map: Dict[str, int] = {}
        self.edge_type_to_id = {name: idx for idx, name in enumerate(self.edge_types)}
        self.edge_id_to_type = {idx: name for name, idx in self.edge_type_to_id.items()}

        self.edge_labels: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.edge_nodes: np.ndarray = np.empty((0, 4), dtype=np.int32)
        self.edge_sizes: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.edge_type_ids: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.edge_types_data = self.edge_type_ids
        self.train_mask: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.val_mask: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.test_mask: np.ndarray = np.empty((0,), dtype=np.uint8)

        self.edge_index: Optional[torch.Tensor] = None
        self.edge_features: Optional[torch.Tensor] = None
        self.num_nodes = 0
        self.num_edges = 0

        self._load_or_build_cache()

    def _cache_dir_for_namespace(self, namespace: str) -> str:
        return os.path.join(self.data_dir, f".{namespace}_cache")

    def _cache_fallback_namespaces(self) -> List[str]:
        raw = os.environ.get("GRAPH_CACHE_FALLBACK_NAMESPACES", "")
        if raw.strip():
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
        else:
            tokens = list(self.DEFAULT_CACHE_FALLBACK_NAMESPACES)
        # keep order, deduplicate, and avoid current namespace duplication
        seen = {self.cache_namespace}
        ordered: List[str] = []
        for ns in tokens:
            if ns in seen:
                continue
            seen.add(ns)
            ordered.append(ns)
        return ordered

    def _cache_ready_for_dir(self, cache_dir: str) -> bool:
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.isfile(meta_path):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception:
            return False

        expected = [
            os.path.join(cache_dir, "edge_labels.npy"),
            os.path.join(cache_dir, "edge_nodes.npy"),
            os.path.join(cache_dir, "edge_sizes.npy"),
            os.path.join(cache_dir, "edge_type_ids.npy"),
            os.path.join(cache_dir, "train_mask.npy"),
            os.path.join(cache_dir, "val_mask.npy"),
            os.path.join(cache_dir, "test_mask.npy"),
            os.path.join(cache_dir, "processed_edge_index.npy"),
            os.path.join(cache_dir, "processed_edge_attr.npy"),
        ]
        if meta.get("cache_version") != self.CACHE_VERSION:
            return False
        if bool(meta.get("include_cell_edges")) != self.include_cell_edges:
            return False
        if not all(os.path.isfile(path) for path in expected):
            return False
        return True

    def _find_reusable_cache_dir(self) -> Optional[Tuple[str, str]]:
        for namespace in self._cache_fallback_namespaces():
            candidate = self._cache_dir_for_namespace(namespace)
            if self._cache_ready_for_dir(candidate):
                return namespace, candidate
        return None

    def _meta_path(self) -> str:
        return os.path.join(self.cache_dir, "meta.json")

    def _raw_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{name}.npy")

    def _processed_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"processed_{name}.npy")

    def _count_lines(self, path: str) -> int:
        count = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for count, _ in enumerate(handle, start=1):
                pass
        return count

    def _load_node_count(self) -> int:
        path = os.path.join(self.data_dir, "node_embeddings.npy")
        arr = np.load(path, mmap_mode="r")
        return int(arr.shape[0])

    def _load_or_build_cache(self) -> None:
        with profile_phase(
            "graph.cache.ensure",
            {
                "data_dir": self.data_dir,
                "cache_dir": self.cache_dir,
                "include_cell_edges": self.include_cell_edges,
            },
        ):
            os.makedirs(self.cache_dir, exist_ok=True)
            if not self._cache_ready():
                reusable = self._find_reusable_cache_dir()
                if reusable is not None:
                    namespace, cache_dir = reusable
                    logger.info(
                        "[GraphCache] Reusing namespace '%s' for cache loading: %s",
                        namespace,
                        cache_dir,
                    )
                    self.cache_dir = cache_dir
                else:
                    self._build_cache()
            self._load_cache()

    def _cache_ready(self) -> bool:
        return self._cache_ready_for_dir(self.cache_dir)

    def _build_cache(self) -> None:
        with profile_phase(
            "graph.cache.build",
            {
                "data_dir": self.data_dir,
                "cache_dir": self.cache_dir,
                "include_cell_edges": self.include_cell_edges,
            },
        ):
            self._parse_raw_metadata_to_binary()
            self._build_processed_graph_cache()

    def _parse_raw_metadata_to_binary(self) -> None:
        edge_labels_txt = os.path.join(self.data_dir, "edge_labels.txt")
        edge_lists_txt = os.path.join(self.data_dir, "edge_lists.txt")
        edge_type_txt = os.path.join(self.data_dir, "edge_type.txt")
        train_mask_txt = os.path.join(self.data_dir, "train_mask.txt")
        val_mask_txt = os.path.join(self.data_dir, "validate_mask.txt")
        test_mask_txt = os.path.join(self.data_dir, "test_mask.txt")
        edge_embedding_map_json = os.path.join(self.data_dir, "edge_embedding_map.json")

        with profile_phase("graph.cache.count_edges", {"path": edge_labels_txt}):
            num_edges = int(self._count_lines(edge_labels_txt))

        edge_labels = np.lib.format.open_memmap(self._raw_path("edge_labels"), mode="w+", dtype=np.uint8, shape=(num_edges,))
        edge_nodes = np.lib.format.open_memmap(self._raw_path("edge_nodes"), mode="w+", dtype=np.int32, shape=(num_edges, 4))
        edge_nodes[:] = -1
        edge_sizes = np.lib.format.open_memmap(self._raw_path("edge_sizes"), mode="w+", dtype=np.uint8, shape=(num_edges,))
        edge_type_ids = np.lib.format.open_memmap(self._raw_path("edge_type_ids"), mode="w+", dtype=np.uint8, shape=(num_edges,))
        train_mask = np.lib.format.open_memmap(self._raw_path("train_mask"), mode="w+", dtype=np.uint8, shape=(num_edges,))
        val_mask = np.lib.format.open_memmap(self._raw_path("val_mask"), mode="w+", dtype=np.uint8, shape=(num_edges,))
        test_mask = np.lib.format.open_memmap(self._raw_path("test_mask"), mode="w+", dtype=np.uint8, shape=(num_edges,))

        with profile_phase("graph.cache.load.edge_labels", {"path": edge_labels_txt}):
            with open(edge_labels_txt, "r", encoding="utf-8", errors="ignore") as handle:
                for idx, line in enumerate(handle):
                    edge_labels[idx] = np.uint8(int(line.strip()))

        with profile_phase("graph.cache.load.edge_lists", {"path": edge_lists_txt}):
            with open(edge_lists_txt, "r", encoding="utf-8", errors="ignore") as handle:
                for idx, line in enumerate(handle):
                    parts = [int(value) for value in line.rstrip("\n").split("\t") if value]
                    size = min(len(parts), 4)
                    edge_sizes[idx] = np.uint8(size)
                    if size > 0:
                        edge_nodes[idx, :size] = np.asarray(parts[:size], dtype=np.int32)

        with profile_phase("graph.cache.load.edge_types", {"path": edge_type_txt}):
            with open(edge_type_txt, "r", encoding="utf-8", errors="ignore") as handle:
                for idx, line in enumerate(handle):
                    edge_type_ids[idx] = np.uint8(self.edge_type_to_id[line.strip()])

        def _load_mask(src: str, dst: np.memmap, tag: str) -> None:
            with profile_phase(tag, {"path": src}):
                with open(src, "r", encoding="utf-8", errors="ignore") as handle:
                    count = 0
                    for count, line in enumerate(handle, start=1):
                        if count > num_edges:
                            break
                        dst[count - 1] = np.uint8(int(line.strip()))
                    if count < num_edges:
                        dst[count:num_edges] = 0

        _load_mask(train_mask_txt, train_mask, "graph.cache.load.train_mask")
        _load_mask(val_mask_txt, val_mask, "graph.cache.load.val_mask")
        _load_mask(test_mask_txt, test_mask, "graph.cache.load.test_mask")

        with open(edge_embedding_map_json, "r", encoding="utf-8") as handle:
            self.edge_features_map = json.load(handle)

        num_nodes = self._load_node_count()
        meta = {
            "cache_version": self.CACHE_VERSION,
            "include_cell_edges": self.include_cell_edges,
            "num_edges_raw": num_edges,
            "num_nodes": num_nodes,
            "edge_features_map": self.edge_features_map,
        }
        with open(self._meta_path(), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

        record_profile_event(
            "graph_cache_raw_ready",
            {
                "data_dir": self.data_dir,
                "cache_dir": self.cache_dir,
                "num_edges_raw": num_edges,
                "num_nodes": num_nodes,
            },
        )

    def _build_processed_graph_cache(self) -> None:
        edge_labels = np.load(self._raw_path("edge_labels"), mmap_mode="r")
        edge_nodes = np.load(self._raw_path("edge_nodes"), mmap_mode="r")
        edge_sizes = np.load(self._raw_path("edge_sizes"), mmap_mode="r")
        edge_type_ids = np.load(self._raw_path("edge_type_ids"), mmap_mode="r")
        val_mask = np.load(self._raw_path("val_mask"), mmap_mode="r")
        test_mask = np.load(self._raw_path("test_mask"), mmap_mode="r")

        excluded_names = set(self.target_classes)
        excluded_names.add("token_cell")
        if not self.include_cell_edges:
            excluded_names.update({"cell_row", "cell_column"})
        excluded_ids = np.array(sorted(self.edge_type_to_id[name] for name in excluded_names), dtype=np.uint8)

        with profile_phase("graph.cache.count_processed_edges", {"data_dir": self.data_dir}):
            keep = (edge_labels == 1) & (val_mask == 0) & (test_mask == 0)
            keep &= ~np.isin(edge_type_ids, excluded_ids)
            count_size2 = int(np.count_nonzero(keep & (edge_sizes == 2)))
            count_size4 = int(np.count_nonzero(keep & (edge_sizes == 4)))
            num_processed_edges = count_size2 + count_size4 * 6

        edge_index = np.lib.format.open_memmap(
            self._processed_path("edge_index"),
            mode="w+",
            dtype=np.int64,
            shape=(2, num_processed_edges),
        )
        edge_attr = np.lib.format.open_memmap(
            self._processed_path("edge_attr"),
            mode="w+",
            dtype=np.float32,
            shape=(num_processed_edges,),
        )

        cursor = 0
        chunk_size = 1_000_000
        pair_indices = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            dtype=np.int64,
        )
        with profile_phase(
            "graph.cache.build_processed_graph",
            {"data_dir": self.data_dir, "num_processed_edges": num_processed_edges},
        ):
            for start in range(0, edge_labels.shape[0], chunk_size):
                end = min(start + chunk_size, edge_labels.shape[0])
                labels_chunk = edge_labels[start:end]
                sizes_chunk = edge_sizes[start:end]
                type_chunk = edge_type_ids[start:end]
                val_chunk = val_mask[start:end]
                test_chunk = test_mask[start:end]
                nodes_chunk = edge_nodes[start:end]

                keep_chunk = (labels_chunk == 1) & (val_chunk == 0) & (test_chunk == 0)
                keep_chunk &= ~np.isin(type_chunk, excluded_ids)

                mask2 = keep_chunk & (sizes_chunk == 2)
                if bool(np.any(mask2)):
                    nodes2 = np.asarray(nodes_chunk[mask2, :2], dtype=np.int64)
                    n2 = int(nodes2.shape[0])
                    edge_index[:, cursor : cursor + n2] = nodes2.T
                    edge_attr[cursor : cursor + n2] = type_chunk[mask2].astype(np.float32, copy=False)
                    cursor += n2

                mask4 = keep_chunk & (sizes_chunk == 4)
                if bool(np.any(mask4)):
                    nodes4 = np.asarray(nodes_chunk[mask4, :4], dtype=np.int64)
                    type4 = np.asarray(type_chunk[mask4], dtype=np.float32)
                    expanded = nodes4[:, pair_indices].reshape(-1, 2)
                    n4 = int(expanded.shape[0])
                    edge_index[:, cursor : cursor + n4] = expanded.T
                    edge_attr[cursor : cursor + n4] = np.repeat(type4, 6)
                    cursor += n4

        if cursor != num_processed_edges:
            raise RuntimeError(f"Processed edge count mismatch: expected {num_processed_edges}, wrote {cursor}")

        record_profile_event(
            "graph_cache_processed_ready",
            {
                "data_dir": self.data_dir,
                "num_processed_edges": int(num_processed_edges),
                "include_cell_edges": self.include_cell_edges,
            },
        )

    def _load_cache(self) -> None:
        with open(self._meta_path(), "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        self.edge_features_map = meta["edge_features_map"]
        self.num_nodes = int(meta["num_nodes"])
        self.edge_labels = np.load(self._raw_path("edge_labels"), mmap_mode="r")
        self.edge_nodes = np.load(self._raw_path("edge_nodes"), mmap_mode="r")
        self.edge_sizes = np.load(self._raw_path("edge_sizes"), mmap_mode="r")
        self.edge_type_ids = np.load(self._raw_path("edge_type_ids"), mmap_mode="r")
        self.edge_types_data = self.edge_type_ids
        self.train_mask = np.load(self._raw_path("train_mask"), mmap_mode="r")
        self.val_mask = np.load(self._raw_path("val_mask"), mmap_mode="r")
        self.test_mask = np.load(self._raw_path("test_mask"), mmap_mode="r")

        with profile_phase("graph.cache.load_processed_graph", {"data_dir": self.data_dir}):
            edge_index_np = np.load(self._processed_path("edge_index"), mmap_mode="r")
            edge_attr_np = np.load(self._processed_path("edge_attr"), mmap_mode="r")
            self.edge_index = torch.from_numpy(edge_index_np)
            self.edge_features = torch.from_numpy(edge_attr_np)
        self.num_edges = int(self.edge_index.size(1))

        record_profile_event(
            "graph_cache_loaded",
            {
                "data_dir": self.data_dir,
                "cache_dir": self.cache_dir,
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
                "include_cell_edges": self.include_cell_edges,
            },
        )

    def get_edge_index(self):
        if self.edge_index is None or self.edge_features is None:
            raise RuntimeError("Processed graph tensors are not initialized.")
        print(f"Edge index of the graph has shape: {self.edge_index.shape}")
        return self.edge_index, self.edge_features

    def _expand_rows_to_edges(self, rows: np.ndarray, sizes: np.ndarray, *, jts_col_only: bool = False) -> np.ndarray:
        if rows.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        if jts_col_only:
            return np.asarray(rows[:, :2], dtype=np.int64)

        mask2 = sizes == 2
        parts: List[np.ndarray] = []
        if bool(np.any(mask2)):
            parts.append(np.asarray(rows[mask2, :2], dtype=np.int64))
        mask4 = sizes == 4
        if bool(np.any(mask4)):
            nodes4 = np.asarray(rows[mask4, :4], dtype=np.int64)
            pair_indices = np.array(
                [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                dtype=np.int64,
            )
            parts.append(nodes4[:, pair_indices].reshape(-1, 2))
        if not parts:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    def _expand_labels(self, labels: np.ndarray, sizes: np.ndarray, *, jts_col_only: bool = False) -> np.ndarray:
        if labels.size == 0:
            return np.empty((0,), dtype=np.uint8)
        if jts_col_only:
            return np.asarray(labels, dtype=np.uint8)
        reps = np.where(sizes == 4, 6, 1)
        return np.repeat(np.asarray(labels, dtype=np.uint8), reps)

    def _split_mask(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train_mask
        if split == "val":
            return self.val_mask
        if split == "test":
            return self.test_mask
        raise ValueError(f"Unknown split={split}. Expected one of: train,val,test")

    def _get_task_edges_np(self, target_task: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        split_mask = self._split_mask(split)
        target_id = self.edge_type_to_id[target_task]
        select = (self.edge_type_ids == target_id) & (split_mask == 1)
        rows = np.asarray(self.edge_nodes[select], dtype=np.int64)
        sizes = np.asarray(self.edge_sizes[select], dtype=np.uint8)
        labels = np.asarray(self.edge_labels[select], dtype=np.uint8)
        jts_col_only = target_task == "joinable_table_search"
        edges = self._expand_rows_to_edges(rows, sizes, jts_col_only=jts_col_only)
        expanded_labels = self._expand_labels(labels, sizes, jts_col_only=jts_col_only)
        return edges, expanded_labels

    def get_train_edges(self, target_task) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        edges, labels = self._get_task_edges_np(target_task, "train")
        print(f"Training edges: {len(edges)}")
        return edges, labels, [target_task] * int(len(edges))

    def get_val_edges(self, target_task) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        edges, labels = self._get_task_edges_np(target_task, "val")
        print(f"Validation edges: {len(edges)}")
        return edges, labels, [target_task] * int(len(edges))

    def get_test_edges(self, target_task) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        edges, labels = self._get_task_edges_np(target_task, "test")
        print(f"Test edges: {len(edges)}")
        return edges, labels, [target_task] * int(len(edges))

    def get_edges_by_type(self, edge_type: str) -> List[Tuple[int, int]]:
        target_id = self.edge_type_to_id[edge_type]
        select = self.edge_type_ids == target_id
        rows = np.asarray(self.edge_nodes[select], dtype=np.int64)
        sizes = np.asarray(self.edge_sizes[select], dtype=np.uint8)
        edges = self._expand_rows_to_edges(rows, sizes, jts_col_only=edge_type == "joinable_table_search")
        return [tuple(map(int, pair)) for pair in edges.tolist()]

    def get_target_edges(self) -> Dict[str, List[Tuple[int, int]]]:
        return {target_class: self.get_edges_by_type(target_class) for target_class in self.target_classes}

    def sample_negative_edges(self, num_samples: int) -> List[Tuple[int, int]]:
        negative_samples: List[Tuple[int, int]] = []
        if self.edge_index is None:
            return negative_samples
        nodes_list = torch.unique(self.edge_index).cpu().tolist()
        existing_edges = set(zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()))

        while len(negative_samples) < num_samples:
            node1 = random.choice(nodes_list)
            node2 = random.choice(nodes_list)
            if node1 != node2 and (node1, node2) not in existing_edges and (node2, node1) not in existing_edges:
                negative_samples.append((node1, node2))
        return negative_samples

    def get_adjacency_matrix(self) -> torch.Tensor:
        if self.edge_index is None:
            raise RuntimeError("edge_index not initialized")
        values = torch.ones(self.edge_index.size(1))
        return torch.sparse_coo_tensor(self.edge_index, values, (self.num_nodes, self.num_nodes))

    def get_node_mapping(self) -> Dict[int, int]:
        return {idx: idx for idx in range(self.num_nodes)}

    def get_node_embeddings(self, *, mmap_mode: Optional[str] = None) -> torch.Tensor:
        path = f"{self.data_dir}/node_embeddings.npy"
        with profile_phase("graph.load.node_embeddings", {"path": path, "mmap_mode": mmap_mode or "none"}):
            if mmap_mode is None:
                embeddings = np.load(path)
            else:
                embeddings = np.load(path, mmap_mode=mmap_mode)
        self.node_embeddings = torch.from_numpy(embeddings)
        record_profile_event(
            "node_embeddings_ready",
            {
                "path": path,
                "mmap_mode": mmap_mode or "none",
                "shape": list(self.node_embeddings.shape),
                "dtype": str(self.node_embeddings.dtype),
            },
        )
        return self.node_embeddings

    def print_statistics(self):
        print("=== Graph Statistics ===")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {self.num_edges}")

        print("\n=== Edge Type Statistics ===")
        type_counts = defaultdict(int)
        for type_id in np.asarray(self.edge_type_ids):
            type_counts[self.edge_id_to_type[int(type_id)]] += 1
        for edge_type, count in type_counts.items():
            print(f"{edge_type}: {count}")

        print("\n=== Data Split Statistics ===")
        print(f"Training edges: {int(np.sum(self.train_mask))}")
        print(f"Validation edges: {int(np.sum(self.val_mask))}")
        print(f"Test edges: {int(np.sum(self.test_mask))}")
