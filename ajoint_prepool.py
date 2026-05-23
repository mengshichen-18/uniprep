from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _get_node_embeddings_cached(graph) -> torch.Tensor:
    cached = getattr(graph, "node_embeddings", None)
    if isinstance(cached, torch.Tensor):
        return cached
    data_dir = getattr(graph, "data_dir", None)
    if isinstance(data_dir, str):
        path = os.path.join(data_dir, "node_embeddings.npy")
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        # Auto-memmap very large embedding matrices to avoid OOM.
        if size >= 8 * 1024**3:
            logger.info(f"[Embeddings] Using mmap_mode='r' for large node_embeddings.npy (size={size} bytes)")
            return graph.get_node_embeddings(mmap_mode="r")
    return graph.get_node_embeddings()


def _materialize_prepooled_embeddings_to_disk(
    graph,
    *,
    row_alpha: float,
    col_alpha: float,
    chunk_size: int,
) -> torch.Tensor:
    data_dir = getattr(graph, "data_dir", None)
    if not isinstance(data_dir, str):
        raise RuntimeError("Graph has no data_dir; cannot materialize prepooled embeddings.")

    src_path = os.path.join(data_dir, "node_embeddings.npy")
    cache_name = f"node_embeddings_prepooled_row{row_alpha:.4f}_col{col_alpha:.4f}_chunk{int(chunk_size)}.npy"
    dst_path = os.path.join(data_dir, cache_name)
    if os.path.exists(dst_path):
        logger.info(f"[Prepool] Using cached prepooled embeddings: {dst_path}")
        return torch.from_numpy(np.load(dst_path, mmap_mode="r"))

    logger.info(f"[Prepool] Materializing prepooled embeddings to disk: {dst_path}")
    src_np = np.load(src_path, mmap_mode="r")
    n_nodes, emb_dim = int(src_np.shape[0]), int(src_np.shape[1])
    dst_np = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src_np.dtype, shape=src_np.shape)

    # Copy original embeddings to the output file in chunks.
    copy_chunk = 8192
    for start in range(0, n_nodes, copy_chunk):
        end = min(start + copy_chunk, n_nodes)
        dst_np[start:end] = src_np[start:end]

    edges = _build_cell_pool_edges_cached(graph)

    def _apply_pool(src_ids: torch.Tensor, dst_ids: torch.Tensor, *, alpha: float, tag: str) -> None:
        if src_ids.numel() == 0:
            logger.warning(f"[Prepool] {tag}: no edges found; skip.")
            return
        src_ids = src_ids.detach().cpu().to(dtype=torch.long)
        dst_ids = dst_ids.detach().cpu().to(dtype=torch.long)
        dst_unique = torch.unique(dst_ids, sorted=True)
        num_dst = int(dst_unique.numel())
        if num_dst == 0:
            logger.warning(f"[Prepool] {tag}: no dst nodes found; skip.")
            return

        sums = torch.zeros((num_dst, emb_dim), dtype=torch.float32)
        counts = torch.zeros((num_dst,), dtype=torch.long)

        ones = torch.ones((min(chunk_size, src_ids.numel()),), dtype=torch.long)
        for start in range(0, int(src_ids.numel()), int(chunk_size)):
            end = min(start + int(chunk_size), int(src_ids.numel()))
            chunk_src = src_ids[start:end].numpy()
            chunk_dst = dst_ids[start:end]
            dst_pos = torch.searchsorted(dst_unique, chunk_dst)

            chunk_emb = torch.from_numpy(np.asarray(src_np[chunk_src], dtype=np.float32))
            sums.index_add_(0, dst_pos, chunk_emb)

            if ones.numel() != (end - start):
                ones = torch.ones((end - start,), dtype=torch.long)
            counts.index_add_(0, dst_pos, ones)

        pooled = sums / counts.clamp_min(1).to(dtype=torch.float32).unsqueeze(1)
        logger.info(f"[Prepool] {tag}: pooled {num_dst} dst nodes (edges={int(src_ids.numel())})")

        # Write back in dst chunks to limit peak RAM.
        write_chunk = 4096
        dst_unique_np = dst_unique.numpy()
        for start in range(0, num_dst, write_chunk):
            end = min(start + write_chunk, num_dst)
            dst_idx = dst_unique_np[start:end]
            pooled_chunk = pooled[start:end]
            if alpha == 0.0:
                updated = pooled_chunk
            else:
                orig = torch.from_numpy(np.asarray(dst_np[dst_idx], dtype=np.float32))
                updated = orig * float(alpha) + pooled_chunk * (1.0 - float(alpha))
            dst_np[dst_idx] = updated.numpy().astype(dst_np.dtype, copy=False)

    _apply_pool(edges["cell_row_src"], edges["cell_row_dst"], alpha=float(row_alpha), tag="cell->row")
    _apply_pool(edges["cell_col_src"], edges["cell_col_dst"], alpha=float(col_alpha), tag="cell->column")

    # Re-open as read-only memmap for use as x.
    logger.info(f"[Prepool] Done. Loading prepooled embeddings via mmap: {dst_path}")
    return torch.from_numpy(np.load(dst_path, mmap_mode="r"))


def _build_cell_pool_edges_cached(graph) -> Dict[str, torch.Tensor]:
    cached = getattr(graph, "_cell_pool_edges", None)
    if isinstance(cached, dict):
        return cached

    cell_row_src: List[int] = []
    cell_row_dst: List[int] = []
    cell_col_src: List[int] = []
    cell_col_dst: List[int] = []

    edge_labels = getattr(graph, "edge_labels", None)
    edge_lists = getattr(graph, "edge_lists", None)
    edge_types = getattr(graph, "edge_types_data", None)
    val_mask = getattr(graph, "val_mask", None)
    test_mask = getattr(graph, "test_mask", None)
    if edge_labels is None or edge_lists is None or edge_types is None:
        raise RuntimeError("Graph does not expose edge_labels/edge_lists/edge_types_data; cannot prepool.")

    for i, etype in enumerate(edge_types):
        if etype not in ("cell_row", "cell_column"):
            continue
        if edge_labels[i] == 0:
            continue
        if test_mask is not None and test_mask[i] == 1:
            continue
        if val_mask is not None and val_mask[i] == 1:
            continue

        nodes = edge_lists[i]
        if len(nodes) != 2:
            continue
        cell_id, parent_id = int(nodes[0]), int(nodes[1])
        if etype == "cell_row":
            cell_row_src.append(cell_id)
            cell_row_dst.append(parent_id)
        else:
            cell_col_src.append(cell_id)
            cell_col_dst.append(parent_id)

    edges = {
        "cell_row_src": torch.tensor(cell_row_src, dtype=torch.long),
        "cell_row_dst": torch.tensor(cell_row_dst, dtype=torch.long),
        "cell_col_src": torch.tensor(cell_col_src, dtype=torch.long),
        "cell_col_dst": torch.tensor(cell_col_dst, dtype=torch.long),
    }
    graph._cell_pool_edges = edges
    return edges


def _avg_pool_src_to_dst_inplace(
    node_embeddings: torch.Tensor,
    *,
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    blend_alpha: float,
    chunk_size: int,
    tag: str,
) -> int:
    if src_ids.numel() == 0:
        logger.warning(f"[Prepool] {tag}: no edges found; skip.")
        return 0
    if src_ids.numel() != dst_ids.numel():
        raise ValueError(f"[Prepool] {tag}: src_ids and dst_ids must have same length.")
    if chunk_size <= 0:
        raise ValueError(f"[Prepool] {tag}: chunk_size must be > 0.")
    if not (0.0 <= blend_alpha <= 1.0):
        raise ValueError(f"[Prepool] {tag}: blend_alpha must be in [0,1].")

    src_ids = src_ids.cpu()
    dst_ids = dst_ids.cpu()
    dst_unique, dst_inv = torch.unique(dst_ids, sorted=True, return_inverse=True)
    num_dst = int(dst_unique.numel())
    emb_dim = int(node_embeddings.size(1))

    sums = torch.zeros((num_dst, emb_dim), dtype=node_embeddings.dtype)
    counts = torch.zeros((num_dst,), dtype=torch.long)

    ones = torch.ones((min(chunk_size, src_ids.numel()),), dtype=torch.long)
    for start in range(0, src_ids.numel(), chunk_size):
        end = min(start + chunk_size, src_ids.numel())
        chunk_src = src_ids[start:end]
        chunk_dst = dst_inv[start:end]

        chunk_emb = node_embeddings[chunk_src]
        sums.index_add_(0, chunk_dst, chunk_emb)

        if ones.numel() != (end - start):
            ones = torch.ones((end - start,), dtype=torch.long)
        counts.index_add_(0, chunk_dst, ones)

    counts_clamped = counts.clamp_min(1).to(dtype=node_embeddings.dtype)
    pooled = sums / counts_clamped.unsqueeze(1)

    if blend_alpha == 0.0:
        node_embeddings[dst_unique] = pooled
    else:
        original = node_embeddings[dst_unique]
        node_embeddings[dst_unique] = original * blend_alpha + pooled * (1.0 - blend_alpha)

    logger.info(f"[Prepool] {tag}: updated {num_dst} dst nodes (edges={int(src_ids.numel())})")
    return num_dst


def _compute_pooled_overlay(
    node_embeddings_np: np.ndarray,
    *,
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    blend_alpha: float,
    chunk_size: int,
    tag: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if src_ids.numel() == 0:
        logger.warning(f"[Prepool] {tag}: no edges found; skip.")
        return torch.empty((0,), dtype=torch.long), torch.empty((0, 0), dtype=torch.float32)
    if src_ids.numel() != dst_ids.numel():
        raise ValueError(f"[Prepool] {tag}: src_ids and dst_ids must have same length.")
    if chunk_size <= 0:
        raise ValueError(f"[Prepool] {tag}: chunk_size must be > 0.")
    if not (0.0 <= blend_alpha <= 1.0):
        raise ValueError(f"[Prepool] {tag}: blend_alpha must be in [0,1].")

    src_ids = src_ids.detach().cpu().to(dtype=torch.long)
    dst_ids = dst_ids.detach().cpu().to(dtype=torch.long)
    dst_unique, dst_inv = torch.unique(dst_ids, sorted=True, return_inverse=True)
    num_dst = int(dst_unique.numel())
    emb_dim = int(node_embeddings_np.shape[1])
    if num_dst == 0:
        logger.warning(f"[Prepool] {tag}: no dst nodes found; skip.")
        return torch.empty((0,), dtype=torch.long), torch.empty((0, 0), dtype=torch.float32)

    sums = torch.zeros((num_dst, emb_dim), dtype=torch.float32)
    counts = torch.zeros((num_dst,), dtype=torch.long)

    ones = torch.ones((min(chunk_size, src_ids.numel()),), dtype=torch.long)
    for start in range(0, src_ids.numel(), chunk_size):
        end = min(start + chunk_size, src_ids.numel())
        chunk_src = src_ids[start:end].numpy()
        chunk_dst = dst_inv[start:end]
        chunk_emb = torch.from_numpy(np.asarray(node_embeddings_np[chunk_src], dtype=np.float32))
        sums.index_add_(0, chunk_dst, chunk_emb)
        if ones.numel() != (end - start):
            ones = torch.ones((end - start,), dtype=torch.long)
        counts.index_add_(0, chunk_dst, ones)

    pooled = sums / counts.clamp_min(1).to(dtype=torch.float32).unsqueeze(1)

    if blend_alpha == 0.0:
        updated = pooled
    else:
        orig = torch.from_numpy(np.asarray(node_embeddings_np[dst_unique.numpy()], dtype=np.float32))
        updated = orig * float(blend_alpha) + pooled * (1.0 - float(blend_alpha))

    logger.info(f"[Prepool] {tag}: pooled {num_dst} dst nodes (edges={int(src_ids.numel())})")
    return dst_unique, updated


def _build_prepool_overlay_cached(
    graph,
    *,
    row_alpha: float,
    col_alpha: float,
    chunk_size: int,
    enable_row: bool = True,
    enable_col: bool = True,
) -> Dict[str, torch.Tensor]:
    cfg = (float(row_alpha), float(col_alpha), int(chunk_size), bool(enable_row), bool(enable_col))
    cached = getattr(graph, "_prepool_overlay_cache", None)
    cached_cfg = getattr(graph, "_prepool_overlay_cfg", None)
    if isinstance(cached, dict) and cached_cfg == cfg:
        return cached

    node_embeddings = _get_node_embeddings_cached(graph)
    node_embeddings_np = node_embeddings.detach().cpu().numpy()
    edges = _build_cell_pool_edges_cached(graph)

    if enable_row:
        row_ids, row_emb = _compute_pooled_overlay(
            node_embeddings_np,
            src_ids=edges["cell_row_src"],
            dst_ids=edges["cell_row_dst"],
            blend_alpha=row_alpha,
            chunk_size=chunk_size,
            tag="cell->row",
        )
    else:
        logger.info("[Prepool] row overlay disabled by config.")
        row_ids = torch.empty((0,), dtype=torch.long)
        row_emb = None
    if enable_col:
        col_ids, col_emb = _compute_pooled_overlay(
            node_embeddings_np,
            src_ids=edges["cell_col_src"],
            dst_ids=edges["cell_col_dst"],
            blend_alpha=col_alpha,
            chunk_size=chunk_size,
            tag="cell->column",
        )
    else:
        logger.info("[Prepool] column overlay disabled by config.")
        col_ids = torch.empty((0,), dtype=torch.long)
        col_emb = None

    num_nodes = int(getattr(graph, "num_nodes", 0))
    if num_nodes <= 0:
        raise RuntimeError("Graph has invalid num_nodes; cannot build prepool overlay.")
    row_map = None
    col_map = None
    if enable_row:
        row_map = torch.full((num_nodes,), -1, dtype=torch.int32)
    if enable_col:
        col_map = torch.full((num_nodes,), -1, dtype=torch.int32)
    if row_map is not None and row_ids.numel() > 0:
        row_map[row_ids] = torch.arange(row_ids.numel(), dtype=torch.int32)
    if col_map is not None and col_ids.numel() > 0:
        col_map[col_ids] = torch.arange(col_ids.numel(), dtype=torch.int32)

    cache = {
        "row_map": row_map,
        "row_emb": row_emb,
        "col_map": col_map,
        "col_emb": col_emb,
    }
    graph._prepool_overlay_cache = cache
    graph._prepool_overlay_cfg = cfg
    logger.info(
        f"[Prepool] Overlay cached rows={int(row_ids.numel())} cols={int(col_ids.numel())} "
        f"dim={int(row_emb.size(1)) if row_emb is not None and row_emb.numel() > 0 else int(col_emb.size(1)) if col_emb is not None and col_emb.numel() > 0 else 0}"
    )
    return cache


def _get_prepooled_embeddings_cached(
    graph,
    *,
    enable: bool,
    row_alpha: float,
    col_alpha: float,
    chunk_size: int,
) -> torch.Tensor:
    if not enable:
        return _get_node_embeddings_cached(graph)

    cfg = (float(row_alpha), float(col_alpha), int(chunk_size))
    cached = getattr(graph, "_prepooled_node_embeddings", None)
    cached_cfg = getattr(graph, "_prepooled_node_embeddings_cfg", None)
    if isinstance(cached, torch.Tensor) and cached_cfg == cfg:
        return cached

    if isinstance(cached, torch.Tensor) and cached_cfg is not None and cached_cfg != cfg:
        logger.info(f"[Prepool] config changed {cached_cfg} -> {cfg}; reloading raw node embeddings from disk.")
        node_embeddings = _get_node_embeddings_cached(graph)
    else:
        node_embeddings = _get_node_embeddings_cached(graph)

    # If embeddings are memory-mapped (or otherwise not safely writable), materialize
    # a prepooled .npy cache on disk instead of doing in-place updates in RAM.
    try:
        np_view = node_embeddings.detach().cpu().numpy()
        flags = getattr(np_view, "flags", None)
        needs_disk = isinstance(np_view, np.memmap) or (flags is not None and (not bool(flags.writeable)))
    except Exception:
        needs_disk = False
    if needs_disk:
        node_embeddings = _materialize_prepooled_embeddings_to_disk(
            graph,
            row_alpha=row_alpha,
            col_alpha=col_alpha,
            chunk_size=chunk_size,
        )
        graph._prepooled_node_embeddings = node_embeddings
        graph._prepooled_node_embeddings_cfg = cfg
        return node_embeddings
    edges = _build_cell_pool_edges_cached(graph)

    _avg_pool_src_to_dst_inplace(
        node_embeddings,
        src_ids=edges["cell_row_src"],
        dst_ids=edges["cell_row_dst"],
        blend_alpha=row_alpha,
        chunk_size=chunk_size,
        tag="cell->row",
    )
    _avg_pool_src_to_dst_inplace(
        node_embeddings,
        src_ids=edges["cell_col_src"],
        dst_ids=edges["cell_col_dst"],
        blend_alpha=col_alpha,
        chunk_size=chunk_size,
        tag="cell->column",
    )

    graph._prepooled_node_embeddings = node_embeddings
    graph._prepooled_node_embeddings_cfg = cfg
    return node_embeddings
