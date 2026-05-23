from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch


logger = logging.getLogger(__name__)


def _parse_edge_update_splits(raw: str) -> List[str]:
    if raw is None:
        return []
    splits = [s.strip() for s in raw.split(",") if s.strip()]
    allowed = {"train", "val", "test"}
    unknown = [s for s in splits if s not in allowed]
    if unknown:
        raise ValueError(f"Unknown splits in --edge_update_splits: {unknown}. Allowed: {sorted(allowed)}")
    return splits


def _append_edges_to_graph(
    graph,
    edge_pairs: List[Tuple[int, int]],
    *,
    edge_type: str,
    direction: str = "undirected",
    dedup_cache: Optional[Set[Tuple[int, int, int]]] = None,
) -> int:
    if not edge_pairs:
        return 0

    if edge_type not in graph.edge_features_map:
        raise KeyError(
            f"edge_type={edge_type} not found in graph.edge_features_map keys={list(graph.edge_features_map.keys())}"
        )
    edge_type_id = int(graph.edge_features_map[edge_type])

    add_reverse = direction == "undirected"
    new_edges: List[Tuple[int, int]] = []
    new_edge_attrs: List[int] = []

    for u, v in edge_pairs:
        u = int(u)
        v = int(v)
        if dedup_cache is None or (u, v, edge_type_id) not in dedup_cache:
            new_edges.append((u, v))
            new_edge_attrs.append(edge_type_id)
            if dedup_cache is not None:
                dedup_cache.add((u, v, edge_type_id))

        if add_reverse:
            if dedup_cache is None or (v, u, edge_type_id) not in dedup_cache:
                new_edges.append((v, u))
                new_edge_attrs.append(edge_type_id)
                if dedup_cache is not None:
                    dedup_cache.add((v, u, edge_type_id))

    if not new_edges:
        return 0

    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    new_edge_attr = torch.tensor(new_edge_attrs, dtype=torch.float)

    if getattr(graph, "edge_index", None) is None or graph.edge_index.numel() == 0:
        graph.edge_index = new_edge_index
        graph.edge_features = new_edge_attr
    else:
        graph.edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1).contiguous()
        graph.edge_features = torch.cat([graph.edge_features, new_edge_attr], dim=0)

    if hasattr(graph, "edges") and isinstance(graph.edges, list):
        graph.edges.extend(new_edges)
    if hasattr(graph, "nodes"):
        for u, v in new_edges:
            graph.nodes.add(u)
            graph.nodes.add(v)
        graph.num_nodes = len(graph.nodes)
    graph.num_edges = int(graph.edge_index.size(1))

    return len(new_edges)


def _filter_edges_by_type_ids(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    drop_type_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not drop_type_ids:
        return edge_index, edge_attr
    edge_attr_long = edge_attr.long()
    keep = torch.ones((edge_attr_long.numel(),), dtype=torch.bool)
    for tid in drop_type_ids:
        keep &= edge_attr_long != int(tid)
    return edge_index[:, keep].contiguous(), edge_attr[keep].contiguous()


_TASK_OBJECT_NODE_KIND: Dict[str, str] = {
    "entity_matching": "row",
    "schema_matching": "column",
    "union_table_search": "table",
    # 0210: JTS supervision uses column-column pairs (table context comes via message passing).
    "joinable_table_search": "column",
}


def _get_node_ids_for_kind_cached(graph, kind: str) -> torch.Tensor:
    cache = getattr(graph, "_heavy_node_ids_cache", None)
    if isinstance(cache, dict) and kind in cache:
        return cache[kind]

    if getattr(graph, "edge_index", None) is None or getattr(graph, "edge_features", None) is None:
        raise RuntimeError("Graph edge_index/edge_features not initialized.")

    edge_index = graph.edge_index.detach().cpu()
    edge_attr = graph.edge_features.detach().cpu().long()

    def _ids_from(edge_type: str, endpoint: int) -> torch.Tensor:
        if edge_type not in graph.edge_features_map:
            return torch.empty((0,), dtype=torch.long)
        type_id = int(graph.edge_features_map[edge_type])
        mask = edge_attr == type_id
        if not bool(mask.any()):
            return torch.empty((0,), dtype=torch.long)
        return edge_index[endpoint, mask].to(dtype=torch.long)

    if kind == "row":
        ids = _ids_from("row_table", 0)
    elif kind == "column":
        ids = _ids_from("column_table", 0)
    elif kind == "table":
        ids_a = _ids_from("row_table", 1)
        ids_b = _ids_from("column_table", 1)
        if ids_a.numel() == 0 and ids_b.numel() == 0:
            ids = torch.empty((0,), dtype=torch.long)
        elif ids_a.numel() == 0:
            ids = torch.unique(ids_b)
        elif ids_b.numel() == 0:
            ids = torch.unique(ids_a)
        else:
            ids = torch.unique(torch.cat([ids_a, ids_b], dim=0))
    else:
        raise ValueError(f"Unknown node kind={kind}. Expected one of: row,column,table")

    if not isinstance(cache, dict):
        cache = {}
        graph._heavy_node_ids_cache = cache
    cache[kind] = ids
    logger.info(f"[Heavy] node_kind={kind} n_nodes={int(ids.numel())}")
    return ids


def _sample_cycle_pairs(
    node_ids: torch.Tensor,
    num_pairs: int,
    *,
    seed: int,
) -> torch.Tensor:
    node_ids = node_ids.detach().cpu().to(dtype=torch.long)
    if num_pairs <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    if node_ids.numel() < 2:
        raise ValueError("Cannot sample pairs: node_ids has < 2 nodes.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    n = int(node_ids.numel())
    remaining = int(num_pairs)
    rounds: List[torch.Tensor] = []
    round_idx = 0

    while remaining > 0:
        perm = torch.randperm(n, generator=gen)
        roll = (round_idx % max(1, n - 1)) + 1
        src = node_ids[perm]
        dst = node_ids[perm.roll(shifts=roll)]
        edge_round = torch.stack([src, dst], dim=0)
        if edge_round.size(1) > remaining:
            edge_round = edge_round[:, :remaining]
        rounds.append(edge_round)
        remaining -= int(edge_round.size(1))
        round_idx += 1

    return torch.cat(rounds, dim=1).contiguous()

