from __future__ import annotations

from typing import Dict, List, TypedDict


class TaskFeatgenConfig(TypedDict):
    short_name: str
    task_scope: str
    teacher_pool_atoms: List[str]
    known_existing_atoms: List[str]
    symbolic_source_groups_csv: str
    decoder_static_groups_csv: str
    decoder_static_atoms: List[str]
    human_exemplar_atoms: List[str]
    task_description: str
    data_description: str
    selection_notes: str
    current_generated_runtime_support: bool
    notes: str


TASK_FEATGEN_CONFIGS: Dict[str, TaskFeatgenConfig] = {
    "entity_matching": {
        "short_name": "em",
        "task_scope": "row_pair",
        "teacher_pool_atoms": [
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
        ],
        "known_existing_atoms": [
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
        ],
        "symbolic_source_groups_csv": "embedding_similarity,row_value_overlap,row_profile,serial_value_alignment,serial_lexical_plus",
        "decoder_static_groups_csv": "serial_value_alignment",
        "decoder_static_atoms": [
            "row_serial_token_jaccard",
            "row_serial_edit_similarity",
            "row_numeric_value_overlap",
        ],
        "human_exemplar_atoms": [
            "row_serial_edit_similarity",
            "row_value_containment_max",
            "row_token_idf_jaccard",
        ],
        "task_description": (
            "Entity matching on heterogeneous table rows. Each candidate pair links two rows that may describe "
            "the same real-world entity even when cell values are noisy, partially missing, reordered, or "
            "formatted differently."
        ),
        "data_description": (
            "Each row is represented by normalized cell values, row-level token and value sets, serialized row "
            "text, header-aware signals, simple numeric summaries, and row embeddings. Good atom features should "
            "be smooth, interpretable, and robust across datasets with different schemas and value styles."
        ),
        "selection_notes": (
            "The teacher pool is manually defined. The preserved exemplar atoms are also manually selected by us; "
            "the model must treat them as examples of desired feature style and as protected atoms that should not "
            "be regenerated. For EM, the preserved exemplars are intentionally chosen to span complementary views "
            "of matching evidence: soft serial alignment, smooth value containment, and rare-token salience. The "
            "generated atoms should explore nearby but distinct views, and should still contribute at least a little "
            "header-aware or aligned-value evidence instead of collapsing into only one local pattern."
        ),
        "current_generated_runtime_support": True,
        "notes": "0428 atom-generation is fully wired only for EM at the moment.",
    },
    "joinable_table_search": {
        "short_name": "jts",
        "task_scope": "column_pair",
        "teacher_pool_atoms": [
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
        ],
        "known_existing_atoms": [
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
        ],
        "symbolic_source_groups_csv": "jaccard_containment,value_distribution,overlap_coverage,value_profile,header_similarity",
        "decoder_static_groups_csv": "",
        "decoder_static_atoms": [],
        "human_exemplar_atoms": [
            "containment_max",
            "coverage_max",
            "unique_ratio_sim",
        ],
        "task_description": (
            "Joinable table search on column pairs. Each candidate pair compares two columns from different tables "
            "and predicts whether one column can meaningfully join with the other under noisy values and schema drift."
        ),
        "data_description": (
            "Each column is represented by normalized cell values, value sets, coverage and containment statistics, "
            "header tokens, simple numeric and uniqueness ratios, and coarse value-distribution summaries."
        ),
        "selection_notes": (
            "The teacher pool is manually defined. The exemplar atoms are manually selected by us to cover overlap, "
            "coverage, and profile-style evidence from complementary viewpoints; they are examples and exclusions, "
            "not outputs to regenerate."
        ),
        "current_generated_runtime_support": True,
        "notes": "0428 task-aware atom-generation prompt and runtime plumbing are available for JTS.",
    },
    "schema_matching": {
        "short_name": "sm",
        "task_scope": "column_pair",
        "teacher_pool_atoms": [
            "header_token_jaccard",
            "header_edit_similarity",
            "unique_ratio_sim",
            "missing_ratio_sim",
            "numeric_ratio_sim",
            "avg_len_ratio_sim",
            "value_jaccard",
            "value_containment_max",
        ],
        "known_existing_atoms": [
            "header_token_jaccard",
            "header_edit_similarity",
            "unique_ratio_sim",
            "missing_ratio_sim",
            "numeric_ratio_sim",
            "avg_len_ratio_sim",
            "value_jaccard",
            "value_containment_max",
        ],
        "symbolic_source_groups_csv": "header_similarity,value_stats,value_overlap",
        "decoder_static_groups_csv": "value_stats",
        "decoder_static_atoms": [
            "unique_ratio_sim",
            "missing_ratio_sim",
            "numeric_ratio_sim",
            "avg_len_ratio_sim",
        ],
        "human_exemplar_atoms": [
            "header_edit_similarity",
            "value_containment_max",
            "unique_ratio_sim",
            "numeric_ratio_sim",
        ],
        "task_description": (
            "Schema matching on column pairs. Each candidate pair compares two columns and predicts whether they "
            "express the same semantic field despite header variation, sparse data, and type mismatch."
        ),
        "data_description": (
            "Each column is summarized by header tokens and text, normalized value sets, missingness, uniqueness, "
            "numeric ratio, and average string length. Useful atom features should balance header semantics with "
            "value evidence and profile stability."
        ),
        "selection_notes": (
            "The teacher pool is manually defined. The exemplar atoms are manually chosen by us to cover header, "
            "value-overlap, and profile-stat evidence from different angles; treat them as reference style and do "
            "not regenerate them."
        ),
        "current_generated_runtime_support": True,
        "notes": "0428 task-aware atom-generation prompt and runtime plumbing are available for SM.",
    },
    "union_table_search": {
        "short_name": "uts",
        "task_scope": "table_pair",
        "teacher_pool_atoms": [
            "col_overlap_a2b_mean",
            "col_overlap_b2a_mean",
            "col_overlap_a2b_cov",
            "col_overlap_b2a_cov",
            "header_jaccard",
            "col_count_ratio",
            "row_count_ratio",
        ],
        "known_existing_atoms": [
            "col_overlap_a2b_mean",
            "col_overlap_b2a_mean",
            "col_overlap_a2b_cov",
            "col_overlap_b2a_cov",
            "header_jaccard",
            "col_count_ratio",
            "row_count_ratio",
        ],
        "symbolic_source_groups_csv": "column_overlap,header_jaccard,table_size_ratio",
        "decoder_static_groups_csv": "",
        "decoder_static_atoms": [],
        "human_exemplar_atoms": [
            "col_overlap_a2b_mean",
            "col_overlap_b2a_mean",
            "header_jaccard",
        ],
        "task_description": (
            "Union table search on table pairs. Each candidate pair compares two whole tables and predicts whether "
            "their schemas and value populations are compatible enough to union."
        ),
        "data_description": (
            "Each table is represented by per-column value sets, aggregate header tokens, row count, and column count. "
            "Useful atom features should capture directional overlap, header alignment, and size compatibility without "
            "becoming brittle to sampling noise."
        ),
        "selection_notes": (
            "The exemplar atoms are manually selected by us to anchor three distinct but complementary views for union "
            "compatibility: forward directional overlap, reverse directional overlap, and header alignment. Generated "
            "atoms should prioritize bidirectional column-overlap quality, coverage, directional agreement, best-match "
            "consistency, and value-population compatibility. Avoid spending too many slots on plain row/column count "
            "ratios or header-only variants; at most one generated atom should be mostly size-driven unless it is "
            "tightly coupled to overlap structure."
        ),
        "current_generated_runtime_support": True,
        "notes": "0428 task-aware atom-generation prompt and runtime plumbing are available for UTS.",
    },
}


DATASET_CSV_DEFAULT = "magellan,santos_benchmark,wikidbs"
TASK_ORDER_DEFAULT = [
    "entity_matching",
    "joinable_table_search",
    "schema_matching",
    "union_table_search",
]
