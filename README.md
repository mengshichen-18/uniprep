# Uniprep

Joint multi-task table discovery with LLM-generated atom features and online symbolic feature channels.

- Symbolic spec generation: `scripts/run_generate_symbolic_spec_v3.sh`
- End-to-end featgen + training: `scripts/run_task_featgen_pipeline.sh`

Symbolic specs and LLM-generated atom features are not bundled; generate them with your own API key.

## Step 1: Generate c12 symbolic specs

```bash
cd /path/to/uniprep
export OPENAI_API_KEY="<your_key>"

BATCH_DIR=./symbolic_specs/batches/$(date +%Y%m%d_%H%M%S)_c12
mkdir -p "${BATCH_DIR}/c12"/{entity_matching,joinable_table_search,schema_matching,union_table_search}

# EM (default selected EM atoms: 10)
bash ./scripts/run_generate_symbolic_spec_v3.sh \
  --task entity_matching \
  --feature-pool row_value_jaccard,row_value_containment_max,row_token_jaccard,row_serial_token_jaccard,row_serial_edit_similarity,row_numeric_value_overlap,row_serial_char3_jaccard,row_serial_char4_jaccard,row_token_idf_jaccard,row_numeric_rel_diff_sim \
  --output "${BATCH_DIR}/c12/entity_matching/cand_01.json"

# JTS (default selected JTS atoms: 7)
bash ./scripts/run_generate_symbolic_spec_v3.sh \
  --task joinable_table_search \
  --feature-pool jaccard,containment_max,unique_ratio_sim,numeric_ratio_sim,avg_len_ratio_sim,header_token_jaccard,header_edit_similarity \
  --output "${BATCH_DIR}/c12/joinable_table_search/cand_01.json"

# SM (default selected SM atoms: 6)
bash ./scripts/run_generate_symbolic_spec_v3.sh \
  --task schema_matching \
  --feature-pool header_token_jaccard,header_edit_similarity,unique_ratio_sim,missing_ratio_sim,numeric_ratio_sim,avg_len_ratio_sim \
  --output "${BATCH_DIR}/c12/schema_matching/cand_01.json"

# UTS (default selected UTS atoms: 5)
bash ./scripts/run_generate_symbolic_spec_v3.sh \
  --task union_table_search \
  --feature-pool col_overlap_a2b_mean,col_overlap_b2a_mean,col_overlap_a2b_cov,col_overlap_b2a_cov,header_jaccard \
  --output "${BATCH_DIR}/c12/union_table_search/cand_01.json"
```

Notes:
- Default generation channel count is `12` (c12).
- Output convention is fixed to `<batch_dir>/c12/{task}/cand_01.json`.

## Step 2: Train online

```bash
cd /path/to/uniprep

# Run for a single task/dataset combination (repeat per task and dataset).
TASK=entity_matching DATASET=wikidbs GPU_ID=0 SEED=0 \
  GRAPH_DIR=/path/to/wikidbs_040303_no_token \
  TABLE_ROOT=/path/to/wikidbs_040303/datalake_plus \
  bash ./scripts/run_task_featgen_pipeline.sh
```

Training script defaults:
- atom feature generation mode controlled by `ATOM_GEN_MODE` (`real` | `dryrun`)
- symbolic spec generation is optional (`ENABLE_SYMBOLIC=1` to turn on)
- symbolic template fixed to `<ARTIFACT_DIR>/c12/{task}/cand_01.json`
- `GRAPH_DIR` and `TABLE_ROOT` must be set explicitly (no hard-coded defaults)
- missing specs will fail fast with a clear message

## Optional quick smoke (EM only)

```bash
DATASET=magellan \
  GRAPH_DIR=/path/to/magellan_040303_no_token \
  TABLE_ROOT=/path/to/magellan_040303/datalake_plus \
  bash ./scripts/run_em_featgen_smoke.sh
```
