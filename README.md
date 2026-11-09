# Uniprep

Joint multi-task table discovery with LLM-generated atom features and online symbolic feature channels.

## Pipeline

Three sequential steps per task/dataset, all driven by `scripts/run_task_featgen_pipeline.sh`:

1. **Atom generation** — LLM generates new atom feature functions to extend the base feature pool
2. **Symbolic spec generation** — LLM composes a multi-channel symbolic spec over the full atom pool
3. **Training** — GNN + static atoms + symbolic channels trained end-to-end

## Quick start

```bash
export DEEPSEEK_API_KEY="<your_key>"

TASK=entity_matching \
DATASET=wikidbs \
GPU_ID=0 \
GRAPH_DIR=/path/to/wikidbs_040303_no_token \
TABLE_ROOT=/path/to/wikidbs_040303/datalake_plus \
bash scripts/run_task_featgen_pipeline.sh
```

The default feature generator is `deepseek-flash`. Set `LLM_MODEL` and
`OPENAI_BASE_URL` to use another compatible API. Model selection and decision
thresholds use validation data only; each task/dataset is trained independently.

Set `ATOM_GEN_MODE=dryrun` and `SYMBOLIC_GEN_MODE=dryrun` to test without API calls. See the script header for all configurable parameters.

## Smoke test

EM on Magellan, dryrun by default:

```bash
GRAPH_DIR=/path/to/magellan_040303_no_token \
TABLE_ROOT=/path/to/magellan_040303/datalake_plus \
bash scripts/run_em_featgen_smoke.sh
```
