# DRA Training

Deep Research Agents — training and inference pipeline.

## Installation

Install the project in editable mode. This registers all packages so imports
resolve from any working directory (no `sys.path` manipulation required):

```bash
pip install -e .
```

## Layout

```
utils/         Leaf helper library — no dependencies on src/ or pipeline/.
               config, io_utils, text_utils, llm_client, vllm_manager.
pipeline/      Orchestration — controller factory, multi-GPU workers,
               evaluation/fusion helpers. Consumes utils/ and src/.
               (orchestration.py, eval_utils.py)
src/           Component packages, each importing from utils/:
               deep_research_agents, searcher_component, reasoner_component,
               controller_component, evaluation, indexing_corpus_dataset.
experiments/   CLI entry points (run_dra_inference.py). Import pipeline/ + utils/.
```

Dependency direction is one-way: `experiments → pipeline → src → utils`. `utils`
is a leaf and never imports from `src` or `pipeline`.

## Running

```bash
python experiments/run_dra_inference.py --agentic-model <agent> ...
```

See the module docstring in `experiments/run_dra_inference.py` for supported
agents and options.
