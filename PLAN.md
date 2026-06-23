# Refactor Plan: centralize `utils/` + fix imports

**Aim:** all shared helpers live in `utils/` (a true leaf dependency), and
`src/*` components + the pipeline orchestration consume them. Replace the
scattered `sys.path.insert` hacks with a proper editable install.

## Target layout

```
pyproject.toml          # editable install; no more sys.path hacks
utils/                  # leaf helpers only: config, io_utils, text_utils, llm_client, vllm_manager
src/pipeline/           # orchestration.py (was utils/pipeline.py) + eval_utils.py; imports as `pipeline.*`
src/                    # components, import utils (namespace root, not a package)
experiments/            # CLI entry points, import pipeline + utils
```

Package **names** are unchanged (`utils.*`, `deep_research_agents.*`, ...), so
import strings barely move. Two source roots: `.` (utils) and `src/` (pipeline +
components). `src/pipeline/` is discovered as the top-level package `pipeline`.

```toml
[tool.setuptools.packages.find]
where = [".", "src"]
include = ["utils*", "pipeline*", "deep_research_agents*", "evaluation*",
           "searcher_component*", "reasoner_component*", "controller_component*",
           "indexing_corpus_dataset*"]
```

## Phase 1 — Packaging + import normalization

1. Create `pyproject.toml`; `pip install -e .`.
2. Convert 13 headless imports (`from prompts...`, `from agents...`,
   `from agent_tools...`) to fully-qualified `deep_research_agents.*`:
   searchr1.py, glm_agent.py, stepsearch.py, searcho1.py, selfask.py,
   oss_agent.py, oss_bedrock_agent.py, webweaver_agent.py, cpm_report.py,
   react.py, drtulu_agent.py, react_tools.py.
3. Delete all `sys.path.insert` blocks: run_dra_inference.py, base_agent.py,
   drtulu_agent.py, react.py, research.py, searcho1.py, searchr1.py,
   selfask.py, stepsearch.py, webweaver_agent.py, react_tools.py,
   index_builder_test.py, pipeline.py, eval_utils.py.
4. Gut `src/deep_research_agents/__init__.py` path-registration block.

Exit: `python -c "import deep_research_agents.agents, utils.text_utils"` from
any cwd; `python -m compileall src utils experiments`; zero `sys.path.insert`.

## Phase 2 — Move orchestration out of `utils/`

1. Create `src/pipeline/__init__.py`.
2. `git mv utils/pipeline.py src/pipeline/orchestration.py`,
   `git mv utils/eval_utils.py src/pipeline/eval_utils.py`.
3. Fix cross-ref in orchestration.py (`from utils.eval_utils` ->
   `from pipeline.eval_utils`). utils.* imports stay.
4. Update `experiments/run_dra_inference.py` consumers (lines ~81, 96, 268,
   851, 853).

Exit: `grep -rE "^from (deep_research_agents|evaluation|searcher_component)" utils/`
returns nothing (utils is a leaf).

## Phase 3 — Consolidate duplicated parsing helpers

Lift generic parsers into `utils/text_utils.py` (e.g. `try_parse_json`, shared
XML tool-call parsing); leave format-specific bits local. Audit:
webweaver_agent.py, cpm_explore_agent.py, glm_agent.py. Independent / deferrable.

## Phase 4 — Cleanup

- Remove dead `archive/utils/` etc. (zero live imports).
- Fix stale `utils.py` comments in index_builder_test.py.
- Update README/CLAUDE.md with `pip install -e .` + layout.

## Verification

1. Import smoke test from non-root cwd.
2. `python -m compileall src utils pipeline experiments`.
3. Short end-to-end run (smallest agent/dataset) — confirm multiprocessing
   spawn workers still pickle (`_gpu_worker` stays module-level).
4. `grep -rn "sys.path.insert"` -> zero.

## Risks

- multiprocessing.spawn pickling: worker fns must stay module-level in new home.
- Headless-import conversion is highest-touch; do it with path-hack removal in
  one commit.
- External optional dep `agentic_retrieval_research...` in pipeline.py is
  unrelated — leave as-is.
