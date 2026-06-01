# SuperClaw Router Migration Notes

This document tracks the migration of SuperClaw's smart routing logic into LiteLLM as custom router strategies.

## Scope

The goal is to vendor and adapt SuperClaw's LatentFactorRouter and its dependencies into the `litellm/router_strategy/` namespace. This enables LiteLLM to use SuperClaw's data-driven routing policies while maintaining a clean separation between core LiteLLM logic and specialized router implementations.

## Source to Target Mapping

| SuperClaw Source Path | LiteLLM Target Path | Status |
|---|---|---|
| `external/llmrouter-lib` (shim) | `litellm/router_strategy/_llmrouter/` | **Wave 1 Complete** |
| `custom_routers/common/` | `litellm/router_strategy/_common/` | **Wave 1 Complete** |
| `custom_routers/latentfactorrouter/latent_factor_model.py` | `litellm/router_strategy/latent_factor_router/latent_factor_model.py` | **Wave 2 Complete** |
| `custom_routers/latentfactorrouter/latent_factor_updater.py` | `litellm/router_strategy/latent_factor_router/latent_factor_updater.py` | **Wave 2 Complete** |
| `custom_routers/latentfactorrouter/router.py` | `litellm/router_strategy/latent_factor_router/router.py` | **Wave 2 Complete** |
| `custom_routers/latentfactorrouter/trainer.py` | `litellm/router_strategy/latent_factor_router/trainer.py` | **Wave 2 Complete** |
| `custom_routers/complexity/` | TBD | Out of Scope |
| `custom_routers/super/` | TBD | Out of Scope |

## Wave 1 Completed: Shared Foundation

Wave 1 established the infrastructure and shared utilities required by the router implementations.

- **`_llmrouter/`**: A frozen minimal fork of the `ulab-uiuc/LLMRouter` shim (pinned commit `22a991f`). It provides core abstractions: `MetaRouter`, `BaseTrainer`, and I/O helpers `save_model`, `load_model`, and `load_jsonl`.
- **`_common/`**: Vendored snapshot of SuperClaw's shared router utilities (Source SHA `e1737666604b0c0d67b6c5be171da9b234d18512`). Includes data normalization, embedding/intent caches, and stratified splitting logic.
- **Import Rewrites**:
    - `_common/data_utils.py`: Line 145 rewritten to use vendored `_llmrouter.load_jsonl`.
    - `_common/intent_classifier.py`: Line 39 rewritten to use sibling-relative `IntentCache` import.
- **Dependencies**: Added `superclaw_routers` extras group to `pyproject.toml` including `numpy`, `scipy`, `scikit-learn`, `openai`, `tiktoken`, `tqdm`, and `pyyaml`.

## Wave 2 Completed: LatentFactor Core

Wave 2 vendored the core logic for the Latent Factor router and trainer.

- **Files Vendored**:
    - `latent_factor_model.py`: Core model definition. Byte-identical to source.
    - `latent_factor_updater.py`: Parameter update logic. Line content identical; `TYPE_CHECKING` import points to `.router`.
    - `router.py`: Main `LatentFactorRouter` implementation.
    - `trainer.py`: `LatentFactorTrainer` implementation.
- **Import Rewrites**:
    - `router.py`: External `llmrouter` and `custom_routers.common` imports now point to `_llmrouter` and `_common`.
    - `trainer.py`: `BaseTrainer` now imports from `_llmrouter`.
    - `latent_factor_updater.py`: Rewired `TYPE_CHECKING` import for `LatentFactorRouter` to sibling-local `.router`.

## Inference Path

During inference, the router determines the optimal provider/model for a given request.

- **Current State**: Hook wiring is present via `LatentFactorRouterLiteLLM`. Full end-to-end runtime QA with live model deployments remains TODO (Wave 4).
- **Inference Flow**: Request text -> Embedding/Intent Extraction -> `route_single`/`route_batch` lookup -> Model selection.

## Training Path

Training involves building the latent factor models from historical performance data.

- **Current State**: Python API (`LatentFactorTrainer`) and CLI entrypoint (`cli.py`) are present. Actual training round-trip verification remains TODO (Wave 4) as the current environment lacks optional dependencies (`pyyaml`, `openai`).
- **Logic**: Uses Matrix Factorization/Latent Factor models to predict performance based on prompt features and model capabilities.

## Data Collection Path

Data collection captures model inputs and performance metrics (latency, cost, quality) to provide training signals.

- **Current State**: No separate data collection implementation was migrated. The training CLI consumes JSONL via `--data`. Data collection remains out-of-scope for this plan except for existing tests/fixtures.
- **Planned Mechanism**: Middleware or post-processing hooks in the LiteLLM router lifecycle to log interactions in the format expected by the `load_jsonl` utility.

## Wave 3 Completed: LiteLLM Integration and Training Entry Point

Wave 3 focused on wiring the vendored core into LiteLLM's hook system and providing a CLI for training.

- **Hook Wiring**: `latent_factor_router.py` now implements `LatentFactorRouterLiteLLM` (a `CustomLogger` pre-routing hook). It imports the vendored `.router.LatentFactorRouter` instead of the external `custom_routers` package.
- **Public Exports**: `__init__.py` now exposes lazy public exports for key symbols. Importing these symbols triggers a helpful message if `superclaw_routers` extras are missing.
    - `LatentFactorRouter`
    - `LatentFactorTrainer`
    - `LatentFactorRouterLiteLLM`
    - `LatentFactorRouterConfig`
- **CLI Entrypoint**: `cli.py` added with stdlib `argparse`. It supports:
    - Global `--help` and subcommand `train --help`.
    - `train` command with flags: `--config`, `--data`, `--output`, and `--device`.
- **Verification**: Verified that the CLI module compiles and that `--help` commands output correctly. Full training logic execution is deferred to Wave 4.

## Dependency Strategy

To avoid bloating the core LiteLLM installation, all router-specific dependencies are confined to the `superclaw_routers` extra.

```bash
pip install litellm[superclaw_routers]
```

## Verification Notes

- **Static Analysis**: Verified via `python -m compileall`.
- **Manual Check**: All migrated files must maintain verbatim parity with source (except documented rewrites) to simplify future upstream syncs.
- **Living Document**: This file is updated at the conclusion of each migration wave.
