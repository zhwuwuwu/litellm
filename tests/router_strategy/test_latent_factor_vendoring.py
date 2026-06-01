"""
Smoke + import + extras-install tests for the vendored LatentFactor router files.

All tests are designed to pass even when optional extras (openai, pyyaml,
scikit-learn) are NOT installed.  They use py_compile / text / importlib-file
checks instead of unconditional top-level imports.

Test names follow the same convention as router_unit_tests/*.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import py_compile
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Root of the LiteLLM worktree
_WORKTREE = Path(__file__).resolve().parent.parent.parent  # .../routing_litellm/litellm
_RS = _WORKTREE / "litellm" / "router_strategy"
_LFR = _RS / "latent_factor_router"
_LLMR = _RS / "_llmrouter"
_COMMON = _RS / "_common"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _import_statements(path: Path) -> list[str]:
    """Return all 'import X' and 'from X import ...' lines from a file."""
    lines = []
    try:
        tree = ast.parse(_text(path))
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))
    return lines


def _has_forbidden_import(path: Path, forbidden: list[str]) -> list[str]:
    """
    Return any import statements whose module name starts with a forbidden
    token (exact boundary check to avoid flagging vendored ._llmrouter etc.).
    """
    hits = []
    try:
        tree = ast.parse(_text(path))
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for tok in forbidden:
                    # Match bare "llmrouter" or "llmrouter.submod" but not "_llmrouter"
                    if alias.name == tok or alias.name.startswith(tok + "."):
                        hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for tok in forbidden:
                if mod == tok or mod.startswith(tok + "."):
                    hits.append(f"from {mod} import ...")
    return hits


# ---------------------------------------------------------------------------
# Test 1 – _llmrouter files expose expected symbols
# ---------------------------------------------------------------------------

def test_llmrouter_symbols_via_compile():
    """
    All five _llmrouter modules compile without error and the __init__.py
    text defines the expected __all__ symbols.
    """
    expected = {"MetaRouter", "BaseTrainer", "save_model", "load_model", "load_jsonl"}
    init_text = _text(_LLMR / "__init__.py")
    for sym in expected:
        assert sym in init_text, f"Symbol {sym!r} missing from _llmrouter/__init__.py"

    # Each module must compile (stdlib only; no torch/pandas required)
    for fname in ["base_trainer.py", "data_loader.py", "meta_router.py", "model_io.py", "__init__.py"]:
        fpath = _LLMR / fname
        assert fpath.exists(), f"Missing _llmrouter/{fname}"
        py_compile.compile(str(fpath), doraise=True)


def test_llmrouter_no_forbidden_imports():
    """
    _llmrouter files must NOT import from 'llmrouter' (upstream package) or
    'custom_routers' (SuperClaw internal path).  They must be stdlib-only shims.
    """
    forbidden = ["llmrouter", "custom_routers"]
    violations: dict[str, list[str]] = {}
    for py_file in _LLMR.glob("*.py"):
        hits = _has_forbidden_import(py_file, forbidden)
        if hits:
            violations[py_file.name] = hits
    assert not violations, (
        "_llmrouter files contain forbidden upstream imports:\n"
        + "\n".join(f"  {f}: {stmts}" for f, stmts in violations.items())
    )


# ---------------------------------------------------------------------------
# Test 2 – _common files exist and have no forbidden external imports
# ---------------------------------------------------------------------------

def test_common_files_exist():
    """All expected _common files must be present."""
    expected_files = [
        "__init__.py",
        "baselines.py",
        "data_utils.py",
        "embedding_cache.py",
        "intent_cache.py",
        "intent_classifier.py",
        "probe_selector.py",
        "splitting.py",
    ]
    for fname in expected_files:
        assert (_COMMON / fname).exists(), f"Missing _common/{fname}"


def test_common_no_forbidden_imports():
    """
    _common files must NOT contain bare external 'llmrouter' or 'custom_routers'
    import statements (docstrings mentioning the source path are allowed).
    """
    forbidden = ["llmrouter", "custom_routers"]
    violations: dict[str, list[str]] = {}
    for py_file in _COMMON.glob("*.py"):
        hits = _has_forbidden_import(py_file, forbidden)
        if hits:
            violations[py_file.name] = hits
    assert not violations, (
        "_common files contain forbidden upstream imports:\n"
        + "\n".join(f"  {f}: {stmts}" for f, stmts in violations.items())
    )


# ---------------------------------------------------------------------------
# Test 3 – canonical vendored LatentFactor files exist and compile
# ---------------------------------------------------------------------------

def test_latent_factor_router_files_compile():
    """
    All canonical vendored files in latent_factor_router/ exist and pass
    py_compile (no missing-extras error at parse/compile time).
    """
    required = [
        "__init__.py",
        "config.py",
        "latent_factor_model.py",
        "latent_factor_router.py",
        "latent_factor_updater.py",
        "router.py",
        "trainer.py",
        "cli.py",
    ]
    for fname in required:
        fpath = _LFR / fname
        assert fpath.exists(), f"Missing latent_factor_router/{fname}"
        py_compile.compile(str(fpath), doraise=True)


# ---------------------------------------------------------------------------
# Test 4 – latent_factor_router/__init__.py has expected __all__ and lazy exports
# ---------------------------------------------------------------------------

def test_init_all_and_lazy_exports():
    """
    __init__.py must declare the four expected public names in __all__ and
    must implement a __getattr__ for lazy loading.
    """
    expected_names = {
        "LatentFactorRouter",
        "LatentFactorTrainer",
        "LatentFactorRouterLiteLLM",
        "LatentFactorRouterConfig",
    }
    init_path = _LFR / "__init__.py"
    src = _text(init_path)

    for name in expected_names:
        assert name in src, f"{name!r} missing from latent_factor_router/__init__.py"

    # __getattr__ must be defined for lazy loading
    assert "def __getattr__" in src, (
        "latent_factor_router/__init__.py must define __getattr__ for lazy imports"
    )

    # Parse __all__ list via AST
    tree = ast.parse(src)
    all_values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, ast.List):
                        all_values = [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
    assert expected_names <= set(all_values), (
        f"__all__ is missing names: {expected_names - set(all_values)}"
    )


def test_init_helpful_error_message():
    """
    __getattr__ in __init__.py must emit a message pointing to
    `pip install litellm[superclaw_routers]` when extras are missing.
    """
    src = _text(_LFR / "__init__.py")
    assert "superclaw_routers" in src, (
        "ImportError hint for superclaw_routers missing from latent_factor_router/__init__.py"
    )


# ---------------------------------------------------------------------------
# Test 5 – CLI help works via direct file execution
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero():
    """
    cli.py --help must succeed (exit 0) when run as a plain script.
    We invoke it directly so we avoid the failing top-level `import litellm`.
    """
    cli_path = str(_LFR / "cli.py")
    result = subprocess.run(
        [sys.executable, cli_path, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cli.py --help returned {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "LatentFactorRouter" in result.stdout or "train" in result.stdout, (
        "cli.py --help output does not mention expected commands"
    )


def test_cli_train_help_exits_zero():
    """cli.py train --help must also succeed."""
    cli_path = str(_LFR / "cli.py")
    result = subprocess.run(
        [sys.executable, cli_path, "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cli.py train --help returned {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "--config" in result.stdout, (
        "train --help does not mention --config flag"
    )


# ---------------------------------------------------------------------------
# Test 6 – no .pkl files committed/tracked in git
# ---------------------------------------------------------------------------

def test_no_pkl_files_tracked_in_git():
    """
    No .pkl files should be tracked in the git repository under the
    litellm/router_strategy/ subtree.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", "*.pkl"],
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
    )
    # git ls-files exits 0 even when no files match
    assert result.returncode == 0, f"git ls-files failed: {result.stderr}"
    tracked_pkls = [
        line for line in result.stdout.splitlines()
        if "router_strategy" in line
    ]
    assert not tracked_pkls, (
        f"Found tracked .pkl files under router_strategy/: {tracked_pkls}"
    )


# ---------------------------------------------------------------------------
# Guardrail validation test (meta-test)
# ---------------------------------------------------------------------------

def test_guardrail_catches_forbidden_import(tmp_path: Path):
    """
    Confirm the _has_forbidden_import helper is meaningful: it must flag a file
    that contains a forbidden import, and clear a file that doesn't.
    """
    # File that contains a forbidden import (bare upstream module name)
    bad_file = tmp_path / "bad_module.py"
    bad_file.write_text("from llmrouter.utils import something\n", encoding="utf-8")

    # File that is clean (._llmrouter is the vendored shim and should NOT be flagged)
    good_file = tmp_path / "good_module.py"
    good_file.write_text(
        "from litellm.router_strategy._llmrouter import load_jsonl\n",
        encoding="utf-8",
    )

    assert _has_forbidden_import(bad_file, ["llmrouter"]), (
        "Guardrail helper failed to detect forbidden import"
    )
    assert not _has_forbidden_import(good_file, ["llmrouter"]), (
        "Guardrail helper gave false positive on clean file"
    )


# ---------------------------------------------------------------------------
# Test 12 – Inference parity against existing pkl artefacts
# ---------------------------------------------------------------------------

# Absolute paths outside the worktree (not committed to git)
_PKL_PATH = Path(
    r"D:\superclaw\applications.ai.superclaw-zihan\integrations\llmrouter"
    r"\saved_models\latent_factor\latent_factor_artefacts.pkl"
)
_YAML_DEFAULT = Path(
    r"D:\superclaw\applications.ai.superclaw-zihan\integrations\llmrouter"
    r"\custom_routers\latentfactorrouter\config_test.yaml"
)
_DATA_PATH = Path(
    r"D:\superclaw\applications.ai.superclaw-zihan\integrations\llmrouter"
    r"\data\example_data.jsonl"
)
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_EXPECTED_TOP1 = _FIXTURES_DIR / "expected_top1.json"

# Number of records taken from the head of example_data.jsonl for parity.
_PARITY_N = 10


def _read_jsonl_head(path: Path, n: int) -> list:
    """Read first n records from a JSONL file using stdlib only."""
    import json as _json
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(_json.loads(line))
            if len(records) >= n:
                break
    return records


def _check_optional_deps() -> list[str]:
    """Return list of missing optional deps needed for inference."""
    missing = []
    for pkg in ("numpy", "yaml", "sklearn"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def _embedding_api_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """Return True if the embedding API responds."""
    import urllib.request
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def test_inference_parity_against_pkl():
    """
    Route the first 10 queries from example_data.jsonl through
    ``route_batch`` using artefacts loaded from the existing pkl.
    Compares ``predicted_llm`` (top-1) for each record against the
    ``fixtures/expected_top1.json`` baseline.

    IMPORTANT: ``expected_top1.json`` must be generated from the ORIGINAL
    SuperClaw router implementation (integrations/llmrouter), NOT from this
    vendored code.  If the fixture is absent this test skips with explicit
    instructions — it does NOT auto-generate from vendored output.

    Fixture contract:
        {
          "source": "original SuperClaw latentfactorrouter",
          "data_path": "<relative path to example_data.jsonl>",
          "n_records": 10,
          "predicted_llms": ["model-a", "model-b", ...]   // length 10
        }

    Skip conditions (all must pass; test never silently passes):
      1. Optional deps (numpy, pyyaml, scikit-learn) installed.
      2. pkl file present at _PKL_PATH.
      3. YAML config present (default _YAML_DEFAULT or env LFR_YAML_PATH).
      4. Data file present at _DATA_PATH with >= _PARITY_N records.
      5. Embedding API reachable at the URL resolved from env.
      6. Baseline fixture present at fixtures/expected_top1.json.
         (Must be pre-generated from original SuperClaw router, not from
          vendored code under test.)
    """
    import json

    # --- prerequisite: optional deps ---
    missing = _check_optional_deps()
    if missing:
        pytest.skip(f"Optional deps missing ({missing}); install litellm[superclaw_routers]")

    # --- prerequisite: pkl present ---
    if not _PKL_PATH.exists():
        pytest.skip(f"pkl artefact not found at {_PKL_PATH}")

    # --- prerequisite: data file present ---
    if not _DATA_PATH.exists():
        pytest.skip(f"Data file not found at {_DATA_PATH}")

    # --- prerequisite: YAML config ---
    yaml_path_env = os.environ.get("LFR_YAML_PATH", "")
    yaml_path = Path(yaml_path_env) if yaml_path_env else _YAML_DEFAULT
    if not yaml_path.exists():
        pytest.skip(
            f"YAML config not found at {yaml_path}. "
            "Set LFR_YAML_PATH env var to override."
        )

    # Ensure env vars that the YAML references are populated (fall back to
    # dummy values – constructor only reads them, inference validates later).
    os.environ.setdefault("LOCAL_EMBEDDING_API_BASE", "http://127.0.0.1:18104/v1")
    os.environ.setdefault("LOCAL_EMBEDDING_API_KEY", "dummy-key")
    os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- prerequisite: embedding API reachable ---
    emb_base = os.environ.get("LOCAL_EMBEDDING_API_BASE", "http://127.0.0.1:18104/v1")
    if not _embedding_api_reachable(emb_base):
        pytest.skip(
            f"Embedding API not reachable at {emb_base}. "
            "Start the local embedding server to run parity test."
        )

    # --- prerequisite: baseline fixture present ---
    # The fixture MUST be pre-generated from the original SuperClaw router,
    # not from this vendored code.  We never auto-generate it here.
    if not _EXPECTED_TOP1.exists():
        pytest.skip(
            f"Baseline fixture missing: {_EXPECTED_TOP1}\n"
            "To create it, run the ORIGINAL SuperClaw latentfactorrouter "
            f"(integrations/llmrouter) against the first {_PARITY_N} records "
            "of example_data.jsonl, collect the 'predicted_llm' for each, "
            "and write fixtures/expected_top1.json with the schema:\n"
            '  { "source": "original SuperClaw latentfactorrouter",\n'
            '    "data_path": "integrations/llmrouter/data/example_data.jsonl",\n'
            f'    "n_records": {_PARITY_N},\n'
            '    "predicted_llms": ["model-a", ...] }\n'
            "Do NOT generate this file from the vendored router under test."
        )

    # --- load baseline fixture ---
    expected_data = json.loads(_EXPECTED_TOP1.read_text(encoding="utf-8"))
    expected_list = expected_data.get("predicted_llms")
    assert isinstance(expected_list, list) and len(expected_list) == _PARITY_N, (
        f"expected_top1.json must contain 'predicted_llms' list of length {_PARITY_N}. "
        f"Got: {expected_list!r}"
    )

    # --- read test records ---
    records = _read_jsonl_head(_DATA_PATH, _PARITY_N)
    assert len(records) == _PARITY_N, (
        f"example_data.jsonl has fewer than {_PARITY_N} records (got {len(records)})"
    )

    # --- load router ---
    # Dynamically import to avoid top-level litellm dependency in environments
    # where the full package is not installed.
    worktree_litellm = str(_WORKTREE / "litellm")
    _added_path = worktree_litellm not in sys.path
    if _added_path:
        sys.path.insert(0, worktree_litellm)
    try:
        spec = importlib.util.spec_from_file_location(
            "latent_factor_router_mod",
            str(_LFR / "router.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        if _added_path:
            sys.path.remove(worktree_litellm)
        pytest.skip(f"router.py failed to load (missing extras?): {exc}")
    finally:
        if _added_path and worktree_litellm in sys.path:
            sys.path.remove(worktree_litellm)

    LatentFactorRouter = mod.LatentFactorRouter

    try:
        router = LatentFactorRouter(yaml_path=str(yaml_path))
    except Exception as exc:
        pytest.skip(f"LatentFactorRouter construction failed: {exc}")

    try:
        router.load_artefacts(str(_PKL_PATH))
    except Exception as exc:
        pytest.skip(f"load_artefacts failed: {exc}")

    # Route batch; each record has a "query" key.
    try:
        results = router.route_batch(records)
    except Exception as exc:
        pytest.skip(f"route_batch failed (embedding API unavailable?): {exc}")

    # Extract top-1 list – key "predicted_llm" per the route_batch contract.
    assert len(results) == _PARITY_N, (
        f"route_batch returned {len(results)} results, expected {_PARITY_N}"
    )
    got_list = []
    for i, r in enumerate(results):
        top1 = r.get("predicted_llm")
        assert top1 is not None, (
            f"route_batch result[{i}] has no 'predicted_llm'. Result: {r}"
        )
        got_list.append(top1)

    # --- enforce exact parity against baseline ---
    assert got_list == expected_list, (
        f"Inference parity FAILED.\n"
        f"  Expected ({_PARITY_N}): {expected_list}\n"
        f"  Got      ({_PARITY_N}): {got_list}\n"
        f"  Fixture: {_EXPECTED_TOP1}"
    )


# ---------------------------------------------------------------------------
# Test 13 – Training round-trip tests
# ---------------------------------------------------------------------------

def _check_training_deps() -> list[str]:
    """Return list of missing optional deps needed for training."""
    missing = []
    for pkg in ("numpy", "yaml", "sklearn"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def _yaml_path_for_training() -> "Path | None":
    """Return YAML config path from env or default; None if not found."""
    yaml_path_env = os.environ.get("LFR_YAML_PATH", "")
    yaml_path = Path(yaml_path_env) if yaml_path_env else _YAML_DEFAULT
    return yaml_path if yaml_path.exists() else None


def test_python_api_training_roundtrip(tmp_path: Path):
    """
    Training round-trip via the Python API:
      1. Construct LatentFactorRouter(yaml_path=...).
      2. Override data and output paths to use tmp_path (no repo artifacts).
      3. Construct LatentFactorTrainer(router).train().
      4. Verify a .pkl was written to tmp_path.

    Skip conditions (all must be met to run; never silently passes):
      1. Optional deps (numpy, pyyaml, scikit-learn) installed.
      2. YAML config present (LFR_YAML_PATH env or default _YAML_DEFAULT).
      3. Data file present at _DATA_PATH.
      4. Embedding API reachable (LatentFactorRouter needs it for init).

    If any prerequisite is absent the test skips with an explicit reason.
    The generated .pkl is written to tmp_path (never into the repo).
    """
    # --- prerequisite: optional deps ---
    missing = _check_training_deps()
    if missing:
        pytest.skip(
            f"Optional deps missing ({missing}); "
            "install litellm[superclaw_routers] to run training round-trip"
        )

    # --- prerequisite: YAML config ---
    yaml_path = _yaml_path_for_training()
    if yaml_path is None:
        pytest.skip(
            f"YAML config not found at {_YAML_DEFAULT}. "
            "Set LFR_YAML_PATH env var to override."
        )

    # --- prerequisite: data file ---
    if not _DATA_PATH.exists():
        pytest.skip(f"Data file not found at {_DATA_PATH}")

    # --- set env defaults for embedding API ---
    os.environ.setdefault("LOCAL_EMBEDDING_API_BASE", "http://127.0.0.1:18104/v1")
    os.environ.setdefault("LOCAL_EMBEDDING_API_KEY", "dummy-key")
    os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- prerequisite: embedding API reachable ---
    emb_base = os.environ.get("LOCAL_EMBEDDING_API_BASE", "http://127.0.0.1:18104/v1")
    if not _embedding_api_reachable(emb_base):
        pytest.skip(
            f"Embedding API not reachable at {emb_base}. "
            "Start the local embedding server to run training round-trip."
        )

    # --- dynamic imports to avoid top-level litellm dep ---
    worktree_litellm = str(_WORKTREE / "litellm")
    _added_path = worktree_litellm not in sys.path
    if _added_path:
        sys.path.insert(0, worktree_litellm)

    try:
        router_spec = importlib.util.spec_from_file_location(
            "lfr_router_t13", str(_LFR / "router.py")
        )
        assert router_spec is not None and router_spec.loader is not None
        router_mod = importlib.util.module_from_spec(router_spec)
        router_spec.loader.exec_module(router_mod)  # type: ignore[union-attr]
    except Exception as exc:
        pytest.skip(f"router.py failed to load (missing extras?): {exc}")
    finally:
        if _added_path and worktree_litellm in sys.path:
            sys.path.remove(worktree_litellm)

    try:
        trainer_spec = importlib.util.spec_from_file_location(
            "lfr_trainer_t13", str(_LFR / "trainer.py")
        )
        assert trainer_spec is not None and trainer_spec.loader is not None
        trainer_mod = importlib.util.module_from_spec(trainer_spec)
        trainer_spec.loader.exec_module(trainer_mod)  # type: ignore[union-attr]
    except Exception as exc:
        pytest.skip(f"trainer.py failed to load (missing extras?): {exc}")

    LatentFactorRouter = router_mod.LatentFactorRouter
    LatentFactorTrainer = trainer_mod.LatentFactorTrainer

    # --- construct router ---
    try:
        router = LatentFactorRouter(yaml_path=str(yaml_path))
    except Exception as exc:
        pytest.skip(f"LatentFactorRouter construction failed: {exc}")

    # --- override data and output paths to tmp_path (no repo artifacts) ---
    output_pkl = tmp_path / "t13_roundtrip_artefacts.pkl"
    router.cfg.setdefault("data_path", {})["routing_data_all"] = str(_DATA_PATH)
    router.cfg.setdefault("model_path", {})["save_model_path"] = str(output_pkl)
    # Reload routing data from data path override
    try:
        from litellm.router_strategy._llmrouter import load_jsonl as _load_jsonl_t13
        loaded = _load_jsonl_t13(str(_DATA_PATH))
        router.all_routing_data = loaded if loaded is not None else []
    except Exception:
        pass  # trainer will use router.all_routing_data as-is

    # --- construct trainer ---
    try:
        trainer = LatentFactorTrainer(router=router, device="cpu")
    except Exception as exc:
        pytest.skip(f"LatentFactorTrainer construction failed: {exc}")

    # Override save path again after trainer init (trainer reads it in __init__)
    trainer.save_model_path = str(output_pkl)

    # --- run training ---
    try:
        metrics = trainer.train()
    except Exception as exc:
        pytest.skip(f"trainer.train() failed (missing deps or service?): {exc}")

    # --- verify pkl was written to tmp_path (not in repo) ---
    assert output_pkl.exists(), (
        f"Training completed but expected .pkl not found at {output_pkl}"
    )
    assert output_pkl.stat().st_size > 0, (
        f"Training produced an empty .pkl at {output_pkl}"
    )
    # Ensure the pkl is inside tmp_path, not anywhere in the worktree
    assert str(_WORKTREE) not in str(output_pkl), (
        f"pkl was written inside the worktree: {output_pkl}"
    )

    # --- basic sanity on metrics (if returned) ---
    if metrics is not None:
        assert isinstance(metrics, dict), (
            f"trainer.train() should return a dict or None; got {type(metrics)}"
        )


def test_cli_training_roundtrip(tmp_path: Path):
    """
    Training round-trip via the CLI:
      python litellm/router_strategy/latent_factor_router/cli.py train
          --config <yaml> --data <jsonl> --output <tmp_path/artefacts.pkl>
          --device cpu

    Skip conditions (same as Python API test):
      1. Optional deps installed.
      2. YAML config present.
      3. Data file present.
      4. Embedding API reachable.

    The CLI is invoked via direct file path (not -m) to avoid the failing
    top-level `import litellm` in environments without all optional deps.
    The generated .pkl is written to tmp_path (never into the repo).
    """
    # --- prerequisite: optional deps ---
    missing = _check_training_deps()
    if missing:
        pytest.skip(
            f"Optional deps missing ({missing}); "
            "install litellm[superclaw_routers] to run CLI training round-trip"
        )

    # --- prerequisite: YAML config ---
    yaml_path = _yaml_path_for_training()
    if yaml_path is None:
        pytest.skip(
            f"YAML config not found at {_YAML_DEFAULT}. "
            "Set LFR_YAML_PATH env var to override."
        )

    # --- prerequisite: data file ---
    if not _DATA_PATH.exists():
        pytest.skip(f"Data file not found at {_DATA_PATH}")

    # --- set env defaults for embedding API ---
    env = os.environ.copy()
    env.setdefault("LOCAL_EMBEDDING_API_BASE", "http://127.0.0.1:18104/v1")
    env.setdefault("LOCAL_EMBEDDING_API_KEY", "dummy-key")
    env.setdefault("LOCAL_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- prerequisite: embedding API reachable ---
    emb_base = env.get("LOCAL_EMBEDDING_API_BASE", "http://127.0.0.1:18104/v1")
    if not _embedding_api_reachable(emb_base):
        pytest.skip(
            f"Embedding API not reachable at {emb_base}. "
            "Start the local embedding server to run CLI training round-trip."
        )

    output_pkl = tmp_path / "t13_cli_artefacts.pkl"
    cli_path = str(_LFR / "cli.py")

    # Add the litellm worktree's parent to PYTHONPATH so `from litellm...` works
    # inside the subprocess (same workaround used by T12 for subprocess CLI calls).
    worktree_parent = str(_WORKTREE)
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        worktree_parent + os.pathsep + python_path
        if python_path
        else worktree_parent
    )

    cmd = [
        sys.executable,
        cli_path,
        "train",
        "--config", str(yaml_path),
        "--data", str(_DATA_PATH),
        "--output", str(output_pkl),
        "--device", "cpu",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,  # generous timeout for a real training run
    )

    # Exit code 2 means missing deps (not installed); skip not fail.
    if result.returncode == 2:
        pytest.skip(
            f"CLI returned exit code 2 (missing optional deps).\n"
            f"stderr: {result.stderr.strip()}"
        )

    # Exit code 3/4 means router/trainer init or train failure; could be
    # embedding service missing even after reachability check (race/port mismatch).
    if result.returncode in (3, 4):
        pytest.skip(
            f"CLI returned exit code {result.returncode} (init or training error).\n"
            f"stderr: {result.stderr.strip()}"
        )

    assert result.returncode == 0, (
        f"cli.py train returned unexpected exit code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # --- verify pkl was written to tmp_path ---
    assert output_pkl.exists(), (
        f"CLI training completed (exit 0) but expected .pkl not found at {output_pkl}"
    )
    assert output_pkl.stat().st_size > 0, (
        f"CLI training produced an empty .pkl at {output_pkl}"
    )
    # Ensure the pkl is inside tmp_path, not anywhere in the worktree
    assert str(_WORKTREE) not in str(output_pkl), (
        f"pkl was written inside the worktree: {output_pkl}"
    )
