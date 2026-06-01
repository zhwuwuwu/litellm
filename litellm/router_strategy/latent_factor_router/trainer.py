"""
LatentFactorRouter — trainer.py
================================
Training orchestrator for :class:`LatentFactorRouter`.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .router import LatentFactorRouter

import numpy as np
from litellm.router_strategy._llmrouter import BaseTrainer

# ---------------------------------------------------------------------------
# Academic colour palette (colourblind-friendly)
# ---------------------------------------------------------------------------
ACADEMIC_COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "baseline": "#6C757D",
    "grid": "#E0E0E0",
    "background": "#FAFAFA",
}
BASELINE_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]

logger = logging.getLogger(__name__)


class LatentFactorTrainer(BaseTrainer):
    """
    Trainer for :class:`LatentFactorRouter`.

    Parameters
    ----------
    router : LatentFactorRouter
    optimizer : optional
    device : str
    """

    def __init__(self, router: "LatentFactorRouter", optimizer: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(router=router, optimizer=optimizer, device=device)

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..")
        )

        def _resolve(rel_or_abs: str) -> str:
            if not rel_or_abs:
                return ""
            if os.path.isabs(rel_or_abs):
                return rel_or_abs
            return os.path.join(project_root, rel_or_abs)

        save_rel = router.cfg.get("model_path", {}).get("save_model_path", "")
        self.save_model_path: str = _resolve(save_rel) or os.path.join(
            project_root, "saved_models/latent_factor/latent_factor_artefacts.pkl"
        )

        eval_rel = router.cfg.get("model_path", {}).get("eval_results_path", "")
        if eval_rel:
            self.eval_results_path: str = _resolve(eval_rel)
        else:
            stem, _ = os.path.splitext(self.save_model_path)
            stem = stem.removesuffix("_artefacts")
            self.eval_results_path = stem + "_eval_results.json"

        n_items = len(router.all_routing_data)
        logger.info(
            f"[LatentFactorTrainer] Initialized.\n"
            f"  Unified data pool  : {n_items} queries\n"
            f"  save_model_path    : {self.save_model_path}\n"
            f"  eval_results_path  : {self.eval_results_path}"
        )
        if n_items == 0:
            logger.warning(
                "[LatentFactorTrainer] router.all_routing_data is empty. "
                "Check data_path.routing_data_all in the YAML."
            )

    def loss_func(self, outputs: Any, batch: Any):
        """Not applicable — LatentFactorModel uses ridge regression."""
        raise NotImplementedError(
            "LatentFactorModel uses ridge regression; there is no differentiable loss."
        )

    def train(self, dataloader=None) -> Dict[str, Any]:
        """
        Execute the LatentFactorRouter training pipeline.

        Steps
        -----
        1. prepare_data        — embed + classify intents + stratified split
        2. extract_matrices    — extract P (performance) and C (cost) matrices
        3. fit_latent_model     — fit latent factor model with 5-fold CV
        4. select_probe_set     — select probe set for cold-start
        5. evaluate             — accuracy + cost + baseline metrics (test set)
        6. save_artefacts       — persist to .pkl
        7. _print_results       — human-readable summary to stdout
        8. _save_results        — JSON + TXT report written to disk

        Parameters
        ----------
        dataloader : ignored

        Returns
        -------
        dict
            Full evaluation metrics from the held-out test split.
        """
        logger.info("\n" + "=" * 68)
        logger.info("  LatentFactorRouter Training Pipeline")
        logger.info("=" * 68)

        raw_data = self.router.all_routing_data
        if not raw_data:
            raise RuntimeError(
                "[LatentFactorTrainer] No data to train on. "
                "Ensure data_path.routing_data_all is set correctly in YAML."
            )
        logger.info(f"\n[1/6] Data pool: {len(raw_data)} queries.")

        logger.info("\n[2/6] Embedding + classifying intents + splitting...")
        train_data, test_data = self.router.prepare_data(raw_data)

        logger.info("\n[3/6] Extracting P and C matrices...")
        E, P, C, model_names = self.router.extract_matrices(train_data)
        logger.info(f"  E shape: {E.shape}, P shape: {P.shape}, C shape: {C.shape}")
        logger.info(f"  Models: {model_names}")

        logger.info(
            f"\n[4/6] Fitting latent factor model "
            f"(latent_dim={self.router.latent_dim}, alpha={self.router.alpha}, "
            f"n_folds={self.router.n_folds})..."
        )
        self.router.fit_latent_model(E, P, C)

        logger.info("\n[5/6] Selecting probe set for cold-start...")
        probe_indices, probe_embeddings = self.router.select_probe_set(E)
        logger.info(f"  Selected {len(probe_indices)} probes")

        logger.info("\n[6/6] Evaluating on test set...")
        results = self.router.evaluate(test_data)

        if getattr(self.router, "_verbose_enabled", False):
            logger.info("\n[6/6] Running verbose evaluation (per-query predicted vs ground truth)...")
            verbose_stats = self.router.verbose_eval(test_data)
            results["verbose_stats"] = verbose_stats

        Lp, Lc = self.router._latent_model.get_latent_representations()
        results["latent_dim"] = self.router.latent_dim
        results["alpha"] = self.router.alpha
        results["n_folds"] = self.router.n_folds
        results["E_shape"] = E.shape
        results["P_shape"] = P.shape
        results["C_shape"] = C.shape
        results["n_models"] = len(model_names)
        results["n_probes"] = len(probe_indices)
        results["Lp_shape"] = Lp.shape if Lp is not None else None
        results["Lc_shape"] = Lc.shape if Lc is not None else None
        results["model_type"] = getattr(self.router, "model_type", "ridge")

        logger.info(f"\nSaving artefacts -> {self.save_model_path}")
        self.router.save_artefacts(self.save_model_path)

        self._print_results(results)
        self._save_results(results, self.eval_results_path)

        # Generate Pareto frontier plot
        if self.eval_results_path:
            plot_path = str(
                pathlib.Path(self.eval_results_path).with_name("pareto_frontier.png")
            )
        else:
            plot_path = "pareto_frontier.png"
        saved_plot = plot_pareto_frontier(results, save_path=plot_path)
        if saved_plot:
            logger.info(f"[LatentFactorTrainer] Pareto plot saved -> {saved_plot}")

        # ── Performance-weight sweep + combined Pareto plot ───────────
        sweep = self.sweep_perf_weight(test_data)
        results["weight_sweep"] = sweep

        if self.eval_results_path:
            ablation_plot_path = str(
                pathlib.Path(self.eval_results_path).with_name(
                    "weight_ablation_pareto.png"
                )
            )
            ablation_json_path = str(
                pathlib.Path(self.eval_results_path).with_name(
                    "weight_ablation_results.json"
                )
            )
        else:
            ablation_plot_path = "weight_ablation_pareto.png"
            ablation_json_path = "weight_ablation_results.json"

        saved_ablation = plot_weight_ablation_pareto(
            sweep_results=sweep,
            save_path=ablation_plot_path,
        )
        if saved_ablation:
            logger.info(
                f"[LatentFactorTrainer] Weight-ablation Pareto plot -> {saved_ablation}"
            )

        # Persist sweep JSON
        try:
            _p = pathlib.Path(ablation_json_path)
            _p.parent.mkdir(parents=True, exist_ok=True)
            with open(_p, "w", encoding="utf-8") as _fh:
                json.dump(sweep, _fh, indent=2)
            logger.info(
                f"[LatentFactorTrainer] Weight-ablation JSON  -> {ablation_json_path}"
            )
        except Exception as exc:
            logger.warning(
                "[LatentFactorTrainer] Could not save sweep JSON: %s", exc
            )

        return results

    # ------------------------------------------------------------------
    # Performance-weight sweep
    # ------------------------------------------------------------------

    def sweep_perf_weight(
        self,
        test_data: List[Dict[str, Any]],
        perf_weights: Optional[List[float]] = None,
        *,
        step: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Evaluate the router at multiple ``perf_weight`` values.

        Because the latent factor model predicts performance **P** and
        cost **C** independently, ``_perf_weight`` / ``_cost_weight`` only
        affect the final balance-score formula at inference time.  This
        method exploits that property to sweep the entire trade-off curve
        **without retraining**.

        Embeddings are pre-computed once and reused across all weight values,
        and scoring is batched to avoid the overhead of repeated per-query
        routing calls.

        Parameters
        ----------
        test_data : list[dict]
            Held-out evaluation data (same format accepted by
            ``router.evaluate``).
        perf_weights : list[float], optional
            Explicit list of performance weights to try.
            Defaults to ``numpy.linspace(0.0, 1.0, round(1/step) + 1)``.
        step : float
            Step size when *perf_weights* is not given (default 0.1).

        Returns
        -------
        list[dict]
            One entry per weight value, each containing::

                {
                    "perf_weight": float,
                    "cost_weight": float,
                    "accuracy":    float,
                    "avg_cost":    float,
                    "baselines":   { ... },
                    "quality_cost_score": float,
                }
        """
        import numpy as np

        weights_to_sweep: List[float]
        if perf_weights is not None:
            weights_to_sweep = list(perf_weights)
        else:
            n_steps = int(round(1.0 / step)) + 1
            base = np.linspace(0.7, 1.0, n_steps).tolist()
            extras = [0.97, 1.0]
            weights_to_sweep = sorted(set(base + extras))

        # Pre-compute embeddings and P/C predictions once for all weight values.
        # Embeddings are weight-independent; this avoids repeated embedding calls.
        self.router._require_trained()
        if self.router.embedder is None:
            self.router._ensure_embedder_for_inference()

        queries = [d["query"] for d in test_data]
        E = self.router._embed(queries)
        P_pred, C_pred = self.router._latent_model.predict(E)

        # Baseline metrics (same for every weight value) — compute once.
        baselines = self.router.compute_baselines(test_data)

        # Remember original weights so we can restore them
        orig_pw = self.router._perf_weight
        orig_cw = self.router._cost_weight

        sweep_results: List[Dict[str, Any]] = []

        logger.info(
            f"\n[sweep_perf_weight] Sweeping {len(weights_to_sweep)} "
            f"weight values on {len(test_data)} test queries..."
        )

        for pw in weights_to_sweep:
            cw = round(1.0 - pw, 10)

            balance = self.router._score_models(P_pred, C_pred, pw, cw)
            chosen = np.argmax(balance, axis=1)

            eval_res = self.router.compute_metrics_from_chosen(chosen, test_data)
            eval_res["baselines"] = baselines
            eval_res["quality_cost_score"] = pw * eval_res["accuracy"] - cw * eval_res["avg_cost"]

            pareto_candidates = dict(baselines.get("per_model", {}))
            router_acc = eval_res["accuracy"]
            router_cost = eval_res["avg_cost"]
            pareto_candidates["LatentFactorRouter"] = {
                "accuracy": router_acc,
                "avg_cost": router_cost,
            }
            from .router import find_pareto_optimal_models
            eval_res["pareto_optimal_all"] = find_pareto_optimal_models(pareto_candidates)
            eval_res["pareto_optimal_base"] = baselines.get("pareto_optimal", [])

            entry: Dict[str, Any] = {
                "perf_weight": pw,
                "cost_weight": cw,
                "accuracy": eval_res.get("accuracy", 0.0),
                "avg_cost": eval_res.get("avg_cost", 0.0),
                "quality_cost_score": eval_res.get("quality_cost_score", 0.0),
                "baselines": eval_res.get("baselines", {}),
                "pareto_optimal_base": eval_res.get("pareto_optimal_base", []),
                "pareto_optimal_all": eval_res.get("pareto_optimal_all", []),
                "llm_request_counts": eval_res.get("llm_request_counts", {}),
            }
            sweep_results.append(entry)

            logger.info(
                f"  pw={pw:.2f}  acc={entry['accuracy']*100:6.2f}%  "
                f"cost=${entry['avg_cost']:.4f}  "
                f"qc_score={entry['quality_cost_score']:.4f}"
            )

        # Restore original weights
        self.router._perf_weight = orig_pw
        self.router._cost_weight = orig_cw

        logger.info("[sweep_perf_weight] Done.")
        return sweep_results

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Return a human-readable text report."""
        buf = io.StringIO()

        def w(line: str = "") -> None:
            buf.write(line + "\n")

        router_acc = results.get("accuracy", 0.0)
        router_cost = results.get("avg_cost", 0.0)
        n = results.get("total", 0)

        w("=" * 68)
        w("  LATENT FACTOR ROUTER — EVALUATION RESULTS (test split)")
        w("=" * 68)
        w(f"  Router Accuracy  : {router_acc*100:6.2f}%   ({n} queries)")
        w(f"  Router Avg Cost  : ${router_cost:.4f}")

        w()
        w("  -- Model Configuration --")
        w(f"  Model type       : {results.get('model_type', 'ridge')}")
        w(f"  Latent dim       : {results.get('latent_dim', 'N/A')}")
        w(f"  Alpha (ridge)    : {results.get('alpha', 'N/A')}")
        w(f"  N folds (CV)     : {results.get('n_folds', 'N/A')}")
        w(f"  E matrix shape   : {results.get('E_shape', 'N/A')}")
        w(f"  P matrix shape   : {results.get('P_shape', 'N/A')}")
        w(f"  C matrix shape   : {results.get('C_shape', 'N/A')}")
        w(f"  Lp matrix shape  : {results.get('Lp_shape', 'N/A')}")
        w(f"  Lc matrix shape  : {results.get('Lc_shape', 'N/A')}")
        w(f"  N models         : {results.get('n_models', 'N/A')}")
        w(f"  N probes         : {results.get('n_probes', 'N/A')}")

        baselines = results.get("baselines", {})
        if baselines:
            w()
            w(f"  {'Baseline':<22} {'Accuracy':>10} {'Avg Cost':>12}")
            w(f"  {'-'*22} {'-'*10} {'-'*12}")

            def _row(label, acc, cost, mark=""):
                w(f"  {label:<22} {acc*100:9.2f}% ${cost:11.4f}  {mark}")

            _row(
                "Oracle (ceiling)",
                baselines.get("oracle", {}).get("accuracy", 0),
                baselines.get("oracle", {}).get("avg_cost", 0),
                "← upper bound",
            )
            _row(
                "Best Single",
                baselines.get("best_single", {}).get("accuracy", 0),
                baselines.get("best_single", {}).get("avg_cost", 0),
                f"({baselines.get('best_single', {}).get('model', '')})",
            )
            _row(
                "Cheapest Single",
                baselines.get("cheapest_single", {}).get("accuracy", 0),
                baselines.get("cheapest_single", {}).get("avg_cost", 0),
                f"({baselines.get('cheapest_single', {}).get('model', '')})",
            )
            _row(
                "Random",
                baselines.get("random", {}).get("accuracy", 0),
                baselines.get("random", {}).get("avg_cost", 0),
            )
            _row(
                "Worst Single",
                baselines.get("worst_single", {}).get("accuracy", 0),
                baselines.get("worst_single", {}).get("avg_cost", 0),
                "← lower bound",
            )
            w(f"  {'─'*22} {'─'*10} {'─'*12}")
            _row("LatentFactorRouter (ours)", router_acc, router_cost, "◄")

        w()
        w(f"  Oracle gap         : {results.get('oracle_gap', 0)*100:+.2f}pp")
        w(f"  Oracle efficiency  : {results.get('oracle_efficiency', 0)*100:.1f}%")
        w(f"  vs. Best Single    : {results.get('vs_best_single', 0)*100:+.2f}pp")
        w(f"  vs. Random         : {results.get('vs_random', 0)*100:+.2f}pp")
        w(f"  Cost savings       : {results.get('cost_savings_vs_best', 0)*100:+.1f}%")
        w(f"  Quality-cost score : {results.get('quality_cost_score', 0):.4f}")

        llm_counts = results.get("llm_request_counts", {})
        if llm_counts:
            w()
            w("  -- Requests per LLM during evaluation --")
            w(f"  {'Model':<30} {'Requests':>10}")
            w(f"  {'-'*30} {'-'*10}")
            for m, cnt in sorted(llm_counts.items(), key=lambda x: x[1], reverse=True):
                w(f"  {m:<30} {cnt:>10}")

        per_model = baselines.get("per_model", {}) if baselines else {}
        if per_model:
            pareto_base = results.get("pareto_optimal_base", [])
            pareto_all = results.get("pareto_optimal_all", [])
            w()
            w(f"  {'Model':<30} {'Accuracy':>10} {'Avg Cost':>12} {'Pareto':>8}")
            w(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*8}")
            for m, stats in sorted(
                per_model.items(), key=lambda x: x[1]["accuracy"], reverse=True
            ):
                pareto_mark = "*" if m in pareto_base else ""
                w(
                    f"  {m:<30} {stats['accuracy']*100:9.2f}% "
                    f"${stats['avg_cost']:11.4f} {pareto_mark:>8}"
                )

            w()
            w(f"  Pareto-optimal base models ({len(pareto_base)}): "
              f"{', '.join(pareto_base) if pareto_base else 'none'}")
            w(f"  Pareto-optimal incl. router ({len(pareto_all)}): "
              f"{', '.join(pareto_all) if pareto_all else 'none'}")

        w("=" * 68)

        summary = {
            k: v
            for k, v in results.items()
            if k
            not in (
                "baselines",
                "dataset_accuracy",
                "intent_accuracy",
                "routing_details",
                "routing_distribution",
            )
        }
        w()
        w("Summary JSON:")
        w(json.dumps(summary, indent=2))

        return buf.getvalue()

    def _print_results(self, results: Dict[str, Any]) -> None:
        logger.info(self._format_results(results))

    def _save_results(
        self, results: Dict[str, Any], json_path: str
    ) -> None:
        if not json_path:
            logger.warning(
                "[LatentFactorTrainer] eval_results_path is empty — skipping file save."
            )
            return

        p = pathlib.Path(json_path)
        txt_path = p.with_suffix(".txt")
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = dict(results)
        payload["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"

        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        report = self._format_results(results)
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(report)

        logger.info(f"[LatentFactorTrainer] Evaluation results saved:")
        logger.info(f"  JSON : {p}")
        logger.info(f"  TXT  : {txt_path}")


def _find_pareto_frontier(
    points: List[Tuple[float, float]],
    maximize_y: bool = True,
    minimize_x: bool = True,
) -> List[Tuple[float, float]]:
    """Return the Pareto-optimal subset of *(x, y)* points.

    A point is Pareto-optimal when no other point dominates it — i.e. no
    other point is at least as good on *both* axes **and** strictly better
    on at least one.

    Parameters
    ----------
    points : list of (x, y)
    maximize_y : bool
        If ``True`` higher *y* is preferred; otherwise lower.
    minimize_x : bool
        If ``True`` lower *x* is preferred; otherwise higher.

    Returns
    -------
    list of (x, y)
        Pareto-optimal subset (order preserved).
    """
    pareto: List[Tuple[float, float]] = []
    for i, (x1, y1) in enumerate(points):
        dominated = False
        for j, (x2, y2) in enumerate(points):
            if i == j:
                continue
            x_ok = (x2 <= x1) if minimize_x else (x2 >= x1)
            y_ok = (y2 >= y1) if maximize_y else (y2 <= y1)
            x_strict = (x2 < x1) if minimize_x else (x2 > x1)
            y_strict = (y2 > y1) if maximize_y else (y2 < y1)
            if (x_ok and y_ok) and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            pareto.append((x1, y1))
    return pareto


def plot_pareto_frontier(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = False,
    with90: bool = False,
    with95: bool = False
) -> Optional[str]:
    """Plot average accuracy vs total inference cost with Pareto frontier.

    Renders all base models as scatter points and highlights the empirical
    Pareto frontier as a connected line.  The router's operating point is
    shown separately.  If ``pareto_optimal_base`` is not already present in
    *results*, the frontier is recomputed from the per-model data.

    Parameters
    ----------
    results : dict
        Evaluation results dict returned by ``LatentFactorTrainer.train()``
        or ``LatentFactorRouter.evaluate()``.  Expected keys:

        * ``accuracy`` – router accuracy
        * ``avg_cost`` – router average cost
        * ``baselines.per_model`` – {model: {accuracy, avg_cost}}
        * ``pareto_optimal_base`` – list of Pareto-optimal base model names
          (computed on the fly if missing)
        * ``pareto_optimal_all`` – list including router

    save_path : str, optional
        File path to save the figure (e.g. ``"pareto.png"``).
        If *None*, the figure is not written to disk.
    show : bool
        Whether to call ``plt.show()`` (useful in notebooks).

    Returns
    -------
    str or None
        Path to saved figure, or *None* if nothing was saved.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError as exc:
        logger.warning(
            "[plot_pareto_frontier] matplotlib/numpy not available: %s", exc
        )
        return None

    # ------------------------------------------------------------------
    # Extract data
    # ------------------------------------------------------------------
    baselines = results.get("baselines", {})
    per_model = baselines.get("per_model", {})
    if not per_model:
        logger.warning(
            "[plot_pareto_frontier] No per_model baselines found in results."
        )
        return None

    router_acc = results.get("accuracy", 0.0)
    router_cost = results.get("avg_cost", 0.0)
    pareto_all = results.get("pareto_optimal_all", [])

    # Pre-computed Pareto set or recompute from per_model data
    pareto_base: List[str] = results.get("pareto_optimal_base", [])
    if not pareto_base:
        pts = [
            (per_model[m]["avg_cost"], per_model[m]["accuracy"], m)
            for m in per_model
        ]
        frontier = _find_pareto_frontier(
            [(c, a) for c, a, _ in pts],
            maximize_y=True,
            minimize_x=True,
        )
        frontier_set = set(frontier)
        pareto_base = [
            m for c, a, m in pts if (c, a) in frontier_set
        ]

    names = list(per_model.keys())
    accs = np.array([per_model[m]["accuracy"] for m in names])
    costs = np.array([per_model[m]["avg_cost"] for m in names])
    pareto_mask = np.array([m in pareto_base for m in names])

    # ------------------------------------------------------------------
    # Academic figure setup (300 DPI, colourblind-friendly)
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "DejaVu Sans", "Liberation Sans",
            "Bitstream Vera Sans", "sans-serif",
        ],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # ------------------------------------------------------------------
    # Non-Pareto base models (grey scatter)
    # ------------------------------------------------------------------
    non_pareto = ~pareto_mask
    if non_pareto.any():
        ax.scatter(
            costs[non_pareto],
            accs[non_pareto],
            c=ACADEMIC_COLORS["baseline"],
            s=80,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
            label="Base models",
        )

    # ------------------------------------------------------------------
    # Pareto-optimal base models (distinct markers per model)
    # ------------------------------------------------------------------
    pareto_idx = np.where(pareto_mask)[0]
    for rank, idx in enumerate(pareto_idx):
        marker = BASELINE_MARKERS[rank % len(BASELINE_MARKERS)]
        model_label = names[idx] if len(names[idx]) <= 22 else names[idx][:19] + "..."
        ax.scatter(
            costs[idx],
            accs[idx],
            marker=marker,
            s=200,
            color=ACADEMIC_COLORS["primary"],
            edgecolors="black",
            linewidths=1,
            alpha=0.9,
            zorder=4,
            label=f"Pareto: {model_label}",
        )

    # ------------------------------------------------------------------
    # Pareto frontier line
    # ------------------------------------------------------------------
    pareto_points = [
        (per_model[m]["avg_cost"], per_model[m]["accuracy"])
        for m in pareto_base
        if m in per_model
    ]
    if pareto_points:
        pareto_points.sort(key=lambda p: p[0])
        px, py = zip(*pareto_points)
        ax.plot(
            px, py, "o-",
            color=ACADEMIC_COLORS["primary"],
            linewidth=3,
            markersize=0,          # markers already drawn individually above
            alpha=0.7,
            zorder=2,
            label="Pareto frontier (base)",
        )

    # ------------------------------------------------------------------
    # Router operating point
    # ------------------------------------------------------------------
    router_on_pareto = "LatentFactorRouter" in pareto_all
    router_color = ACADEMIC_COLORS["quaternary"] if router_on_pareto else ACADEMIC_COLORS["warning"]
    router_label = (
        "LatentFactorRouter (Pareto)" if router_on_pareto
        else "LatentFactorRouter"
    )
    ax.scatter(
        [router_cost],
        [router_acc],
        c=router_color,
        s=220,
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label=router_label,
    )

    # ------------------------------------------------------------------
    # Random reference line
    # ------------------------------------------------------------------
    random_acc = baselines.get("random", {}).get("accuracy")
    if random_acc is not None:
        ax.axhline(
            random_acc,
            color=ACADEMIC_COLORS["secondary"],
            linestyle=":",
            linewidth=1,
            alpha=0.6,
            label=f"Random ({random_acc*100:.1f}%)",
        )

    # ------------------------------------------------------------------
    # Best Single accuracy reference lines (100%, 95%, 90%)
    # ------------------------------------------------------------------
    best_single_acc = baselines.get("best_single", {}).get("accuracy")
    best_single_cost = baselines.get("best_single", {}).get("avg_cost", 0.0)

    def _pareto_intersect_cost(sorted_pts, target_acc):
        for j in range(len(sorted_pts) - 1):
            c_low, a_low = sorted_pts[j]
            c_high, a_high = sorted_pts[j + 1]
            if a_low <= target_acc <= a_high:
                denom = c_high - c_low
                if denom != 0:
                    slope = (a_high - a_low) / denom
                    return (c_low + (target_acc - a_low) / slope) if slope != 0 else None
                break
        for c, a in sorted_pts:
            if a >= target_acc:
                return c
        return None

    pf_sorted = sorted(pareto_points, key=lambda p: p[0]) if pareto_points else []

    if best_single_acc is not None:
        _acc_levels = [
            (best_single_acc,         ACADEMIC_COLORS["success"],   "-.",  f"Best Single ({best_single_acc*100:.1f}%)",                "vs best single",      10, -30)
        ]
        if with90:
            _acc_levels.append((best_single_acc * 0.95,  ACADEMIC_COLORS["warning"],   "--",  f"95% Best Single ({best_single_acc*0.95*100:.1f}%)",       "vs 95% best single",  10, -30))
        if with95:
            _acc_levels.append((best_single_acc * 0.90,  ACADEMIC_COLORS["secondary"], ":",   f"90% Best Single ({best_single_acc*0.90*100:.1f}%)",       "vs 90% best single",  10, -30))
        for target_acc, color, ls, h_label, ann_suffix, ann_x, ann_y in _acc_levels:
            ax.axhline(
                target_acc,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                alpha=0.7,
                label=h_label,
            )
            if pf_sorted and best_single_cost > 0:
                ic = _pareto_intersect_cost(pf_sorted, target_acc)
                if ic is not None:
                    ax.axvline(
                        ic,
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3,
                    )
                    saving_pct = ((best_single_cost - ic) / best_single_cost) * 100
                    ax.annotate(
                        f"Cost saving: {saving_pct:.1f}%\n({ann_suffix})",
                        (ic, target_acc),
                        textcoords="offset points",
                        xytext=(ann_x, ann_y),
                        fontsize=8,
                        fontweight="bold",
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8),
                    )

        if router_cost == 0:
            ax.plot(
                0, router_acc,
                marker="*", markersize=18, color=ACADEMIC_COLORS["tertiary"],
                markeredgecolor="black", markeredgewidth=0.5, zorder=7,
            )
            if best_single_cost > 0:
                saving_pct = ((best_single_cost - 0) / best_single_cost) * 100
                ax.annotate(
                    f"Cost saving: {saving_pct:.1f}%\n(cost=0, acc={router_acc*100:.1f}%)",
                    xy=(0, router_acc),
                    xytext=(15, -30),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color=ACADEMIC_COLORS["tertiary"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ACADEMIC_COLORS["tertiary"], alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=ACADEMIC_COLORS["tertiary"], linewidth=0.8),
                )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    for i, name in enumerate(names):
        label = name if len(name) <= 22 else name[:19] + "..."
        ax.annotate(
            label,
            (costs[i], accs[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            alpha=0.8,
        )

    router_llm_counts = results.get("llm_request_counts", {})
    if router_llm_counts:
        total_requests = sum(router_llm_counts.values())
        if total_requests > 0:
            llm_pct_text = ", ".join(
                f"{m.lower()}: {c/total_requests*100:.1f}%" for m, c in sorted(router_llm_counts.items(), key=lambda x: x[1], reverse=True) if c > 0
            )
            router_annotation = f"Router\n({llm_pct_text})"
        else:
            router_annotation = "Router"
    else:
        router_annotation = "Router"
    ax.annotate(
        router_annotation,
        (router_cost, router_acc),
        textcoords="offset points",
        xytext=(8, -10),
        fontsize=7,
        fontweight="bold",
        color=router_color,
    )

    # ------------------------------------------------------------------
    # Axes labels, title, legend
    # ------------------------------------------------------------------
    ax.set_xlabel("Average Per-Query Cost ($)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(
        "Pareto Frontier: Performance vs Cost Efficiency",
        fontweight="bold",
        pad=20,
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Secondary x-axis: cost saving % vs best single model
    # ------------------------------------------------------------------
    if best_single_cost > 0:
        _bsc = float(best_single_cost)

        def _cost_to_saving(c):  # type: ignore[override]
            return ((_bsc - np.asarray(c)) / _bsc) * 100

        def _saving_to_cost(s):  # type: ignore[override]
            return _bsc * (1.0 - np.asarray(s) / 100.0)

        ax2 = ax.secondary_xaxis(
            "top", functions=(_cost_to_saving, _saving_to_cost),
        )
        ax2.set_xlabel(
            "Cost Saving vs Best Single Model (%)", fontweight="bold",
        )
        ax2.tick_params(labelsize=8)

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    saved_path = None
    if save_path:
        p = pathlib.Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=300, bbox_inches="tight", facecolor="white")
        saved_path = str(p)
        logger.info(f"[plot_pareto_frontier] Saved -> {saved_path}")

    plt.close(fig)
    return saved_path


def plot_weight_ablation_pareto(
    sweep_results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = False,
    plot_random: bool = False,
    with90: bool = False,
    with95: bool = False
) -> Optional[str]:
    """Plot the performance–cost trade-off across a ``perf_weight`` sweep.

    Produces a **publication-ready** figure (300 DPI, colourblind-friendly)
    with five layers:

    1. **Router operating points** — one per ``perf_weight``, colour-mapped
       by weight value (viridis), connected by a trade-off curve.
    2. **Per-model baselines** — from the first sweep entry's
       ``baselines.per_model``, with Pareto-optimal models highlighted using
       distinct markers.
    3. **Combined Pareto frontier** — computed over *all* points (router
       sweep + base models), drawn as a bold connected line.
    4. **Random baseline reference line**.
    5. **Secondary x-axis** — shows cost saving percentage relative to the
       best single model.

    Parameters
    ----------
    sweep_results : list[dict]
        Output of :meth:`LatentFactorTrainer.sweep_perf_weight`.
        Each dict must contain ``perf_weight``, ``accuracy``, ``avg_cost``,
        and (for the first entry at least) ``baselines``.
    save_path : str, optional
        File path for the saved figure.
    show : bool
        Call ``plt.show()`` after rendering (useful in notebooks).
    plot_random: bool
        Show the random routing results

    Returns
    -------
    str or None
        Path to saved figure, or *None* if nothing was saved.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError as exc:
        logger.warning(
            "[plot_weight_ablation_pareto] matplotlib/numpy not available: %s",
            exc,
        )
        return None

    if not sweep_results:
        logger.warning("[plot_weight_ablation_pareto] Empty sweep_results.")
        return None

    # ------------------------------------------------------------------
    # Collect router sweep points
    # ------------------------------------------------------------------
    pw_vals = [r["perf_weight"] for r in sweep_results]
    router_accs = [r["accuracy"] for r in sweep_results]
    router_costs = [r["avg_cost"] for r in sweep_results]

    # ------------------------------------------------------------------
    # Collect baseline models (from first entry — they don't change)
    # ------------------------------------------------------------------
    first_baselines = sweep_results[0].get("baselines", {})
    per_model = first_baselines.get("per_model", {})
    pareto_base_names: List[str] = sweep_results[0].get(
        "pareto_optimal_base", []
    )

    base_names = list(per_model.keys())
    base_accs = [per_model[m]["accuracy"] for m in base_names]
    base_costs = [per_model[m]["avg_cost"] for m in base_names]

    # ------------------------------------------------------------------
    # Compute combined Pareto frontier (router sweep + base models)
    # ------------------------------------------------------------------
    all_points: List[Tuple[float, float]] = []
    all_labels: List[str] = []

    for i, r in enumerate(sweep_results):
        all_points.append((r["avg_cost"], r["accuracy"]))
        all_labels.append(f"pw={r['perf_weight']:.2f}")

    for m in base_names:
        all_points.append((per_model[m]["avg_cost"], per_model[m]["accuracy"]))
        all_labels.append(m)

    combined_pareto = _find_pareto_frontier(
        all_points, maximize_y=True, minimize_x=True
    )

    # ------------------------------------------------------------------
    # Academic figure setup
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "DejaVu Sans", "Liberation Sans",
            "Bitstream Vera Sans", "sans-serif",
        ],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # ------------------------------------------------------------------
    # 1) Non-Pareto base models (grey)
    # ------------------------------------------------------------------
    pareto_base_set = set(pareto_base_names)
    base_label_added = False
    for i, m in enumerate(base_names):
        if m in pareto_base_set:
            continue  # drawn separately below
        ax.scatter(
            base_costs[i],
            base_accs[i],
            c=ACADEMIC_COLORS["baseline"],
            s=80,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
            label="Base models" if not base_label_added else None,
        )
        base_label_added = True
        lbl = m if len(m) <= 22 else m[:19] + "..."
        ax.annotate(
            lbl,
            (base_costs[i], base_accs[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=6,
            alpha=0.7,
        )

    # ------------------------------------------------------------------
    # 2) Pareto-optimal base models (distinct markers)
    # ------------------------------------------------------------------
    for rank, m in enumerate(pareto_base_names):
        if m not in per_model:
            continue
        marker = BASELINE_MARKERS[rank % len(BASELINE_MARKERS)]
        mlbl = m if len(m) <= 22 else m[:19] + "..."
        ax.scatter(
            per_model[m]["avg_cost"],
            per_model[m]["accuracy"],
            marker=marker,
            s=180,
            color=ACADEMIC_COLORS["primary"],
            edgecolors="black",
            linewidths=1,
            alpha=0.9,
            zorder=4,
            label=f"Pareto: {mlbl}",
        )
        ax.annotate(
            mlbl,
            (per_model[m]["avg_cost"], per_model[m]["accuracy"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=6,
            alpha=0.8,
        )

    # ------------------------------------------------------------------
    # 3) Router sweep points (colour-coded by perf_weight)
    # ------------------------------------------------------------------
    pw_arr = np.array(pw_vals)
    sc = ax.scatter(
        router_costs,
        router_accs,
        c=pw_arr,
        cmap="viridis",
        s=120,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="Router (weight sweep)",
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("perf_weight", fontweight="bold")

    # Connect sweep points with a trade-off curve (sorted by cost)
    sorted_idx = np.argsort(router_costs)
    ax.plot(
        np.array(router_costs)[sorted_idx],
        np.array(router_accs)[sorted_idx],
        "-",
        color=ACADEMIC_COLORS["tertiary"],
        linewidth=2,
        alpha=0.6,
        zorder=4,
        label="Weight trade-off curve",
    )

    # Annotate each router point with its perf_weight
    for i in range(len(pw_vals)):
        ax.annotate(
            f"{pw_vals[i]:.3g}",
            (router_costs[i], router_accs[i]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=6,
            fontweight="bold",
            color=ACADEMIC_COLORS["tertiary"],
            alpha=0.9,
        )

    # Annotate the highest accuracy among points that are with zero cost
    zero_cost_points = [(router_costs[i], router_accs[i], pw_vals[i], i) for i in range(len(router_costs)) if router_costs[i] < 1e-7]
    logger.info(f"{len(zero_cost_points)} zero_cost_points")
    if zero_cost_points:
        best_zero_cost = max(zero_cost_points, key=lambda p: p[1])
        _, best_acc, best_pw, best_idx = best_zero_cost
        best_zero_counts = sweep_results[best_idx].get("llm_request_counts", {})
        if best_zero_counts:
            total_requests = sum(best_zero_counts.values())
            if total_requests > 0:
                llm_pct_text = "Routing breakdown:\n- " + "\n- ".join(
                    f"{m.lower()}: {c/total_requests*100:.1f}%" for m, c in sorted(best_zero_counts.items(), key=lambda x: x[1], reverse=True) if c > 0
                )
                annotation_text = f"Cost saving 100%\nAccuracy up to {best_acc*100:.1f}%\n{llm_pct_text}"
            else:
                annotation_text = f"Cost saving 100%\nAccuracy up to {best_acc*100:.1f}%"
        else:
            annotation_text = f"Highest acc {best_acc*100:.1f}% with zero cost"
        ax.plot(
            0, best_acc,
            marker="*", markersize=18, color=ACADEMIC_COLORS["primary"],
            markeredgecolor="black", markeredgewidth=0.5, zorder=7,
        )
        ax.annotate(
            annotation_text,
            xy=(0, best_acc),
            xytext=(70, -45),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=ACADEMIC_COLORS["primary"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ACADEMIC_COLORS["primary"], alpha=0.9),
            arrowprops=dict(arrowstyle="-", color=ACADEMIC_COLORS["primary"], linewidth=0.8),
        )


    # Annotate the highest accuracy point achieved by the routing algorithm
    max_idx = int(np.argmax(router_accs))
    max_acc = router_accs[max_idx]
    max_cost = router_costs[max_idx]
    max_pw = pw_vals[max_idx]
    max_counts = sweep_results[max_idx].get("llm_request_counts", {})
    if max_counts:
        total_requests = sum(max_counts.values())
        if total_requests > 0:
            llm_pct_text = "Routing breakdown\n- "+"\n- ".join(
                f"{m.lower()}: {c/total_requests*100:.1f}%" for m, c in sorted(max_counts.items(), key=lambda x: x[1], reverse=True) if c > 0
            )
            annotation_text = f"Highest Acc: {max_acc*100:.1f}%\n{llm_pct_text}"
        else:
            annotation_text = f"Highest Acc: {max_acc*100:.1f}%"
    else:
        annotation_text = f"Highest Acc: {max_acc*100:.1f}%"
    ax.plot(
            max_cost, max_acc,
            marker="*", markersize=18, color=ACADEMIC_COLORS["quaternary"],
            markeredgecolor="black", markeredgewidth=0.5, zorder=7,
        )
    ax.annotate(
        annotation_text,
        (max_cost, max_acc),
        textcoords="offset points",
        xytext=(20, -20),
        fontsize=8,
        fontweight="bold",
        color=ACADEMIC_COLORS["quaternary"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ACADEMIC_COLORS["quaternary"], alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=ACADEMIC_COLORS["quaternary"], linewidth=0.8),
        zorder=7,
    )

    # ------------------------------------------------------------------
    # 4) Combined Pareto frontier line
    # ------------------------------------------------------------------
    if combined_pareto:
        cp_sorted = sorted(combined_pareto, key=lambda p: p[0])
        cpx, cpy = zip(*cp_sorted)
        ax.plot(
            cpx,
            cpy,
            "o-",
            color=ACADEMIC_COLORS["quaternary"],
            linewidth=2.5,
            markersize=5,
            alpha=0.8,
            zorder=6,
            label="Combined Pareto frontier",
        )

    # ------------------------------------------------------------------
    # 5) Random reference line
    # ------------------------------------------------------------------
    if plot_random:
        random_acc = first_baselines.get("random", {}).get("accuracy")
        if random_acc is not None:
            ax.axhline(
                random_acc,
                color=ACADEMIC_COLORS["secondary"],
                linestyle=":",
                linewidth=1,
                alpha=0.6,
                label=f"Random ({random_acc*100:.1f}%)",
            )

    # ------------------------------------------------------------------
    # 6) Best Single accuracy reference line
    # ------------------------------------------------------------------
    best_single_acc = first_baselines.get("best_single", {}).get("accuracy")
    best_single_cost = first_baselines.get("best_single", {}).get("avg_cost", 0.0)
    if best_single_acc is not None:
        ax.axhline(
            best_single_acc,
            color=ACADEMIC_COLORS["success"],
            linestyle="-.",
            linewidth=1.5,
            alpha=0.7,
            label=f"Best Single ({best_single_acc*100:.1f}%)",
        )

        # Find intersection of best_single_acc horizontal line with combined Pareto
        if combined_pareto and best_single_cost > 0:
            cp_sorted = sorted(combined_pareto, key=lambda p: p[0])
            intersect_cost = None
            for j in range(len(cp_sorted) - 1):
                c_low, a_low = cp_sorted[j]
                c_high, a_high = cp_sorted[j + 1]
                if a_low <= best_single_acc <= a_high:
                    m = (a_high - a_low) / (c_high - c_low) if c_high != c_low else 0
                    if m != 0:
                        intersect_cost = c_low + (best_single_acc - a_low) / m
                    break
            if intersect_cost is None:
                for c, a in cp_sorted:
                    if a >= best_single_acc:
                        intersect_cost = c
                        break
            if intersect_cost is not None:
                ax.axvline(
                    intersect_cost,
                    color=ACADEMIC_COLORS["success"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=3,
                )
                ax.plot(
                    intersect_cost, best_single_acc,
                    marker="*", markersize=15, color=ACADEMIC_COLORS["success"],
                    markeredgecolor="black", markeredgewidth=0.5, zorder=6,
                )
                saving_pct = ((best_single_cost - intersect_cost) / best_single_cost) * 100
                ax.annotate(
                    f"Cost saving: {saving_pct:.1f}%\n(vs best single)",
                    xy=(intersect_cost, best_single_acc),
                    xytext=(45, -25),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color=ACADEMIC_COLORS["success"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ACADEMIC_COLORS["success"], alpha=0.8),
                    arrowprops=dict(arrowstyle="-", color=ACADEMIC_COLORS["success"], linewidth=0.8),
                )

        # 95% of best_single_acc
        if best_single_acc is not None and with95:
            target_95 = best_single_acc * 0.95
            ax.axhline(
                target_95,
                color=ACADEMIC_COLORS["warning"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"95% Best Single ({target_95*100:.1f}%)",
            )
            if combined_pareto and best_single_cost > 0:
                cp_sorted_95 = sorted(combined_pareto, key=lambda p: p[0])
                intersect_95 = None
                for j in range(len(cp_sorted_95) - 1):
                    c_low, a_low = cp_sorted_95[j]
                    c_high, a_high = cp_sorted_95[j + 1]
                    if a_low <= target_95 <= a_high:
                        m = (a_high - a_low) / (c_high - c_low) if c_high != c_low else 0
                        if m != 0:
                            intersect_95 = c_low + (target_95 - a_low) / m
                        break
                if intersect_95 is None:
                    for c, a in cp_sorted_95:
                        if a >= target_95:
                            intersect_95 = c
                            break
                if intersect_95 is not None:
                    ax.axvline(
                        intersect_95,
                        color=ACADEMIC_COLORS["warning"],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3,
                    )
                    ax.plot(
                        intersect_95, target_95,
                        marker="*", markersize=15, color=ACADEMIC_COLORS["warning"],
                        markeredgecolor="black", markeredgewidth=0.5, zorder=6,
                    )
                    saving_95 = ((best_single_cost - intersect_95) / best_single_cost) * 100
                    ax.annotate(
                        f"Cost saving: {saving_95:.1f}%\n(vs 95% best single)",
                        xy=(intersect_95, target_95),
                        xytext=(15, -25),
                        textcoords="offset points",
                        fontsize=8,
                        fontweight="bold",
                        color=ACADEMIC_COLORS["warning"],
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ACADEMIC_COLORS["warning"], alpha=0.8),
                        arrowprops=dict(arrowstyle="-", color=ACADEMIC_COLORS["warning"], linewidth=0.8),
                    )

        # 90% of best_single_acc
        if best_single_acc is not None and with90:
            target_90 = best_single_acc * 0.90
            ax.axhline(
                target_90,
                color=ACADEMIC_COLORS["secondary"],
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=f"90% Best Single ({target_90*100:.1f}%)",
            )
            if combined_pareto and best_single_cost > 0:
                cp_sorted_90 = sorted(combined_pareto, key=lambda p: p[0])
                intersect_90 = None
                for j in range(len(cp_sorted_90) - 1):
                    c_low, a_low = cp_sorted_90[j]
                    c_high, a_high = cp_sorted_90[j + 1]
                    if a_low <= target_90 <= a_high:
                        m = (a_high - a_low) / (c_high - c_low) if c_high != c_low else 0
                        if m != 0:
                            intersect_90 = c_low + (target_90 - a_low) / m
                        break
                if intersect_90 is None:
                    for c, a in cp_sorted_90:
                        if a >= target_90:
                            intersect_90 = c
                            break
                if intersect_90 is not None:
                    ax.axvline(
                        intersect_90,
                        color=ACADEMIC_COLORS["secondary"],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3,
                    )
                    ax.plot(
                        intersect_90, target_90,
                        marker="*", markersize=15, color=ACADEMIC_COLORS["secondary"],
                        markeredgecolor="black", markeredgewidth=0.5, zorder=6,
                    )
                    saving_90 = ((best_single_cost - intersect_90) / best_single_cost) * 100
                    ax.annotate(
                        f"Cost saving: {saving_90:.1f}%\n(vs 90% best single)",
                        xy=(intersect_90, target_90),
                        xytext=(15, -25),
                        textcoords="offset points",
                        fontsize=8,
                        fontweight="bold",
                        color=ACADEMIC_COLORS["secondary"],
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ACADEMIC_COLORS["secondary"], alpha=0.8),
                        arrowprops=dict(arrowstyle="-", color=ACADEMIC_COLORS["secondary"], linewidth=0.8),
                    )
    # ------------------------------------------------------------------
    # Axes, title, legend
    # ------------------------------------------------------------------
    ax.set_xlabel("Average Per-Query Cost ($)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(
        "Performance\u2013Cost Trade-off: perf_weight Sweep\n"
        "(Router operating points + base-model baselines)",
        fontweight="bold",
        pad=20,
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Secondary x-axis: cost saving % vs best single model
    # ------------------------------------------------------------------
    if best_single_cost > 0:
        _bsc2 = float(best_single_cost)

        def _cost_to_saving2(c):  # type: ignore[override]
            return ((_bsc2 - np.asarray(c)) / _bsc2) * 100

        def _saving_to_cost2(s):  # type: ignore[override]
            return _bsc2 * (1.0 - np.asarray(s) / 100.0)

        ax2 = ax.secondary_xaxis(
            "top", functions=(_cost_to_saving2, _saving_to_cost2),
        )
        ax2.set_xlabel(
            "Cost Saving vs Best Single Model (%)", fontweight="bold",
        )
        ax2.tick_params(labelsize=8)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, Any] = {}
    unique_handles: list = []
    unique_labels: list = []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = True
            unique_handles.append(h)
            unique_labels.append(lbl)
    ax.legend(
        unique_handles,
        unique_labels,
        bbox_to_anchor=(1.15, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    saved_path = None
    if save_path:
        p = pathlib.Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=300, bbox_inches="tight", facecolor="white")
        saved_path = str(p)
        logger.info(f"[plot_weight_ablation_pareto] Saved -> {saved_path}")

    plt.close(fig)
    return saved_path
