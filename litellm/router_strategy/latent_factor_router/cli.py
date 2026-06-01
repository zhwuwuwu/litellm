"""
cli.py — Training entry point for LatentFactorRouter.

Usage
-----
  python -m litellm.router_strategy.latent_factor_router.cli --help
  python -m litellm.router_strategy.latent_factor_router.cli train --help
  python -m litellm.router_strategy.latent_factor_router.cli train \\
      --config path/to/config.yaml \\
      --data   path/to/data.jsonl \\
      --output path/to/artefacts.pkl \\
      --device cpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litellm.router_strategy.latent_factor_router.cli",
        description="LatentFactorRouter training CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    train_p = sub.add_parser("train", help="Train the LatentFactorRouter model")
    train_p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML configuration file.",
    )
    train_p.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help=(
            "Path to the routing data JSONL file. "
            "Overrides data_path.routing_data_all in the YAML."
        ),
    )
    train_p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Path to save the trained artefacts (.pkl). "
            "Overrides model_path.save_model_path in the YAML."
        ),
    )
    train_p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "auto"],
        help="Device for training (default: cpu).",
    )
    return parser


def _patch_cfg(router, data: str | None, output: str | None) -> None:
    """Surgically patch router.cfg in-place so LatentFactorTrainer picks up overrides."""
    if data is not None:
        router.cfg.setdefault("data_path", {})["routing_data_all"] = data
        # Also reload all_routing_data with the new path
        try:
            from litellm.router_strategy._llmrouter import load_jsonl as _load_jsonl
            loaded = _load_jsonl(data)
            router.all_routing_data = loaded if loaded is not None else []
            logging.getLogger(__name__).info(
                "[cli] Reloaded %d rows from --data %s", len(router.all_routing_data), data
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[cli] Could not reload data from %s: %s", data, exc
            )
    if output is not None:
        router.cfg.setdefault("model_path", {})["save_model_path"] = output


def _cmd_train(args: argparse.Namespace) -> int:
    log = logging.getLogger(__name__)

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 1

    if args.data is not None and not os.path.isfile(args.data):
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ imports
    try:
        from litellm.router_strategy.latent_factor_router.router import LatentFactorRouter
    except ImportError as exc:
        print(
            f"ERROR: cannot import LatentFactorRouter — missing dependency: {exc}\n"
            "Install optional deps:  pip install litellm[superclaw_routers]",
            file=sys.stderr,
        )
        return 2

    try:
        from litellm.router_strategy.latent_factor_router.trainer import LatentFactorTrainer
    except ImportError as exc:
        print(
            f"ERROR: cannot import LatentFactorTrainer — missing dependency: {exc}\n"
            "Install optional deps:  pip install litellm[superclaw_routers]",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------------ router
    log.info("[cli] Loading LatentFactorRouter from %s", config_path)
    try:
        router = LatentFactorRouter(yaml_path=config_path)
    except Exception as exc:
        print(f"ERROR: failed to initialise LatentFactorRouter: {exc}", file=sys.stderr)
        return 3

    # Apply --data / --output overrides before constructing trainer
    _patch_cfg(router, args.data, args.output)

    # ------------------------------------------------------------------ trainer
    log.info("[cli] Creating LatentFactorTrainer (device=%s)", args.device)
    try:
        trainer = LatentFactorTrainer(router=router, device=args.device)
    except Exception as exc:
        print(f"ERROR: failed to initialise LatentFactorTrainer: {exc}", file=sys.stderr)
        return 3

    # ------------------------------------------------------------------ train
    log.info("[cli] Starting training …")
    try:
        metrics = trainer.train()
    except RuntimeError as exc:
        print(f"ERROR: training failed: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"ERROR: unexpected error during training: {exc}", file=sys.stderr)
        return 4

    print("\n[cli] Training complete.")
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print(f"  artefacts saved to: {trainer.save_model_path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        sys.exit(_cmd_train(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
