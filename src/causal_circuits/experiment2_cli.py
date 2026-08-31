"""Command-line entry point for Experiment 2."""

from __future__ import annotations

import argparse
import json

from causal_circuits.experiment2_analysis import run_marker_robustness
from causal_circuits.experiment2_causal import audit_verdict_readout, run_causal_validation
from causal_circuits.experiment2_config import Experiment2Config
from causal_circuits.experiment2_pipeline import (
    experiment2_run_identity,
    plot_experiment2,
    run_all_experiment2,
    validate_experiment2_inputs,
    write_resolved_config,
)
from causal_circuits.experiment2_runtime import (
    configure_logging,
    run_stage,
    status_report,
    update_stage_progress,
)
from causal_circuits.experiment2_semantic import (
    extract_semantic_activation_shards,
    fit_semantic_boundary_probes,
)


def _load(args: argparse.Namespace) -> Experiment2Config:
    config = Experiment2Config.from_yaml(args.config).with_paths(
        experiment1_dir=args.experiment1_dir,
        output_dir=args.output_dir,
        data_path=args.data_path,
    )
    config.validate()
    return config


def _run(args: argparse.Namespace) -> None:
    config = _load(args)
    if args.command == "status":
        print(json.dumps(status_report(config.output_dir), indent=2))
        return
    write_resolved_config(config)
    log_path = configure_logging(config.output_dir, args.command)
    update_stage_progress(config.output_dir, args.command, log_path=str(log_path))
    if args.command == "run-all":
        result = run_all_experiment2(config, force=args.force)
        print(json.dumps(result, indent=2))
        return
    handlers = {
        "validate-config": validate_experiment2_inputs,
        "analyze-robustness": run_marker_robustness,
        "extract-semantic": extract_semantic_activation_shards,
        "fit-semantic": fit_semantic_boundary_probes,
        "audit-verdict": audit_verdict_readout,
        "causal-validation": run_causal_validation,
        "plot": lambda selected: [str(path) for path in plot_experiment2(selected)],
    }

    def operation():
        validate_experiment2_inputs(config)
        return handlers[args.command](config)

    result = run_stage(
        config.output_dir,
        args.command,
        operation,
        identity=experiment2_run_identity(config),
    )
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment2.yaml")
    parser.add_argument("--experiment1-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-enter stages already marked complete when using run-all",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("validate-config", "validate the full Experiment 1 cache and Experiment 2 paths"),
        ("analyze-robustness", "run marker-state diagnostics and stronger controls"),
        ("extract-semantic", "extract natural step-final activation shards"),
        ("fit-semantic", "fit probes to natural step-final activations"),
        ("audit-verdict", "audit the counterbalanced single-token verdict readout"),
        ("causal-validation", "run gradient alignment and positive-control interventions"),
        ("plot", "render the Experiment 2 summary figure"),
        ("status", "show durable stage and checkpoint progress"),
        ("run-all", "run every Experiment 2 stage in order"),
    ):
        subparsers.add_parser(command, help=help_text).set_defaults(handler=_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
