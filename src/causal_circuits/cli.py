"""Command-line harness for the ProcessBench mechanistic experiment."""

from __future__ import annotations

import argparse
import json

from causal_circuits.config import ExperimentConfig
from causal_circuits.pipeline import (
    download_data,
    extract_activation_shards,
    fit_and_save_probes,
    plot_artifacts,
    run_and_save_interventions,
)


def _config(path: str) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path)


def _validate(args: argparse.Namespace) -> None:
    config = _config(args.config)
    print(f"Valid: {config.data.dataset} with {config.model.name}")


def _download(args: argparse.Namespace) -> None:
    config = _config(args.config)
    output = download_data(config, force=args.force)
    print(f"ProcessBench saved to {output}")


def _extract(args: argparse.Namespace) -> None:
    result = extract_activation_shards(_config(args.config))
    print(json.dumps(result, indent=2))


def _fit(args: argparse.Namespace) -> None:
    config = _config(args.config)
    result = fit_and_save_probes(config)
    print(
        json.dumps(
            {
                "selected_layer": result.selected_layer,
                "selected_intervention_layer": result.selected_intervention_layer,
                "output_dir": str(config.extraction.output_dir / "probes"),
            },
            indent=2,
        )
    )


def _intervene(args: argparse.Namespace) -> None:
    config = _config(args.config)
    result = run_and_save_interventions(config)
    print(
        json.dumps(
            {
                "intervention_rows": len(result),
                "output_dir": str(config.extraction.output_dir / "interventions"),
            },
            indent=2,
        )
    )


def _plot(args: argparse.Namespace) -> None:
    outputs = plot_artifacts(_config(args.config))
    print("\n".join(map(str, outputs)))


def _run_all(args: argparse.Namespace) -> None:
    config = _config(args.config)
    output = download_data(config)
    print(f"Data: {output}")
    print(json.dumps(extract_activation_shards(config), indent=2))
    fit_and_save_probes(config)
    if not args.skip_interventions:
        run_and_save_interventions(config)
    for path in plot_artifacts(config):
        print(f"Figure: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate the YAML configuration")
    validate.set_defaults(handler=_validate)

    download = subparsers.add_parser("download-data", help="download official ProcessBench data")
    download.add_argument("--force", action="store_true", help="replace the local JSONL")
    download.set_defaults(handler=_download)

    extract = subparsers.add_parser(
        "extract-activations", help="extract resumable step-boundary activation shards"
    )
    extract.set_defaults(handler=_extract)

    probes = subparsers.add_parser("fit-probes", help="fit probes and all non-causal controls")
    probes.set_defaults(handler=_fit)

    interventions = subparsers.add_parser(
        "run-interventions", help="run held-out causal dose-response experiments"
    )
    interventions.set_defaults(handler=_intervene)

    plots = subparsers.add_parser("plot", help="render paper-ready figures from artifacts")
    plots.set_defaults(handler=_plot)

    run_all = subparsers.add_parser("run-all", help="run the complete pipeline")
    run_all.add_argument(
        "--skip-interventions", action="store_true", help="stop after probes and controls"
    )
    run_all.set_defaults(handler=_run_all)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
