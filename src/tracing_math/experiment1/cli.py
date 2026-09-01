"""Command-line interface for Experiment 1."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import click

from tracing_math.experiment1.config import ExperimentConfig
from tracing_math.experiment1.pipeline import (
    download_data,
    extract_activation_shards,
    fit_and_save_probes,
    plot_artifacts,
    run_and_save_interventions,
)


def _with_workers(config: ExperimentConfig, workers: int | None) -> ExperimentConfig:
    if workers is None:
        return config
    analysis = replace(config.analysis, workers=workers)
    updated = replace(config, analysis=analysis)
    updated.validate()
    return updated


def _validate_workers(
    _context: click.Context, _parameter: click.Parameter, value: int | None
) -> int | None:
    if value is not None and (value == 0 or value < -1):
        raise click.BadParameter("must be positive or -1 for all available CPUs")
    return value


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/experiment1.yaml"),
    show_default=True,
    help="Experiment configuration file.",
)
@click.pass_context
def main(context: click.Context, config: Path) -> None:
    """Run the ProcessBench mechanistic experiment."""
    context.obj = ExperimentConfig.from_yaml(config)


@main.command("validate-config")
@click.pass_obj
def validate_config(config: ExperimentConfig) -> None:
    """Validate the scientific configuration without running the model."""
    click.echo(f"Valid: {config.data.dataset} with {config.model.name}")


@main.command("download-data")
@click.option("--force", is_flag=True, help="Replace the local ProcessBench JSONL.")
@click.pass_obj
def download(config: ExperimentConfig, force: bool) -> None:
    """Download and normalize the official ProcessBench dataset."""
    output = download_data(config, force=force)
    click.echo(f"ProcessBench saved to {output}")


@main.command("extract-activations")
@click.pass_obj
def extract_activations(config: ExperimentConfig) -> None:
    """Extract resumable step-boundary activation shards."""
    click.echo(json.dumps(extract_activation_shards(config), indent=2))


@main.command("fit-probes")
@click.option(
    "--workers",
    type=int,
    callback=_validate_workers,
    help="Parallel CPU workers for independent layer fits and bootstrap samples (-1: all).",
)
@click.pass_obj
def fit_probes(config: ExperimentConfig, workers: int | None) -> None:
    """Fit probes and all non-causal controls."""
    config = _with_workers(config, workers)
    result = fit_and_save_probes(config)
    click.echo(
        json.dumps(
            {
                "selected_layer": result.selected_layer,
                "selected_intervention_layer": result.selected_intervention_layer,
                "output_dir": str(config.extraction.output_dir / "probes"),
            },
            indent=2,
        )
    )


@main.command("run-interventions")
@click.pass_obj
def run_intervention_stage(config: ExperimentConfig) -> None:
    """Run held-out causal dose-response experiments."""
    result = run_and_save_interventions(config)
    click.echo(
        json.dumps(
            {
                "intervention_rows": len(result),
                "output_dir": str(config.extraction.output_dir / "interventions"),
            },
            indent=2,
        )
    )


@main.command("plot")
@click.pass_obj
def plot(config: ExperimentConfig) -> None:
    """Render paper-ready figures from existing artifacts."""
    click.echo("\n".join(map(str, plot_artifacts(config))))


@main.command("run-all")
@click.option("--skip-interventions", is_flag=True, help="Stop after probes and controls.")
@click.option(
    "--workers",
    type=int,
    callback=_validate_workers,
    help="Parallel CPU workers for independent layer fits and bootstrap samples (-1: all).",
)
@click.pass_obj
def run_all(config: ExperimentConfig, skip_interventions: bool, workers: int | None) -> None:
    """Run every Experiment 1 stage in order."""
    config = _with_workers(config, workers)
    output = download_data(config)
    click.echo(f"Data: {output}")
    click.echo(json.dumps(extract_activation_shards(config), indent=2))
    fit_and_save_probes(config)
    if not skip_interventions:
        run_and_save_interventions(config)
    for path in plot_artifacts(config):
        click.echo(f"Figure: {path}")
