"""Unified command-line interface for the tracing pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import click

from tracing_math.analysis import run_analysis
from tracing_math.config import ProjectConfig
from tracing_math.operations import (
    analyze_boundary_controls,
    analyze_transition_matching,
    extract_boundary_control_shards,
    fit_and_save_conditional_hidden_state,
    fit_and_save_contextual_baseline,
    fit_and_save_transition_probe,
    prepare_counterfactual_template,
    run_counterfactual_patching,
    run_transition_matching_sensitivity,
)
from tracing_math.pipeline import (
    download_data,
    extract_activation_shards,
    fit_and_save_probes,
    plot_artifacts,
    run_and_save_interventions,
)


def _with_workers(config: ProjectConfig, workers: int | None) -> ProjectConfig:
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
    default=Path("configs/project.yaml"),
    show_default=True,
    help="Unified project configuration file.",
)
@click.pass_context
def main(context: click.Context, config: Path) -> None:
    """Run extraction, fitting, intervention, and analysis workflows."""
    context.obj = ProjectConfig.from_yaml(config)


@main.command("validate-config")
@click.pass_obj
def validate_config(config: ProjectConfig) -> None:
    """Validate configuration without loading data or models."""
    click.echo(f"Valid: {config.data.dataset} with {config.model.name}")


@main.command("download-data")
@click.option("--force", is_flag=True, help="Replace the local normalized dataset.")
@click.pass_obj
def download(config: ProjectConfig, force: bool) -> None:
    """Download and normalize the configured dataset."""
    click.echo(f"Dataset saved to {download_data(config, force=force)}")


@main.command("extract-activations")
@click.pass_obj
def extract_activations(config: ProjectConfig) -> None:
    """Extract resumable step-boundary activation shards."""
    click.echo(json.dumps(extract_activation_shards(config), indent=2))


@main.command("fit-probes")
@click.option("--workers", type=int, callback=_validate_workers)
@click.pass_obj
def fit_probes(config: ProjectConfig, workers: int | None) -> None:
    """Fit hidden-state probes and predictive controls."""
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
def run_intervention_stage(config: ProjectConfig) -> None:
    """Run the gated intervention workflow."""
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


@main.command("render-figures")
@click.pass_obj
def render_figures(config: ProjectConfig) -> None:
    """Render figures from existing artifacts."""
    click.echo("\n".join(map(str, plot_artifacts(config))))


@main.command("analyze")
@click.pass_obj
def analyze(config: ProjectConfig) -> None:
    """Run CPU analyses over frozen prediction artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    click.echo(json.dumps(run_analysis(config), indent=2))


@main.command("fit-conditional")
@click.pass_obj
def fit_conditional(config: ProjectConfig) -> None:
    """Fit the conditional hidden-state comparison."""
    click.echo(json.dumps(fit_and_save_conditional_hidden_state(config), indent=2))


@main.command("fit-contextual-baseline")
@click.pass_obj
def fit_contextual_baseline(config: ProjectConfig) -> None:
    """Fit the frozen visible-text contextual baseline."""
    click.echo(json.dumps(fit_and_save_contextual_baseline(config), indent=2))


@main.command("fit-transition")
@click.pass_obj
def fit_transition(config: ProjectConfig) -> None:
    """Fit the matched transition probe."""
    click.echo(json.dumps(fit_and_save_transition_probe(config), indent=2))


@main.command("transition-diagnostics")
@click.pass_obj
def transition_diagnostics(config: ProjectConfig) -> None:
    """Report transition matching reuse and balance diagnostics."""
    click.echo(json.dumps(analyze_transition_matching(config), indent=2))


@main.command("transition-sensitivity")
@click.pass_obj
def transition_sensitivity(config: ProjectConfig) -> None:
    """Run transition matching sensitivity variants."""
    click.echo(json.dumps(run_transition_matching_sensitivity(config), indent=2))


@main.command("extract-boundary-controls")
@click.pass_obj
def extract_boundaries(config: ProjectConfig) -> None:
    """Extract natural-token and marker-token boundary states."""
    click.echo(json.dumps(extract_boundary_control_shards(config), indent=2))


@main.command("analyze-boundary-controls")
@click.pass_obj
def analyze_boundaries(config: ProjectConfig) -> None:
    """Compare natural-token and marker-token boundary controls."""
    click.echo(json.dumps(analyze_boundary_controls(config), indent=2))


@main.command("prepare-counterfactuals")
@click.option("--force", is_flag=True, help="Replace the existing template.")
@click.pass_obj
def prepare_counterfactuals(config: ProjectConfig, force: bool) -> None:
    """Create a reviewable counterfactual annotation template."""
    click.echo(str(prepare_counterfactual_template(config, force=force)))


@main.command("run-counterfactuals")
@click.pass_obj
def patch_counterfactuals(config: ProjectConfig) -> None:
    """Run verified counterfactual activation patching."""
    click.echo(json.dumps(run_counterfactual_patching(config), indent=2))


@main.command("run-all")
@click.option("--skip-interventions", is_flag=True)
@click.option("--skip-analysis", is_flag=True)
@click.option("--workers", type=int, callback=_validate_workers)
@click.pass_obj
def run_all(
    config: ProjectConfig,
    skip_interventions: bool,
    skip_analysis: bool,
    workers: int | None,
) -> None:
    """Run the core pipeline and optional CPU analysis in order."""
    config = _with_workers(config, workers)
    click.echo(f"Dataset saved to {download_data(config)}")
    click.echo(json.dumps(extract_activation_shards(config), indent=2))
    fit_and_save_probes(config)
    if not skip_interventions:
        run_and_save_interventions(config)
    for path in plot_artifacts(config):
        click.echo(f"Figure: {path}")
    if not skip_analysis:
        click.echo(json.dumps(run_analysis(config), indent=2))


if __name__ == "__main__":
    main()
