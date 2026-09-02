"""Command-line interface for the exploratory workshop follow-ups."""

from __future__ import annotations

import json
from pathlib import Path

import click

from tracing_math.experiment3.config import ExtendedFollowupConfig


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/experiment3.yaml"),
    show_default=True,
)
@click.pass_context
def main(context: click.Context, config: Path) -> None:
    """Run separate, explicitly post-hoc extensions to Experiment 1."""
    context.obj = ExtendedFollowupConfig.from_yaml(config)


@main.command("validate-config")
@click.pass_obj
def validate_config(config: ExtendedFollowupConfig) -> None:
    click.echo(f"Valid extended follow-up for {config.model_name}")


@main.command("fit-transition-probe")
@click.pass_obj
def fit_transition(config: ExtendedFollowupConfig) -> None:
    from tracing_math.experiment3.pipeline import fit_and_save_transition_probe

    click.echo(json.dumps(fit_and_save_transition_probe(config), indent=2))


@main.command("fit-conditional-hidden-state")
@click.pass_obj
def fit_conditional_hidden_state(config: ExtendedFollowupConfig) -> None:
    """Run the nested nuisance-versus-hidden E1 comparison."""
    from tracing_math.experiment3.pipeline import fit_and_save_conditional_hidden_state

    try:
        result = fit_and_save_conditional_hidden_state(config)
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2))


@main.command("analyze-transition-matching")
@click.pass_obj
def analyze_transition_matching(config: ExtendedFollowupConfig) -> None:
    """Report placebo reuse and covariate balance without activation shards."""
    from tracing_math.experiment3.pipeline import analyze_transition_matching

    click.echo(json.dumps(analyze_transition_matching(config), indent=2))


@main.command("run-transition-matching-sensitivity")
@click.pass_obj
def transition_matching_sensitivity(config: ExtendedFollowupConfig) -> None:
    """Refit the frozen transition protocol with one-to-one and inverse-reuse matching."""
    from tracing_math.experiment3.pipeline import run_transition_matching_sensitivity

    click.echo(json.dumps(run_transition_matching_sensitivity(config), indent=2))


@main.command("extract-boundary-controls")
@click.pass_obj
def extract_boundaries(config: ExtendedFollowupConfig) -> None:
    from tracing_math.experiment3.pipeline import extract_boundary_control_shards

    click.echo(json.dumps(extract_boundary_control_shards(config), indent=2))


@main.command("analyze-boundary-controls")
@click.pass_obj
def analyze_boundaries(config: ExtendedFollowupConfig) -> None:
    from tracing_math.experiment3.pipeline import analyze_boundary_controls

    click.echo(json.dumps(analyze_boundary_controls(config), indent=2))


@main.command("prepare-counterfactual-template")
@click.option("--force", is_flag=True, help="Replace an existing unannotated template.")
@click.pass_obj
def prepare_counterfactuals(config: ExtendedFollowupConfig, force: bool) -> None:
    from tracing_math.experiment3.pipeline import prepare_counterfactual_template

    click.echo(str(prepare_counterfactual_template(config, force=force)))


@main.command("run-counterfactual-patching")
@click.pass_obj
def patch_counterfactuals(config: ExtendedFollowupConfig) -> None:
    from tracing_math.experiment3.pipeline import run_counterfactual_patching

    click.echo(json.dumps(run_counterfactual_patching(config), indent=2))


if __name__ == "__main__":
    main()
