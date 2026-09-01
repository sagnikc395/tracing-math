"""Command-line interface for the CPU-only follow-up."""

from __future__ import annotations

import json
from pathlib import Path

import click

from causal_circuits.followup.config import FollowupConfig


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/experiment2_cpu.yaml"),
    show_default=True,
    help="Follow-up configuration file.",
)
def main(config: Path) -> None:
    """Run the CPU-only follow-up over frozen Experiment 1 outputs."""
    from causal_circuits.followup.analysis import run_followup

    click.echo(json.dumps(run_followup(FollowupConfig.from_yaml(config)), indent=2))


if __name__ == "__main__":
    main()
