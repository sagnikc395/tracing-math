"""Contract tests for the single all-experiments notebook."""

import json
from pathlib import Path


def test_notebook_runs_all_experiments_in_order() -> None:
    notebook = json.loads(Path("notebooks/experiment.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    experiment1 = source.index('run_stage("extract-activations")')
    experiment2 = source.index('"tracing_math.followup.cli"')
    experiment3 = source.index('extended_command("fit-transition-probe")')

    assert experiment1 < experiment2 < experiment3
