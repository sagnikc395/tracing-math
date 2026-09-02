"""Contract tests for the unified Colab notebook."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/experiment.ipynb")
CONTEXTUAL_NOTEBOOK_PATH = Path("notebooks/contextual_baseline_colab.ipynb")


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_single_notebook_contains_all_completed_workflows() -> None:
    source = _notebook_source()

    for command in (
        "validate-config",
        "download-data",
        "extract-activations",
        "fit-probes",
        "run-interventions",
        "render-figures",
        "analyze",
        "fit-conditional",
        "fit-transition",
        "transition-diagnostics",
        "extract-boundary-controls",
        "analyze-boundary-controls",
        "prepare-counterfactuals",
    ):
        assert f'run_cli("{command}")' in source

    assert "configs/project.yaml" in source
    assert "run-counterfactuals" not in source


def test_notebook_has_no_authentication_or_publication_logic() -> None:
    source = _notebook_source().lower()

    for forbidden in (
        "github_token",
        "hf_token",
        "git_askpass",
        "authenticated_repository",
        "git push",
        "git commit",
        "h secret",
        "publish",
        "experiment 1",
        "experiment 2",
        "experiment 3",
        "follow-up",
    ):
        assert forbidden not in source


def test_notebook_json_is_valid() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    assert all(cell["cell_type"] in {"code", "markdown", "raw"} for cell in notebook["cells"])


def test_contextual_notebook_has_private_clone_and_narrow_result_push() -> None:
    notebook = json.loads(CONTEXTUAL_NOTEBOOK_PATH.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for required in (
        "GITHUB_TOKEN",
        "HF_TOKEN",
        '"git", "clone"',
        "fit-contextual-baseline",
        "PUSH_RESULTS = False",
        '"git", "push"',
        "results",
    ):
        assert required in source
    for forbidden in ("git add -A", "git add ."):
        assert forbidden not in source
    assert notebook["nbformat"] == 4
