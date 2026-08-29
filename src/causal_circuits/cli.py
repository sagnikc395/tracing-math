"""Command-line interface for reproducible experiment stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from causal_circuits.analysis import summarize_scores
from causal_circuits.config import ExperimentConfig
from causal_circuits.data import load_dms, validate_against_sequence
from causal_circuits.models import HuggingFaceESM2
from causal_circuits.scoring import score_dms


def _read_fasta(path: str | Path) -> str:
    lines = Path(path).read_text().splitlines()
    sequence = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    if not sequence:
        raise ValueError(f"No sequence found in {path}")
    return sequence.upper()


def _validate_config(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    print(f"Configuration valid: {config.data.assay} with {config.model.name}")


def _prepare_dms(args: argparse.Namespace) -> None:
    frame = load_dms(
        args.input,
        mutation_column=args.mutation_column,
        fitness_column=args.fitness_column,
        directionality=args.directionality,
    )
    if args.wild_type_fasta:
        validate_against_sequence(frame, _read_fasta(args.wild_type_fasta))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    count = frame["position"].nunique()
    print(f"Prepared {len(frame)} variants across {count} positions -> {output}")


def _score(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    frame = pd.read_csv(config.data.path)
    sequence = _read_fasta(args.wild_type_fasta)
    validate_against_sequence(frame, sequence)
    model = HuggingFaceESM2(config.model.name, config.model.device)
    result = score_dms(model, sequence, frame)
    output = config.scoring.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(json.dumps(summarize_scores(result), indent=2))
    print(f"Scores written to {output}")


def _analyze(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.scores)
    print(json.dumps(summarize_scores(frame), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate an experiment YAML")
    validate.add_argument("--config", default="configs/experiment.yaml")
    validate.set_defaults(handler=_validate_config)

    prepare = subparsers.add_parser("prepare-dms", help="normalize a ProteinGym-style CSV")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--mutation-column", default="mutant")
    prepare.add_argument("--fitness-column", default="DMS_score")
    prepare.add_argument("--directionality", type=int, choices=(-1, 1), default=1)
    prepare.add_argument("--wild-type-fasta")
    prepare.set_defaults(handler=_prepare_dms)

    score = subparsers.add_parser("score", help="run ESM-2 masked-marginal scoring")
    score.add_argument("--config", default="configs/experiment.yaml")
    score.add_argument("--wild-type-fasta", required=True)
    score.set_defaults(handler=_score)

    analyze = subparsers.add_parser("analyze", help="summarize scored DMS variants")
    analyze.add_argument("--scores", required=True)
    analyze.set_defaults(handler=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
