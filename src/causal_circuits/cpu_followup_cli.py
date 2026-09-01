"""Command-line entry point for the CPU-only follow-up."""

from __future__ import annotations

import argparse
import json

from causal_circuits.cpu_followup import CPUFollowupConfig, run_cpu_followup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment2_cpu.yaml")
    args = parser.parse_args()
    config = CPUFollowupConfig.from_yaml(args.config)
    print(json.dumps(run_cpu_followup(config), indent=2))


if __name__ == "__main__":
    main()
