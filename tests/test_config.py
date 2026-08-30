from causal_circuits.config import ExperimentConfig


def test_default_config_is_valid() -> None:
    config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    assert config.model.name == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert config.data.splits == ("gsm8k", "math", "olympiadbench", "omnimath")
    assert 0.0 in config.intervention.alphas
