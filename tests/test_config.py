from causal_circuits.config import ExperimentConfig


def test_default_config_is_valid() -> None:
    config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    assert config.model.name == "facebook/esm2_t6_8M_UR50D"
    assert config.circuit.top_k_fractions[0] == 0.001
