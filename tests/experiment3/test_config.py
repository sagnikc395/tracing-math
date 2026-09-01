from tracing_math.experiment3.config import ExtendedFollowupConfig


def test_extended_followup_config_is_valid() -> None:
    config = ExtendedFollowupConfig.from_yaml("configs/experiment3.yaml")

    assert config.model_name == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert config.counterfactual_template_size == 160
    assert config.c_values == (0.01, 0.1, 1.0, 10.0)

