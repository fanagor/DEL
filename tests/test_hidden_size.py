"""Unified model capacity, defaults, and validation."""

import pytest
import torch

from differentiable_exact_learning.config import load_config, validate_config
from differentiable_exact_learning.models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("setting", ["omitted", None, 7])
def test_model_dependent_defaults_and_explicit_width(name, setting):
    model_config = {"name": name}
    if setting != "omitted":
        model_config["hidden_size"] = setting
    model = build_model(model_config, input_size=3, symbol_size=4, motion_size=2, expert_states=11)
    width = 7 if setting == 7 else 11 if name.startswith("DFST") else 64
    assert model.hidden_size == width
    assert not hasattr(model, "state_size")
    if name.startswith("DFST"):
        assert model.A.shape == (3, width, width)
        assert model.h0.shape == (width,)
    else:
        assert model.h0.shape == (1, width)
    symbols, motions, _ = model(torch.tensor([[0, 1, 2]]))
    assert symbols.shape == (1, 3, 4)
    assert motions.shape == (1, 3, 2)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_width_rejected_by_config_and_factory(name, value):
    config = load_config(overrides=[f"model.name={name}"])
    config["model"]["hidden_size"] = value
    with pytest.raises(ValueError, match="hidden_size"):
        validate_config(config)
    with pytest.raises(ValueError, match="hidden_size"):
        build_model(config["model"], 3, 4, 2, 11)


def test_config_only_exposes_hidden_size(tmp_path):
    assert load_config()["model"]["hidden_size"] is None
    assert "state_size" not in load_config()["model"]
    for name in MODEL_NAMES:
        config = load_config(overrides=[f"model.name={name}", "model.hidden_size=null"])
        assert config["model"]["hidden_size"] is None
    with pytest.raises(ValueError, match="Unknown configuration field: model.state_size"):
        load_config(overrides=["model.state_size=8"])
    recipe = tmp_path / "old.yaml"
    recipe.write_text("model:\n  state_size: 8\n")
    with pytest.raises(ValueError, match="Unknown configuration field: model.state_size"):
        load_config(recipe)
