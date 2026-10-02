r"""Tests for the appa.utils.io module."""

import os
import torch

from appa.utils.io import safe_load, safe_save


def test_safe_save_load():
    # Create a dummy model
    model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10))

    tmp_file_name = "model_test_tmp.pth"
    tmp_file_name_prev = tmp_file_name.replace(".pth", ".prev.pth")

    weights = lambda m: m.state_dict()["0.weight"]

    safe_save(model.state_dict(), tmp_file_name)

    # Check the model was correctly saved.
    assert os.path.exists(tmp_file_name)

    model_loaded = safe_load(tmp_file_name)

    # Check the model is correctly loaded.
    assert torch.allclose(weights(model), model_loaded["0.weight"])

    prev_weights = weights(model).clone()
    weights(model).add_(1)

    safe_save(model.state_dict(), tmp_file_name)

    model_loaded = safe_load(tmp_file_name)

    # Check the model is correctly loaded.
    assert torch.allclose(weights(model), model_loaded["0.weight"])

    model_prev_loaded = safe_load(tmp_file_name_prev)

    # Check the previous model was correctly cycled.
    assert torch.allclose(prev_weights, model_prev_loaded["0.weight"])

    # Check if errors are correctly detected to load the previous model
    safe_save(model.state_dict(), tmp_file_name)
    safe_save(model.state_dict(), tmp_file_name)  # Save twice to overwrite
    os.remove(tmp_file_name)
    model_loaded = safe_load(tmp_file_name)
    assert torch.allclose(weights(model), model_loaded["0.weight"])

    os.remove(tmp_file_name_prev)
