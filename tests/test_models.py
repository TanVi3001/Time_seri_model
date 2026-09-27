from __future__ import annotations

import pytest
import tensorflow as tf

from copper_forecasting.models import build_direct_gru, build_one_step_model


@pytest.mark.parametrize("model_name", ["SimpleRNN", "LSTM", "GRU", "Bi-LSTM"])
def test_one_step_factories_compile_with_expected_shapes(model_name: str) -> None:
    model = build_one_step_model(model_name, input_shape=(30, 4))

    assert model.input_shape == (None, 30, 4)
    assert model.output_shape == (None, 1)
    assert model.loss == "mse"
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.001)


def test_one_step_factory_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        build_one_step_model("Transformer", input_shape=(30, 4))


def test_direct_gru_outputs_one_value_per_horizon_step() -> None:
    model = build_direct_gru(input_shape=(30, 1), horizon=5)

    assert model.input_shape == (None, 30, 1)
    assert model.output_shape == (None, 5)
    assert model.loss == "mse"
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)


@pytest.mark.parametrize(
    "input_shape",
    [(0, 4), (30,), (30, 4, 1)],
)
def test_one_step_factory_rejects_invalid_input_shape(input_shape) -> None:
    with pytest.raises(ValueError, match="input_shape"):
        build_one_step_model("GRU", input_shape=input_shape)
