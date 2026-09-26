"""Shared Keras model factories for the forecasting experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _validate_input_shape(input_shape: Sequence[int]) -> tuple[int, int]:
    if len(input_shape) != 2:
        raise ValueError("input_shape must contain (lookback, feature_count)")
    if any(not isinstance(dim, (int, np.integer)) or dim <= 0 for dim in input_shape):
        raise ValueError("input_shape dimensions must be positive integers")
    return int(input_shape[0]), int(input_shape[1])


def _validate_learning_rate(learning_rate: float) -> float:
    try:
        rate = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("learning_rate must be a finite positive number") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be a finite positive number")
    return rate


def _compile(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def build_one_step_model(
    model_name: str,
    input_shape: tuple[int, int],
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build a 32-unit recurrent model with one scalar output."""

    shape = _validate_input_shape(input_shape)
    rate = _validate_learning_rate(learning_rate)
    normalized_name = model_name.strip().lower().replace("-", "")
    recurrent_layers = {
        "simplernn": lambda: tf.keras.layers.SimpleRNN(32),
        "lstm": lambda: tf.keras.layers.LSTM(32),
        "gru": lambda: tf.keras.layers.GRU(32),
        "bilstm": lambda: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    }
    if normalized_name not in recurrent_layers:
        raise ValueError(
            "model_name must be one of SimpleRNN, LSTM, GRU, or Bi-LSTM"
        )

    inputs = tf.keras.Input(shape=shape)
    hidden = recurrent_layers[normalized_name]()(inputs)
    outputs = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{normalized_name}_one_step")
    return _compile(model, rate)


def build_direct_gru(
    input_shape: tuple[int, int],
    horizon: int = 5,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build a 32-unit GRU that predicts the full horizon in one pass."""

    shape = _validate_input_shape(input_shape)
    rate = _validate_learning_rate(learning_rate)
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    inputs = tf.keras.Input(shape=shape)
    hidden = tf.keras.layers.GRU(32)(inputs)
    outputs = tf.keras.layers.Dense(int(horizon))(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_direct_multi_step")
    return _compile(model, rate)
