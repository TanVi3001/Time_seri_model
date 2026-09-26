"""Shared, timed Keras training with validation-based early stopping."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class TrainingOutcome:
    model: tf.keras.Model
    history: tf.keras.callbacks.History
    best_epoch: int
    epochs_ran: int
    training_seconds: float


def fit_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> TrainingOutcome:
    """Fit one model and return its best validation epoch and elapsed time."""

    if epochs < 1 or batch_size < 1 or patience < 0:
        raise ValueError("epochs and batch_size must be positive; patience cannot be negative")
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("training and validation data must not be empty")
    if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0]:
        raise ValueError("each input window must have exactly one target")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
    )
    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )
    training_seconds = time.perf_counter() - started

    validation_losses = np.asarray(history.history.get("val_loss", []), dtype=float)
    if validation_losses.size == 0 or not np.isfinite(validation_losses).all():
        raise RuntimeError("training did not produce finite validation losses")
    best_epoch = int(np.argmin(validation_losses)) + 1
    epochs_ran = len(history.epoch)
    return TrainingOutcome(
        model=model,
        history=history,
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        training_seconds=float(training_seconds),
    )
