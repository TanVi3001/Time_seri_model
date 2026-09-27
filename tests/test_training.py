from __future__ import annotations

import numpy as np

from copper_forecasting.models import build_one_step_model
from copper_forecasting.training import fit_model


def test_fit_model_records_best_epoch_and_epochs_ran() -> None:
    X_train = np.arange(48, dtype=np.float32).reshape(16, 3, 1) / 48.0
    y_train = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(-1, 1)
    X_validation = np.arange(12, dtype=np.float32).reshape(4, 3, 1) / 12.0
    y_validation = np.linspace(0.1, 0.9, 4, dtype=np.float32).reshape(-1, 1)
    model = build_one_step_model("GRU", input_shape=(3, 1))

    result = fit_model(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=5,
        batch_size=4,
        patience=2,
    )

    validation_losses = result.history.history["val_loss"]
    assert result.model is model
    assert result.best_epoch == int(np.argmin(validation_losses)) + 1
    assert result.epochs_ran == len(validation_losses)
    assert 1 <= result.best_epoch <= result.epochs_ran <= 5
    assert result.training_seconds >= 0.0
