# Copper Forecasting GRU, Bi-LSTM, IMS and DMS Design

**Status:** Draft for review  
**Date:** 2026-09-26

## Goal

Provide a reproducible copper-price forecasting demo that compares RNN, LSTM, GRU and Bi-LSTM under one shared one-step protocol, then compares iterative multi-step (IMS) with direct multi-output (DMS) forecasts under a separate copper-only protocol. Save reproducible run details, predictions, metrics and figures, and use those artifacts to update the existing Notion project.

## Current repository and data

The repository contains Investing.com exports in `data/raw_investing/`, a merge script, and `data/copper_investing_2006_2026.csv`. The canonical CSV has 4,658 complete rows from 2006-01-26 through 2026-09-25, with copper, WTI, gold, silver and five other aligned variables. The existing RNN and LSTM notebooks use a 22-observation window, only WTI/gold/silver inputs, a different split, and paper-inspired stacked widths. Their existing scores therefore cannot be compared directly with the new protocol.

The selected CSV is a refreshed Investing.com dataset, not the exact 1990-2009 data used in Chen et al. (2023). The project will document this difference and will not claim an exact reproduction of the paper or its simulated-annealing search. The raw exports and `merge_investing_data.py` remain the source for rebuilding the local dataset.

## Architecture

Add a small shared Python package, `copper_forecasting`, for data preparation, metrics, model construction, experiment execution and artifact writing. A top-level runner will expose the experiments through a reproducible command. It will use TensorFlow/Keras and the existing project data rather than duplicating the pipeline in separate notebooks. The current notebooks remain available as earlier paper-inspired experiments.

The runner will write run metadata, test predictions, aggregate metrics and plots beneath `results/`. Each run records the dataset path and date range, selected columns, lookback and horizon, chronological split dates, scaler, model configuration, seed, library versions, parameter count, best epoch, training time, inference time and metric values.

## Shared one-step architecture comparison

- **Target:** `closed_copper_price`.
- **Inputs:** the past 30 observations of `closed_copper_price`, `close_wti_oil`, `close_gold` and `close_silver`.
- **Forecast:** the next copper price only.
- **Split:** chronological 80% train, 10% validation and 10% test, based on raw observation positions before windows are formed.
- **Scaling:** fit MinMax scalers using train rows only; reuse them for validation and test. Inverse-transform copper forecasts and labels before reporting price metrics.
- **Windowing:** every input contains only observations through the forecast origin. Validation and test windows may use earlier observed history as context, but never later values.
- **Models:** `SimpleRNN(32) -> Dense(1)`, `LSTM(32) -> Dense(1)`, `GRU(32) -> Dense(1)`, and `Bidirectional(LSTM(32)) -> Dense(1)`.
- **Training:** Adam with learning rate 0.001, MSE loss, batch size 32, at most 100 epochs, early stopping on `val_loss` with patience 10 and best-weight restoration. Use a fixed, recorded seed (default 42); support repeated runs with additional seeds.
- **Measurements:** MSE, RMSE, MAE and MAPE on the same held-out test dates, parameter count, best epoch, training time and inference time. RMSE and MAE are the primary accuracy measures. MAPE is undefined and reported as unavailable when any actual value is zero or within the documented tolerance.

The one-step model comparison is its own experiment. It will not be ranked against the copper-only multi-step experiment because the input features and forecast task differ.

## Multi-step IMS and DMS comparison

- **Inputs:** copper history only, with a 30-observation lookback.
- **Horizon:** five trading observations.
- **Split and scaling:** use the same chronological 80/10/10 procedure, fitting the copper scaler on train rows only.
- **Shared architecture:** GRU with 32 units, so the experiment isolates forecasting strategy.
- **IMS:** train a one-step GRU. At each test origin, forecast one step, append that prediction to the input window, and repeat through step five. Never insert actual values from within the five-step forecast horizon.
- **DMS:** train one GRU to output the full five-value vector `[t+1, ..., t+5]` directly.
- **Evaluation origins:** use the same origins for both strategies, beginning immediately before the first test target and ending at the last origin whose complete five-value horizon remains in test. Every forecast uses information available at its origin only.
- **Reporting:** inverse-transform forecasts, report MAE and RMSE for each lead time and across all origin/lead pairs, and record MSE/MAPE where defined. Save an error-by-lead plot plus training and inference cost.

The direct strategy in this project specifically means one multi-output model. A separate model per lead time is outside the required implementation and can be documented as a possible extension.

## Notion updates

After the implementation has produced verified results, update T4, T5 and T6 with the shared protocol, explanation, code location and relevant results. Update T1 with the reusable metric implementation and MAPE rule; update Dataset chính with the existing data range, columns, source, row count and missing-value handling; add one Experiment Results row per completed run; and align the workspace IMS/DMS notes with the implemented definitions. Mark tasks Done only when the corresponding code has run and its test metrics and artifacts are available. If the local environment prevents model execution, record that blocker and leave the relevant task In Progress.

## Acceptance criteria

1. The documented dataset range, row count, columns and preprocessing match the checked-in CSV and raw-data merge process.
2. A single runner prepares the chronological split and train-only scalers without future leakage.
3. RNN, LSTM, GRU and Bi-LSTM use identical one-step input windows, target dates, split, metrics and training settings.
4. IMS and DMS use the same copper-only test origins and five-step horizon; no actual value inside a forecast horizon is fed back to either strategy.
5. Metrics are calculated after inverse transformation. MAPE is not silently calculated for zero or near-zero actuals.
6. Each run saves enough configuration and environment metadata to reproduce it, along with predictions and metrics; comparison figures are saved as files.
7. The shared metric, split/windowing, IMS/DMS origin and forecast behavior, and a small end-to-end runner path are verified.
8. Notion content, result rows and task states agree with the artifacts actually produced.

## References

- Chen et al. (2023), *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*, PLOS ONE 18(10), e0285631: https://doi.org/10.1371/journal.pone.0285631
- Investing.com historical data exports recorded in `data/raw_investing/` and merged by `merge_investing_data.py`.
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*: https://arxiv.org/abs/1406.1078
- Schuster and Paliwal (1997), *Bidirectional recurrent neural networks*: https://doi.org/10.1109/78.650093
- Ben Taieb et al. (2012), *A review and comparison of strategies for multi-step ahead time series forecasting*: https://proceedings.mlr.press/v22/taieb12.html
