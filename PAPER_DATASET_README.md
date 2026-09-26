# Copper-price paper reproduction setup

This folder contains paper-specific notebooks for comparing a vanilla RNN (`SimpleRNN`) with an LSTM on the copper-price forecasting problem described in:

Chen et al. (2023), *Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm*, PLOS ONE 18(10): e0285631.

## Time window used for the demo

The paper used 1990-01-01 to 2009-12-31. The demo changes this to the latest 20-year window anchored to the requested date:

- Start: `2006-01-01`
- Inclusive end: `2026-09-26`

The canonical CSV is already filtered to the inclusive window 2006-01-01 through 2026-09-26. Because 2026-09-26 is a Saturday, the latest available trading date is 2026-09-25.

## Paper variables

The paper describes nine aligned series:

- copper closing price (target)
- copper volume
- Dollar Index
- 10-year Treasury yield
- Nasdaq index
- Dow Jones index
- WTI crude-oil price
- gold price
- silver price

Following the paper, the three model inputs are WTI, gold and silver. The notebooks also calculate Spearman correlations so this choice can be checked on the refreshed window.

## Data source used by the notebooks

The notebooks now use the downloaded Investing.com data directly. The nine raw series were merged by date, rows missing any variable were removed, and the result is saved as `data/copper_investing_2006_2026.csv` (4,658 rows, 9 variables plus the date column). No Yahoo Finance API is used.

For Kaggle, upload `copper_investing_2006_2026.csv` as a dataset, then change `LOCAL_CSV_PATH` in the notebook to the path shown under `/kaggle/input/...`. For local Jupyter, keep the file at `data/copper_investing_2006_2026.csv`.

## Reproduction choices

- chronological split; no shuffling
- lookback window: 22 observations, matching the paper model input shape
- default test horizon: 242 observations; optional horizons: 363 and 485
- paper-inspired recurrent widths: 39 then 111
- dense layer: 32 units; dropout: 0.2; batch size: 64; epochs: 100; learning rate: `5e-5`
- scalers are fitted on the pre-test training portion to avoid future leakage
- the paper's simulated-annealing search is not run by default; the notebooks compare RNN and LSTM using the paper's reported LSTM widths and learning rate
