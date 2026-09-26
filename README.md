# Copper price time-series demo: RNN and LSTM

This repository reproduces the copper-price forecasting setup inspired by
Chen et al. (2023), using `SimpleRNN` and `LSTM`.

## Main files

- `RNN_Copper_Paper_2006_2026.ipynb`: SimpleRNN experiment.
- `LSTM_Copper_Paper_2006_2026.ipynb`: LSTM experiment.
- `data/copper_investing_2006_2026.csv`: merged dataset for Kaggle/Colab.
- `data/raw_investing/`: Investing.com CSV exports used to build the merged dataset.
- `merge_investing_data.py`: reproducible merge and validation script.
- `PAPER_DATASET_README.md`: dataset and experiment details.

The canonical dataset contains 4,658 complete trading dates from
2006-01-26 to 2026-09-25. It has nine paper variables plus the `date` column.

## Rebuild the merged dataset

Install the dependencies and run:

```bash
pip install -r requirements-paper.txt
python merge_investing_data.py
```

The script writes `data/copper_investing_2006_2026.csv` and checks missing
values and duplicate dates.

## Run on Kaggle
1. Create a Kaggle Notebook.
2. Upload `data/copper_investing_2006_2026.csv` as an input dataset.
3. Import either notebook and run its cells from top to bottom.
4. For the paper-style comparison, run the final horizon-training cell and
   then the visualization cell below it.
The notebooks use `SimpleRNN` or `LSTM` with the same data split and settings
so the two models can be compared fairly.
