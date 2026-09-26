# Dự báo chuỗi thời gian giá đồng

Repo có hai notebook cũ lấy cảm hứng từ bài báo Chen et al. (2023) và package Python dùng chung cho thí nghiệm GRU, Bi-LSTM, IMS và DMS.

## Dữ liệu

Dataset chuẩn là [`data/copper_investing_2006_2026.csv`](data/copper_investing_2006_2026.csv): 4,658 ngày giao dịch từ 2006-01-26 đến 2026-09-25, gồm ngày và chín chuỗi giá/liên quan. Nguồn là các CSV Investing.com trong `data/raw_investing/`; cách parse, ghép, đơn vị và số liệu kiểm tra được ghi trong [`data/DATA_CLEANING_LOG.md`](data/DATA_CLEANING_LOG.md). Chạy `python merge_investing_data.py` để tái tạo file chuẩn.

Dataset này không phải bản tái lập chính xác bộ 1990–2009, 4,870 hàng của bài báo. Đơn vị giá đồng là USD/pound; WTI USD/barrel; vàng và bạc USD/troy ounce. Các đơn vị không được lưu trong CSV gốc và được đối chiếu từ trang instrument/contract.

## Cài đặt trên Windows

Runner đã được chạy với Python 3.12, TensorFlow 2.21 và CPU trên Windows.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Chạy toàn bộ giao thức

```powershell
.\.venv\Scripts\python.exe -m copper_forecasting `
  --dataset data/copper_investing_2006_2026.csv `
  --output-dir results/copper-forecasting-2026-09-26
```

Mặc định, CLI chạy SimpleRNN, LSTM, GRU và Bi-LSTM một bước; tiếp đó chạy GRU IMS và GRU direct multi-output DMS với horizon 5. Có thể đổi seeds, epoch, lookback và horizon bằng CLI:

```powershell
.\.venv\Scripts\python.exe -m copper_forecasting `
  --dataset data/copper_investing_2006_2026.csv `
  --output-dir results/gru-bilstm-only `
  --seeds 42 43 44 `
  --models GRU Bi-LSTM `
  --skip-multistep
```

## Protocol và artifacts

- **So sánh kiến trúc:** lookback 30, đầu vào là lịch sử giá đồng/WTI/vàng/bạc, mục tiêu là giá đồng phiên kế tiếp.
- **So sánh nhiều bước:** chỉ dùng giá đồng, lookback 30, horizon mặc định 5. IMS lặp lại mô hình GRU một bước và cuốn dự báo của chính nó; DMS xuất trực tiếp vector horizon từ một mô hình GRU.
- Cả hai giao thức chia theo thời gian 80/10/10, fit MinMax scaler trên train, dùng Adam 0.001, MSE, batch 32, tối đa 100 epoch, early stopping patience 10 và seed mặc định 42.
- RMSE/MAE/MSE/MAPE được tính sau inverse transform về đơn vị giá gốc. MAPE là `null` nếu có actual bằng hoặc gần 0.
- Không dùng giá tương lai của WTI/vàng/bạc và không đưa actual trong horizon vào IMS/DMS.

Mỗi run có thư mục riêng với `metadata.json`, `metrics.json` và `predictions.csv`; run nhiều bước còn có `metrics_by_horizon.csv`. Thư mục gốc có `one_step_metrics.csv`, `multistep_metrics_by_horizon.csv`, `architecture_comparison.png` và `multistep_errors.png`.

Chạy kiểm thử bằng:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`RNN_Copper_Paper_2006_2026.ipynb` và `LSTM_Copper_Paper_2006_2026.ipynb` vẫn là các thử nghiệm paper-inspired riêng; các thiết lập cũ của chúng được mô tả trong [`PAPER_DATASET_README.md`](PAPER_DATASET_README.md).
