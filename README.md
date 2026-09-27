# Dự báo chuỗi thời gian giá đồng

Repo có hai notebook cũ lấy cảm hứng từ bài báo Chen et al. (2023) và package Python dùng chung cho thí nghiệm GRU, Bi-LSTM, IMS và DMS.

## Dữ liệu

Dataset chuẩn là [`data/copper_investing_2006_2026.csv`](data/copper_investing_2006_2026.csv): 4,658 ngày giao dịch từ 2006-01-26 đến 2026-09-25, gồm ngày và chín chuỗi giá/liên quan. Nguồn là các CSV Investing.com trong `data/raw_investing/`; cách parse, ghép, đơn vị và số liệu kiểm tra được ghi trong [`docs/DATA_CLEANING_LOG.md`](docs/DATA_CLEANING_LOG.md). Chạy `python scripts/merge_investing_data.py` để tái tạo file chuẩn.

Dataset này không phải bản tái lập chính xác bộ 1990–2009, 4,870 hàng của bài báo. Đơn vị giá đồng là USD/pound; WTI USD/barrel; vàng và bạc USD/troy ounce. Các đơn vị không được lưu trong CSV gốc và được đối chiếu từ trang instrument/contract.

## Cài đặt trên Windows

Runner đã được chạy với Python 3.12.10, TensorFlow 2.21 và CPU trên Windows. Dependency được ghim trong `requirements/requirements-win-py312.lock.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\requirements-win-py312.lock.txt
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

## Kết quả thực nghiệm ngày 26-09-2026

Một lần chạy với seed 42; tập test theo thời gian kéo dài từ 2024-07-09 đến 2026-09-25, trên dataset 2006–2026, cho kết quả sau (giá đồng USD/pound):

| Mô hình | MSE | RMSE | MAE | MAPE | Train (s) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| SimpleRNN | 0.430026 | 0.655764 | 0.523579 | 9.119983% | 23.980 | 34 |
| LSTM | 0.315877 | 0.562029 | 0.414590 | 7.145717% | 38.406 | 30 |
| GRU | 0.050938 | 0.225694 | 0.173044 | 3.241153% | 22.252 | 9 |
| Bi-LSTM | 0.332849 | 0.576931 | 0.426178 | 7.347024% | 51.702 | 35 |

GRU có RMSE thấp nhất trong lần chạy này. Kết quả dùng một seed, vì vậy đây là so sánh quan sát được chứ chưa đo độ dao động giữa các lần khởi tạo.

Với 462 test origins và horizon 5 (2,310 dự báo cho mỗi chiến lược), RMSE/MAE theo lead là:

| Lead | IMS RMSE | IMS MAE | DMS RMSE | DMS MAE |
|---:|---:|---:|---:|---:|
| 1 | 0.136772 | 0.094805 | 0.172923 | 0.125267 |
| 2 | 0.189437 | 0.135218 | 0.234678 | 0.171926 |
| 3 | 0.233027 | 0.169301 | 0.256435 | 0.190017 |
| 4 | 0.272330 | 0.202032 | 0.287133 | 0.214027 |
| 5 | 0.306472 | 0.232529 | 0.278796 | 0.203173 |

Gộp mọi origin/lead, IMS đạt RMSE 0.235365 và MAE 0.166777; DMS đạt RMSE 0.249362 và MAE 0.180882. IMS thấp hơn ở lead 1–4; DMS thấp hơn ở lead 5 và có thời gian train/suy luận ngắn hơn trong lần chạy này (20.548/0.446 giây so với 25.573/1.119 giây của IMS). Số tham số lần lượt là 3,525 cho DMS và 3,393 cho IMS.

Artifacts đã lưu tại [`results/copper-forecasting-2026-09-26/`](results/copper-forecasting-2026-09-26/): [bảng one-step](results/copper-forecasting-2026-09-26/one_step_metrics.csv), [metric theo horizon](results/copper-forecasting-2026-09-26/multistep_metrics_by_horizon.csv), [biểu đồ kiến trúc](results/copper-forecasting-2026-09-26/architecture_comparison.png), [biểu đồ sai số horizon](results/copper-forecasting-2026-09-26/multistep_errors.png), [metadata GRU](results/copper-forecasting-2026-09-26/gru_one_step_seed42/metadata.json), [metadata IMS](results/copper-forecasting-2026-09-26/gru_ims_h5_seed42/metadata.json) và [metadata DMS](results/copper-forecasting-2026-09-26/gru_dms_h5_seed42/metadata.json).

Chạy kiểm thử bằng:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`notebooks/RNN_Copper_Paper_2006_2026.ipynb` và `notebooks/LSTM_Copper_Paper_2006_2026.ipynb` vẫn là các thử nghiệm paper-inspired riêng; các thiết lập cũ của chúng được mô tả trong [`docs/PAPER_DATASET_README.md`](docs/PAPER_DATASET_README.md).
