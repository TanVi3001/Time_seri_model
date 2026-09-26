# Kế hoạch triển khai: Copper Forecasting GRU, Bi-LSTM, IMS và DMS

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây runner tái lập được cho so sánh RNN/LSTM/GRU/Bi-LSTM một bước và IMS/DMS năm bước, tạo artifacts, rồi đồng bộ code và kết quả vào Notion.

**Architecture:** Tạo package `copper_forecasting` làm nguồn triển khai chung cho nạp/chia dữ liệu, metric, model, forecast và lưu kết quả. Runner sẽ tạo các lần chạy one-step và multi-step trong các thư mục con của `results/`; notebook RNN/LSTM hiện có được giữ làm ví dụ paper-inspired riêng.

**Tech Stack:** Python, TensorFlow/Keras, pandas, NumPy, scikit-learn, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-copper-forecasting-gru-bilstm-ims-dms-design.md`

## Global Constraints

- Dùng CSV hiện có 4.658 dòng từ 2006-01-26 đến 2026-09-25; ghi rõ đây không phải tái lập chính xác dữ liệu 1990–2009 trong bài báo.
- Architecture comparison dùng mục tiêu `closed_copper_price`, bốn biến đầu vào `[closed_copper_price, close_wti_oil, close_gold, close_silver]`, lookback 30, split theo thời gian 80/10/10 và MinMax scaler fit trên train.
- So sánh `SimpleRNN(32)`, `LSTM(32)`, `GRU(32)` và `Bidirectional(LSTM(32))` với Dense(1), Adam learning rate 0.001, MSE, batch size 32, tối đa 100 epoch, early stopping patience 10, seed mặc định 42.
- IMS/DMS dùng giá đồng một biến, lookback 30, horizon 5, GRU(32), cùng test origins; không đưa giá thực tương lai trong horizon vào input.
- Tính metric trên đơn vị giá gốc sau inverse transform; MAPE không khả dụng nếu actual bằng hoặc gần 0.
- Chỉ đánh dấu task Notion Done khi code tương ứng đã chạy và có metric/artifact kiểm tra được.

## Review Focus

- Ngày trùng, thiếu ngày hoặc giá trị không hợp lệ trong CSV phải bị phát hiện thay vì âm thầm tạo cửa sổ sai. **Test:** `test_read_dataset_rejects_duplicate_dates_and_invalid_values` trong Task 1.
- Giá trị cực trị ở validation/test không được ảnh hưởng scaler hoặc cửa sổ train; mỗi input chỉ dùng hàng tại hoặc trước origin. **Tests:** `test_scalers_fit_train_rows_only`, `test_windows_are_assigned_by_target_indices` và `test_windows_include_only_values_at_or_before_origin` trong Task 1.
- Actual bằng/gần 0, đầu vào không hữu hạn hoặc hai mảng metric lệch độ dài phải được xử lý rõ. **Tests:** `test_mape_is_unavailable_for_near_zero_actuals`, `test_metrics_reject_non_finite_values` và `test_metrics_reject_unequal_lengths` trong Task 2.
- Forecast nhiều bước phải dùng đúng các origin có đủ horizon trong test và IMS phải cuốn dự báo của chính nó. **Tests:** `test_multistep_origins_stay_inside_test` và `test_ims_feeds_predictions_back_into_next_window` trong Task 5.
- Runner phải lưu metadata, dự đoán, metric và số epoch thực chạy. **Tests:** `test_fit_model_records_best_epoch_and_epochs_ran` trong Task 3, `test_one_step_run_writes_metadata_and_predictions` trong Task 4 và `test_cli_smoke_writes_expected_artifacts` trong Task 6.

---

### Task 1: Chuẩn bị dữ liệu, split và cửa sổ

**Files:**
- Create: `copper_forecasting/__init__.py`
- Create: `copper_forecasting/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- `SplitBounds(train_end: int, validation_end: int, total_rows: int)` lưu ranh giới raw rows theo quy ước end-exclusive.
- `WindowBatch(X: np.ndarray, y: np.ndarray, origin_indices: np.ndarray, target_indices: np.ndarray)` lưu từng batch theo origin và target.
- `PreparedData(train: WindowBatch, validation: WindowBatch, test: WindowBatch, feature_scaler, target_scaler, bounds: SplitBounds, input_columns, target_column, lookback, horizon)` là đầu ra chung cho one-step và direct multi-output.
- `read_dataset(path: Path) -> pd.DataFrame` nạp CSV, parse ngày, sắp xếp tăng dần, kiểm tra cột cần dùng, ngày trùng, giá trị thiếu/không hữu hạn.
- `split_bounds(n_rows: int, train_ratio: float = 0.8, validation_ratio: float = 0.1) -> SplitBounds` dùng floor cho ranh giới 80% và 90%.
- `prepare_data(frame, input_columns, target_column, lookback, horizon=1, train_ratio=0.8, validation_ratio=0.1) -> PreparedData` fit scaler trên raw train rows, tạo cửa sổ theo origin và chỉ đưa một cửa sổ vào split nếu toàn bộ target của nó nằm trong split đó.

- [x] **Step 1: Viết test đỏ cho kiểm tra CSV, split, scaler và chỉ số cửa sổ.** Thêm các test nêu trong Review Focus cùng `test_split_bounds_use_chronological_floor_boundaries`.
- [x] **Step 2: Chạy test để xác nhận thất bại đúng vì API/chức năng chưa có.**

Run: `pytest -q tests/test_data.py`

Expected: FAIL vì `copper_forecasting.data` chưa tồn tại.

- [x] **Step 3: Cài đặt dataclass, loader, split và window preparation tối thiểu trong `copper_forecasting/data.py`.** Không fit scaler bằng validation/test; mọi `X` kết thúc tại origin và mọi target vector phải nằm trọn trong đúng split.
- [x] **Step 4: Chạy lại kiểm tra dữ liệu.**

Run: `pytest -q tests/test_data.py`

Expected: PASS.

### Task 2: Metric dùng chung

**Files:**
- Create: `copper_forecasting/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- `evaluate_forecast(y_true, y_pred, zero_tol: float = 1e-8) -> dict[str, float | None]` trả về `mse`, `rmse`, `mae`, `mape`; MAPE là `None` nếu bất kỳ actual nào có trị tuyệt đối `<= zero_tol`.

- [x] **Step 1: Viết test đỏ** cho dự báo hoàn hảo, metric có giá trị tính tay, mảng khác độ dài, NaN/Inf và MAPE không xác định khi actual gần 0.
- [x] **Step 2: Chạy test để xác nhận lỗi do module/hàm chưa tồn tại.**

Run: `pytest -q tests/test_metrics.py`

Expected: FAIL vì `evaluate_forecast` chưa được định nghĩa.

- [x] **Step 3: Cài đặt `evaluate_forecast`** bằng NumPy, kiểm tra shape/finite trước khi tính và không chia cho actual nhỏ hơn ngưỡng.
- [x] **Step 4: Chạy lại metric tests.**

Run: `pytest -q tests/test_metrics.py`

Expected: PASS.

### Task 3: Model factory và huấn luyện có cấu hình chung

**Files:**
- Create: `copper_forecasting/models.py`
- Create: `copper_forecasting/training.py`
- Create: `tests/test_models.py`
- Create: `tests/test_training.py`

**Interfaces:**
- `set_seed(seed: int) -> None` cố định `random`, NumPy và TensorFlow.
- `build_one_step_model(model_name: str, input_shape: tuple[int, int], learning_rate: float = 0.001) -> tf.keras.Model` hỗ trợ `SimpleRNN`, `LSTM`, `GRU`, `Bi-LSTM`, đều có 32 units và Dense(1).
- `build_direct_gru(input_shape: tuple[int, int], horizon: int = 5, learning_rate: float = 0.001) -> tf.keras.Model` tạo GRU(32) với Dense(horizon).
- `TrainingOutcome(model, history, best_epoch: int, epochs_ran: int, training_seconds: float)` lưu model tốt nhất và chi phí train.
- `fit_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=10) -> TrainingOutcome` bật EarlyStopping trên `val_loss`, `restore_best_weights=True`, ghi best epoch, số epoch thực chạy và thời gian train.

- [x] **Step 1: Viết test đỏ** cho tên model hợp lệ/không hợp lệ, input/output shapes, số outputs của DMS, compile MSE/Adam và `test_fit_model_records_best_epoch_and_epochs_ran` sau một lần fit trên dữ liệu tổng hợp nhỏ.
- [x] **Step 2: Chạy tests để xác nhận chúng thất bại vì factory/training chưa có.**

Run: `pytest -q tests/test_models.py tests/test_training.py`

Expected: FAIL vì các API chưa tồn tại.

- [x] **Step 3: Cài đặt seed, hai model factory và `fit_model`** với các giá trị mặc định từ Global Constraints.
- [x] **Step 4: Chạy lại tests model/training.**

Run: `pytest -q tests/test_models.py tests/test_training.py`

Expected: PASS.

### Task 4: Experiment runner one-step và artifacts

**Files:**
- Create: `copper_forecasting/experiments.py`
- Create: `copper_forecasting/artifacts.py`
- Create: `tests/test_experiments.py`

**Interfaces:**
- `RunResult(run_id: str, metadata: dict, metrics: dict, predictions: pd.DataFrame)` lưu dữ liệu của một lần chạy.
- `run_one_step_experiment(model_name, prepared_data, seed=42, epochs=100, batch_size=32, patience=10) -> RunResult` train, suy luận trên test và tính metric sau inverse transform.
- `save_run_result(result: RunResult, output_dir: Path) -> Path` lưu metadata JSON, dự đoán CSV và metric record; trả về thư mục run.
- `run_architecture_comparison(frame, output_dir, seeds=(42,), epochs=100, batch_size=32, patience=10) -> list[RunResult]` gọi chung một `PreparedData` cho RNN/LSTM/GRU/Bi-LSTM và mỗi seed.

- [x] **Step 1: Viết test đỏ** xác nhận bốn model nhận đúng cùng target dates, metric dùng giá gốc, metadata có seed/best epoch/parameter count/timing và file JSON/CSV được ghi.
- [x] **Step 2: Chạy test để xác nhận API runner chưa tồn tại.**

Run: `pytest -q tests/test_experiments.py`

Expected: FAIL vì runner và artifact writer chưa tồn tại.

- [x] **Step 3: Cài đặt runner one-step và lưu artifacts.** Ghi đủ dataset/date range, features, lookback, split boundaries, scaler, optimizer, seed, phiên bản thư viện, metric, tham số, best epoch và thời gian.
- [x] **Step 4: Chạy lại experiment tests.**

Run: `pytest -q tests/test_experiments.py`

Expected: PASS.

### Task 5: IMS, DMS và đánh giá theo lead time

**Files:**
- Create: `copper_forecasting/multistep.py`
- Create: `tests/test_multistep.py`

**Interfaces:**
- `iterative_predict(model, initial_windows: np.ndarray, horizon: int) -> np.ndarray` dự báo theo batch origins; mỗi kết quả được nối vào cửa sổ trước khi dự báo bước kế.
- `direct_predict(model, initial_windows: np.ndarray) -> np.ndarray` trả về ma trận `[n_origins, H]` từ model nhiều đầu ra.
- `evaluate_by_horizon(y_true: np.ndarray, y_pred: np.ndarray, target_scaler, zero_tol=1e-8) -> tuple[pd.DataFrame, dict]` inverse-transform, tính `mse`, `rmse`, `mae`, `mape` từng lead và metric gộp trên toàn bộ origin/lead pairs.
- `run_multistep_comparison(frame, output_dir, seeds=(42,), lookback=30, horizon=5, epochs=100, batch_size=32, patience=10) -> list[RunResult]` huấn luyện GRU one-step cho IMS và GRU multi-output cho DMS trên cùng split/origins cho mỗi seed.

- [x] **Step 1: Viết test đỏ** cho origin có horizon đầy đủ trong test, cuốn prediction đầu vào IMS, shape `[origins, H]`, metric theo lead và đối xứng origins IMS/DMS.
- [x] **Step 2: Chạy test để xác nhận hàm multi-step chưa tồn tại.**

Run: `pytest -q tests/test_multistep.py`

Expected: FAIL vì các hàm multi-step chưa tồn tại.

- [x] **Step 3: Cài đặt IMS/DMS** theo shared GRU(32), copper-only scaler và cùng origin set; suy luận không truy cập nhãn tương lai bên trong horizon.
- [x] **Step 4: Chạy lại multi-step tests.**

Run: `pytest -q tests/test_multistep.py`

Expected: PASS.

### Task 6: CLI, biểu đồ, tài liệu và kiểm tra chạy lại

**Files:**
- Create: `requirements-dev.txt` chứa `-r requirements-paper.txt` và `pytest`.
- Create: `copper_forecasting/cli.py`
- Create: `copper_forecasting/__main__.py`
- Create: `tests/test_cli.py`
- Create: `data/DATA_CLEANING_LOG.md` ghi nguồn, ngày tải, parse/join/filter, phạm vi ngày, số dòng/cột, missing và duplicate counts.
- Modify: `README.md`
- Modify: `PAPER_DATASET_README.md`
- Modify: `requirements-paper.txt` only if the runner needs an additional runtime dependency.

**Interfaces:**
- `python -m copper_forecasting --dataset PATH --output-dir PATH [--seeds N [N ...]] [--epochs N] [--models NAME [NAME ...]] [--skip-multistep]` chạy hai protocol (trừ khi chọn tùy chọn bỏ qua multi-step), lưu kết quả vào output directory. Mặc định là seed 42 và cả bốn model; các tên hợp lệ là `SimpleRNN`, `LSTM`, `GRU`, `Bi-LSTM`.
- `plot_architecture_comparison(results, output_path)` và `plot_multistep_errors(by_horizon, output_path)` tạo hình PNG không cần thao tác notebook.

- [x] **Step 1: Viết test đỏ** `test_cli_smoke_writes_expected_artifacts` cho CLI arguments, file outputs kỳ vọng và smoke run trên CSV tạm ít nhất 100 dòng với một GRU, một seed, một epoch.
- [x] **Step 2: Chạy CLI tests để xác nhận thiếu entry point/artifacts.**

Run: `pytest -q tests/test_cli.py`

Expected: FAIL vì CLI và plots chưa được định nghĩa.

- [x] **Step 3: Cài đặt CLI và hai plot; cập nhật README/dataset README và cleaning log** với lệnh chạy, protocol, nguồn/các URL, đơn vị hiển thị của từng futures series, phạm vi dữ liệu, cột, quy tắc ghép/loại thiếu, giới hạn tái lập và vị trí artifacts.
- [x] **Step 4: Chạy smoke test và toàn bộ test suite.**

Run: `pytest -q tests/test_cli.py`

Expected: PASS và thư mục kết quả có run metadata, prediction CSV, `one_step_metrics.csv`, `multistep_metrics_by_horizon.csv` và PNG.

Run: `pytest -q`

Expected: PASS.

### Task 7: Chạy thí nghiệm hoàn chỉnh và cập nhật Notion

**Files / external pages:**
- Create: `results/` artifacts từ runner.
- Update: Notion T4, T5, T6, T1, Dataset chính, Experiment Results và phần ghi chú IMS/DMS ở Project Workspace.

**Interfaces:**
- Dùng CLI Task 6 để chạy RNN/LSTM/GRU/Bi-LSTM one-step và GRU IMS/DMS với cùng cấu hình từ spec.
- Mỗi Notion Experiment Results row dùng `Model` tương ứng, `Strategy` One-step/IMS/DMS, các cột MSE/RMSE/MAE/MAPE, `Training Time (s)` và Ghi chú cho run ID, seed, lookback/horizon, split, inference time và parameter count.

- [x] **Step 1: Chạy merge/validation của dữ liệu hiện có và ghi kết quả vào cleaning log:** date range, rows, columns, duplicate/missing counts, raw files và quy tắc join.

Run: `python merge_investing_data.py`

Expected: dataset được tái tạo với 4.658 rows, 10 columns, dates 2006-01-26 đến 2026-09-25, 0 missing cells và 0 duplicate dates.

- [x] **Step 2: Chạy runner đầy đủ** trên dataset canonical; lưu kết quả và biểu đồ dưới `results/`.

Run: `python -m copper_forecasting --dataset data/copper_investing_2006_2026.csv --output-dir results --seeds 42`

Expected: có bốn run one-step, hai run multi-step, metric test raw-price và metadata cho từng run.

- [x] **Step 3: Cập nhật T4/T5/T6/T1, Dataset chính, Experiment Results và Project Workspace.** Dùng nội dung fetch hiện tại để chỉ sửa đúng properties/content; task T4–T6 chỉ chuyển Done khi bước 2 thành công.
- [x] **Step 4: Fetch lại các trang/database đã cập nhật** để xác nhận status, metric và protocol khớp artifacts; ghi lại link kết quả trong README.

Expected: trang T4/T5/T6 giải thích đúng mô hình và kết quả; T1 trỏ tới metric dùng chung; Dataset chính nêu URL, 4.658 dòng và biến đổi; cleaning log khớp CSV; Experiment Results có một hàng cho mỗi run; project notes định nghĩa đúng IMS và DMS multi-output.

## Review and execution notes

- Các task phụ thuộc theo thứ tự: Task 1 → Task 2 → Task 3 → Task 4/Task 5 → Task 6 → Task 7.
- Khuyến nghị triển khai native trong cùng phiên vì các task phụ thuộc các interface dùng chung và bước cuối cần kết hợp artifacts local với các trang Notion hiện có.
- Nếu TensorFlow không chạy trong môi trường hiện tại, vẫn hoàn thành code, tests và tài liệu; báo rõ giới hạn, không bịa metric và không đánh dấu các task thực nghiệm Done.
