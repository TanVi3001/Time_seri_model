# Dataset và phạm vi tái lập

## Bài báo và notebook paper-inspired

Chen et al. (2023), [Copper price prediction using LSTM recurrent neural network integrated simulated annealing algorithm](https://doi.org/10.1371/journal.pone.0285631), mô tả dữ liệu 1990-01-01 đến 2009-12-31 và 4,870 quan sát sau xử lý thiếu. Bài báo là tài liệu tham khảo cho bài toán và nguồn dữ liệu; repo không tuyên bố tái lập toàn bộ thuật toán LSTM kết hợp simulated annealing.

Hai notebook cũ (`../notebooks/RNN_Copper_Paper_2006_2026.ipynb`, `../notebooks/LSTM_Copper_Paper_2006_2026.ipynb`) giữ cấu hình paper-inspired riêng: lookback 22, đầu vào WTI/vàng/bạc, SimpleRNN hoặc LSTM với kiến trúc rộng 39 rồi 111, Dense 32, dropout 0.2, batch 64 và learning rate `5e-5`. Không dùng trực tiếp các kết quả cũ này để xếp hạng với runner mới.

## Dataset hiện dùng

File chuẩn là [`../data/copper_investing_2006_2026.csv`](../data/copper_investing_2006_2026.csv), được tạo bởi `../scripts/merge_investing_data.py` từ các CSV trong `../data/raw_investing/`.

- Phạm vi cần lọc: 2006-01-01 đến 2026-09-26; ngày giao dịch thực tế trong file: 2006-01-26 đến 2026-09-25.
- Kích thước chuẩn: 4,658 hàng × 10 cột (cột ngày và chín chuỗi số).
- Sau căn chỉnh theo ngày: 0 giá trị thiếu và 0 ngày trùng.
- Ghép inner join trên `date`; không nội suy. Các ngày không có đủ mọi chuỗi bị loại.
- Giá trị `Price` được parse thành số; volume đồng có hậu tố K/M/B được chuyển sang số lượng.
- Ngày lưu trong file ở dạng ISO `YYYY-MM-DD`, tăng dần.

Các chuỗi giá là đồng COMEX, WTI, vàng và bạc; dữ liệu liên quan còn gồm Dollar Index, lợi suất Treasury 10 năm, NASDAQ Composite, Dow Jones và volume đồng. Đơn vị báo giá theo nguồn: đồng USD/pound, WTI USD/barrel, vàng/bạc USD/troy ounce, lợi suất phần trăm, chỉ số theo điểm. CSV gốc không lưu metadata đơn vị riêng; chi tiết nguồn và giới hạn của ngày export được ghi trong [`DATA_CLEANING_LOG.md`](DATA_CLEANING_LOG.md).

Nguồn lịch sử chính: [đồng](https://www.investing.com/commodities/copper-historical-data), [WTI](https://www.investing.com/commodities/crude-oil-historical-data), [vàng](https://www.investing.com/commodities/gold-historical-data), [bạc](https://www.investing.com/commodities/silver-historical-data). Các chuỗi bổ sung được liệt kê cùng URL trong nhật ký dữ liệu.

## Giao thức runner mới

Package `copper_forecasting` thực hiện hai thí nghiệm tách biệt:

1. **So sánh kiến trúc một bước:** SimpleRNN/LSTM/GRU/Bi-LSTM, đầu vào lịch sử 30 phiên của bốn chuỗi giá đồng/WTI/vàng/bạc, dự đoán giá đồng phiên kế tiếp.
2. **IMS và DMS nhiều bước:** hai GRU dùng cùng một biến là giá đồng, cùng lookback 30, horizon mặc định 5 và cùng origins trong test. IMS cuốn dự báo của chính nó; DMS dự đoán trực tiếp vector `[t+1, …, t+H]`.

Cả hai dùng split theo thời gian 80/10/10, scaler chỉ fit trên train, Adam 0.001, MSE, batch 32, tối đa 100 epoch, early stopping patience 10 và seed mặc định 42. Metric được tính trên giá gốc sau inverse transform. Giá trị tương lai của WTI/vàng/bạc không được đưa vào dự báo; IMS không cập nhật bằng actual trong horizon.

Chạy hướng dẫn và lệnh đầy đủ trong [`README.md`](../README.md). Nhật ký chuyển đổi chi tiết, số hàng từng nguồn và missing/duplicate counts ở [`DATA_CLEANING_LOG.md`](DATA_CLEANING_LOG.md).
