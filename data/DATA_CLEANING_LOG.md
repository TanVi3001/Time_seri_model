# Nhật ký dữ liệu đồng

## Nguồn và ngày ghi nhận

Các file đầu vào là các CSV lịch sử Investing.com trong `data/raw_investing/`. Các file hiện có ghi nhận ngày sửa đổi trong workspace là 26-09-2026; bản CSV không lưu thời điểm export ban đầu, vì vậy ngày này là ngày ghi nhận file cục bộ chứ không phải xác nhận độc lập thời điểm Investing.com xuất dữ liệu.

| Chuỗi | URL lịch sử | Đơn vị báo giá |
|---|---|---|
| Đồng (HG) | [Investing.com](https://www.investing.com/commodities/copper-historical-data) | USD/pound |
| WTI | [Investing.com](https://www.investing.com/commodities/crude-oil-historical-data) | USD/barrel |
| Vàng | [Investing.com](https://www.investing.com/commodities/gold-historical-data) | USD/troy ounce |
| Bạc | [Investing.com](https://www.investing.com/commodities/silver-historical-data) | USD/troy ounce |
| Chỉ số USD | [Investing.com](https://www.investing.com/currencies/us-dollar-index-historical-data) | điểm chỉ số |
| Lợi suất Treasury 10 năm | [Investing.com](https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data) | phần trăm lợi suất |
| NASDAQ Composite | [Investing.com](https://www.investing.com/indices/nasdaq-composite-historical-data) | điểm chỉ số |
| Dow Jones Industrial Average | [Investing.com](https://www.investing.com/indices/us-30-historical-data) | điểm chỉ số |

Đơn vị được suy ra từ thông tin instrument/contract của nguồn; file Investing CSV không chứa metadata đơn vị riêng. Trang đồng ghi unit là pound; specification CME xác nhận báo giá đồng theo USD mỗi pound. WTI được báo theo USD/barrel; vàng và bạc theo USD/troy ounce. Các trường giá được giữ nguyên, không đổi đơn vị.

## Quy tắc parse và ghép

Việc làm sạch có thể tái tạo bằng `python merge_investing_data.py`.

1. Parse `Date` theo định dạng `%m/%d/%Y`; parse `Price` thành số sau khi bỏ dấu phẩy. `Vol.` đồng được chuyển hậu tố K/M/B thành số lượng contracts.
2. Ghép các file bổ sung của cùng chuỗi, sắp ngày tăng dần, bỏ ngày trùng và giữ bản ghi cuối trong thứ tự file. Trong các file hiện có, số hàng hợp lệ sau parse bằng số hàng đầu vào; không có hàng nào bị loại do ngày hoặc `Price` không đọc được.
3. Inner join tám chuỗi theo ngày. Không nội suy hay điền giá bằng 0.
4. Giữ khoảng ngày yêu cầu 2006-01-01 đến 2026-09-26 (hai đầu bao gồm), ép các cột giá/lượng về numeric, loại hàng thiếu bất kỳ cột giá trị nào, sắp ngày tăng và kiểm tra ngày trùng.

Các CSV dài và file bổ sung có phần ngày chồng nhau. Số ngày trùng được loại khi ghép cùng chuỗi:

| Chuỗi | Số hàng trong file nguồn | Ngày trùng bị loại | Ngày còn lại trước inner join |
|---|---:|---:|---:|
| Đồng | 7,798 | 2,473 | 5,325 |
| WTI | 7,821 | 2,456 | 5,365 |
| Vàng | 4,907 | 0 | 4,907 |
| Bạc | 4,828 | 0 | 4,828 |
| Chỉ số USD | 7,792 | 2,411 | 5,381 |
| Treasury 10 năm | 7,790 | 2,409 | 5,381 |
| NASDAQ Composite | 7,698 | 2,499 | 5,199 |
| Dow Jones | 7,700 | 1,744 | 5,956 |

## Kết quả đã kiểm tra

- File chuẩn: `data/copper_investing_2006_2026.csv`.
- Kích thước: 4,658 hàng × 10 cột (ngày và chín chuỗi số).
- Phạm vi thực tế: 2006-01-26 đến 2026-09-25. Ngày 2026-09-26 nằm trong giới hạn yêu cầu nhưng không có phiên giao dịch trong file.
- Sau inner join trước bước drop thiếu: 4,658 hàng; 0 ô thiếu.
- File chuẩn: 0 ngày trùng và 0 ô thiếu.
- Cột: `date`, `closed_copper_price`, `volume_copper`, `close_dollar_index`, `close_t_bill`, `close_Nasdaq`, `close_Dow_Jones_Index`, `close_wti_oil`, `close_gold`, `close_silver`.
- T1–T5 mới không dùng toàn bộ chín biến: so sánh kiến trúc dùng lịch sử bốn giá đồng/WTI/vàng/bạc; IMS/DMS dùng riêng giá đồng.

Chen et al. (2023) mô tả giai đoạn 1990–2009 và 4,870 quan sát sau lọc thiếu. File hiện tại có giai đoạn 2006–2026 và 4,658 hàng nên đây là bộ dữ liệu demo khác, không phải tái lập chính xác CSV của bài báo. Quy tắc ghép và dữ liệu nguồn cục bộ được lưu cùng repo để tạo lại file chuẩn.
