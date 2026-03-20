# Dự án Khai phá dữ liệu - Đề tài 16: Dự đoán lỗi sản xuất

## Cách cài đặt
1. Đảm bảo bạn có Python 3.8+ được cài đặt.
2. Tạo môi trường ảo (khuyến nghị):
   ```
   python -m venv env
   env\Scripts\activate  # Trên Windows
   ```
3. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```
4. Đặt dữ liệu gốc vào thư mục: `data/raw/ai4i2020.csv`
5. (Tùy chọn) Nếu cần tạo lại cấu trúc dự án, chạy:
   ```
   python setup_project.py
   ```

## Cách chạy dự án
- Chạy toàn bộ pipeline: `python scripts/run_pipeline.py`
- Xem phân tích chi tiết: Mở các file trong thư mục `notebooks/` theo thứ tự từ 01 đến 05.

## Cấu trúc nổi bật
- **Khai phá:** Sử dụng Apriori tìm luật kết hợp cho từng loại lỗi riêng biệt (HDF, OSF...).
- **Bán giám sát:** Triển khai Self-Training để cải thiện hiệu năng khi thiếu hãn lỗi.