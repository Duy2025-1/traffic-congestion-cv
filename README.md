# Ước Lượng Mức Độ Ùn Tắc Giao Thông Từ Camera (UTH)

**Học phần:** Xử Lý Ảnh và Thị Giác Máy Tính (121036)  
**Trường:** Đại học Giao thông Vận tải TP. Hồ Chí Minh  
**Nhóm:** 7 thành viên  

## 📌 Phát Biểu Mục Tiêu (Goal Statement)
* **Vấn đề:** Đánh giá mức độ ùn tắc giao thông theo thời gian thực từ camera quan sát TP.HCM.
* **Giả thuyết:** Kết hợp tỷ lệ chiếm dụng mặt đường (Ch.4), vector vận tốc Optical Flow (Ch.3) và tải trọng PCU (Ch.5) cho độ chính xác cao hơn tối thiểu 15% so với chỉ đếm số lượng phương tiện đơn thuần.
* **Tiêu chí thành công:** F1-score phân loại >= 85%, tốc độ xử lý >= 20 FPS.

## 👥 Phân Công Module
* **TV1 (Leader):** Kiến trúc Pipeline & Chỉ số TCI (modules/congestion_evaluator.py, main.py)
* **TV2:** Tiền xử lý & Nắn phối cảnh BEV (modules/preprocessing.py)
* **TV3:** Trừ nền & Tỷ lệ chiếm dụng mặt đường (modules/segmentation.py)
* **TV4:** Ước lượng chuyển động & Vận tốc (modules/optical_flow.py)
* **TV5:** Phát hiện & Phân loại phương tiện PCU (modules/detection.py)
* **TV6:** Đánh giá định lượng & Thí nghiệm (evaluation/metrics.py)
* **TV7:** Demo Dashboard & Thảo luận (demo/app.py)
