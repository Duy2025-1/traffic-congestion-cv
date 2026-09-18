# Hệ Thống Ước Lượng & Đánh Giá Mức Độ Ùn Tắc Giao Thông Từ Camera (UTH)

**Học phần:** Xử Lý Ảnh và Thị Giác Máy Tính (121036)  
**Trường:** Đại học Giao thông Vận tải TP. Hồ Chí Minh (UTH)  
**Quy mô:** Dự án nhóm 7 thành viên  

---

## 📌 1. Phát Biểu Mục Tiêu Dự Án (Goal Statement)
* **Bài toán thực tiễn:** Giám sát và đánh giá định lượng mức độ ùn tắc giao thông theo thời gian thực tại các tuyến đường đô thị TP. Hồ Chí Minh thông qua camera quan sát giao thông.
* **Đột phá kỹ thuật:** Kết hợp đa chiều giữa **Tỷ lệ chiếm dụng lòng đường (Ch.4 - TV3)**, **Vector suy giảm vận tốc qua Optical Flow (Ch.3 - TV4)** và **Mật độ tải trọng xe quy đổi PCU (Ch.5 - TV5)** thay vì chỉ đếm số lượng xe đơn thuần.
* **Tiêu chí thành công:**
  - Nhận diện chính xác 4 nhóm phương tiện đặc thù giao thông Việt Nam.
  - Phân loại mức độ ùn tắc với F1-score $\ge 85\%$.
  - Tốc độ xử lý đạt thời gian thực trên phần cứng tiêu chuẩn.

---

## 👥 2. Kiến Trúc Hệ Thống & Phân Công Nhiệm Vụ

Hệ thống được tổ chức thành 7 module chuyên biệt phối hợp chặt chẽ:

| Thành viên | Nhiệm vụ trọng tâm | Module phụ trách | Công nghệ & Thuật toán |
| :--- | :--- | :--- | :--- |
| **TV1 (Leader)** | Kiến trúc Pipeline tổng thể & Chỉ số TCI | [`modules/congestion_evaluator.py`](modules/congestion_evaluator.py), [`main.py`](main.py), [`config.py`](config.py) | Mô hình dung hợp TCI, Exponential Moving Average |
| **TV2** | Tiền xử lý & Nắn phối cảnh Bird's-Eye View | [`modules/preprocessing.py`](modules/preprocessing.py), [`configs/toadovideo.py`](configs/toadovideo.py) | Grayscale, Gaussian Blur, CLAHE, Perspective Transform |
| **TV3** | Phân đoạn mặt đường & Tỷ lệ chiếm dụng | [`modules/segmentation.py`](modules/segmentation.py) | Trừ nền MOG2/KNN, Khử bóng (Shadow Suppression), ROI Polygon |
| **TV4** | Ước lượng vận tốc dòng xe qua Optical Flow | [`modules/optical_flow.py`](modules/optical_flow.py) | Luồng quang học Farneback Dense, ColorMap JET Heatmap |
| **TV5** | Nhận diện phương tiện & Quy đổi tải trọng PCU | [`modules/detection.py`](modules/detection.py) | YOLOv8 (Slicing inference), Per-class NMS, Chuẩn PCU |
| **TV6** | Đánh giá định lượng & Thực nghiệm | [`evaluation/metrics.py`](evaluation/metrics.py) | Confusion Matrix, Accuracy, Precision, Recall, F1, Speed MAE |
| **TV7** | Web Dashboard Trực quan hóa & Nghiệm thu | [`demo/app.py`](demo/app.py) | Streamlit đa luồng, Đồng bộ 4 khung hình, HUD thời gian thực |

---

## 📐 3. Cơ Sở Lý Thuyết & Mô Hình Toán Học

### 3.1. Chỉ số đánh giá ùn tắc tổng hợp TCI (Traffic Congestion Index)
Chỉ số $TCI \in [0, 1]$ được tính toán tức thời tại mỗi khung hình thông qua hàm kết hợp có trọng số:
$$TCI_{instant} = w_1 \cdot \mathcal{O} + w_2 \cdot \left(1 - \frac{v}{v_{free}}\right) + w_3 \cdot \frac{D_{PCU}}{D_{max}}$$

Trong đó:
* $\mathcal{O} \in [0, 1]$: Tỷ lệ diện tích chiếm dụng mặt đường được trích xuất từ mặt nạ phân đoạn TV3 trong vùng đa giác ROI.
* $v$: Vận tốc trung bình đo được từ trường vector Optical Flow (TV4).
* $v_{free} = 40.0\text{ km/h}$: Vận tốc lưu thông tự do lý tưởng.
* $D_{PCU}$: Tổng tải trọng quy đổi sang Xe con tiêu chuẩn (Passenger Car Unit) của TV5.
* $D_{max} = 50.0\text{ PCU}$: Sức chứa tải trọng thiết kế tối đa của đoạn tuyến quan sát.
* **Bộ trọng số tối ưu (TV1):** $w_1 = 0.40$ (Chiếm dụng), $w_2 = 0.40$ (Vận tốc), $w_3 = 0.20$ (Tải trọng PCU).

Để tránh hiện tượng nhảy ngưỡng do nhiễu tức thời, giá trị TCI hiển thị được làm mịn qua cửa sổ trượt (Moving Average):
$$TCI_{smoothed} = \frac{1}{K} \sum_{i=0}^{K-1} TCI_{t-i} \quad (K = 15\text{ frames})$$

### 3.2. Bảng quy đổi hệ số tải trọng PCU (Chương 5 - TV5)
* 🏍️ **Xe máy (Motorcycle):** $0.33\text{ PCU}$ (3 xe máy $\approx$ 1 ô tô con)
* 🚗 **Ô tô con / Taxi (Car):** $1.00\text{ PCU}$ (Đơn vị xe chuẩn)
* 🚌 **Xe buýt (Bus):** $2.50\text{ PCU}$
* 🚛 **Xe tải / Xe container (Truck):** $3.00\text{ PCU}$

### 3.3. Thang phân cấp mức độ ùn tắc giao thông
* **Mức 1 · Thông thoáng (Free Flow):** $TCI < 0.30$ (Mã màu: Xanh lá `#34d399`)
* **Mức 2 · Bình thường (Moderate):** $0.30 \le TCI < 0.55$ (Mã màu: Vàng `#facc15`)
* **Mức 3 · Ùn ứ (Heavy Congestion):** $0.55 \le TCI < 0.75$ (Mã màu: Cam `#fb923c`)
* **Mức 4 · Tắc nghẽn nghiêm trọng (Severe Jam):** $TCI \ge 0.75$ (Mã màu: Đỏ `#f87171`)

---

## 💻 4. Cài Đặt & Hướng Dẫn Sử Dụng

### 4.1. Chuẩn bị môi trường
Yêu cầu: Python 3.10 trở lên. Khởi tạo môi trường ảo và cài đặt các thư viện phụ thuộc:
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv venv

# Trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

---

### 4.2. Khởi chạy Web Dashboard Trực Quan (Streamlit)
Giao diện Web tương tác cao của TV7 hiển thị đồng thời 4 cửa sổ kiểm soát, HUD trung tâm, biểu đồ xu hướng và công cụ xuất báo cáo nghiệm thu:

```bash
# Khởi chạy ứng dụng Web
streamlit run demo/app.py
```
Truy cập giao diện tại: **`http://localhost:8501`**

#### Các tính năng nổi bật trên Web Dashboard:
1. **Lưới 4 cửa sổ quan sát đồng bộ (2x2 Grid):**
   - **Cửa sổ 1 (TV5):** Khung hình camera gốc kèm bounding box nhận diện 4 nhóm xe và viền đa giác ROI.
   - **Cửa sổ 2 (TV2):** Ảnh nắn phối cảnh Bird's-Eye View (BEV) triệt tiêu hiệu ứng biến dạng góc tụ.
   - **Cửa sổ 3 (TV3):** Mặt nạ chiếm dụng mặt đường nhị phân (Occupancy Mask) đã lọc bóng đổ và nhiễu viền.
   - **Cửa sổ 4 (TV4):** Bản đồ nhiệt vận tốc dòng xe (Optical Flow Heatmap Jet ColorMap).
2. **Bảng điều khiển trung tâm (HUD):** Hiển thị trực tiếp FPS, TCI tức thời/làm mịn, % diện tích chiếm dụng, % suy giảm vận tốc, Tổng tải trọng PCU và Nhãn trạng thái kẹt xe kèm mã màu.
3. **Bộ điều khiển tham số tức thì:** Tùy chỉnh trực tiếp ngưỡng Confidence, IoU NMS, Ngưỡng lọc nhiễu vận tốc, Tham số MOG2 mà không bị reload mô hình (nhờ cơ chế `@st.cache_resource`).
4. **Điều khiển phát linh hoạt:** Hỗ trợ Tạm dừng (Pause), Tiếp tục (Resume) mượt mà không bị tua lại từ đầu, nút Chạy lại từ đầu (Replay) và Tăng tốc bước nhảy frame.
5. **Nghiệm thu Snapshot:** Nút tải ảnh Canvas 2x2 tổng hợp chất lượng cao phục vụ viết báo cáo khoa học.

---

### 4.3. Khởi chạy Pipeline Desktop (OpenCV CLI)
Dành cho kiểm thử hiệu năng tốc độ cao hoặc triển khai trên thiết bị Edge/Server không có giao diện web:

```bash
# Chạy hiển thị giao diện đa cửa sổ OpenCV
python main.py

# Chạy với video tuỳ chọn
python main.py data/raw/traffic_congested.mp4

# Chế độ headless (dành cho chạy ngầm / kiểm thử server)
python main.py --headless --max-frames=100
```

---

### 4.4. Huấn Luyện & Fine-Tuning Đa Video (`train.py`)
Hệ thống cung cấp công cụ huấn luyện linh hoạt cho phép học và tối ưu hoá nhận dạng phương tiện từ **nhiều video bất kỳ**:

```bash
# 1. Trích xuất frame và gán nhãn giả chất lượng cao từ nhiều video:
python train.py --videos data/raw/traffic_congested.mp4 data/raw/traffic_free_flow.mp4 --prepare-data

# 2. Huấn luyện trực tiếp từ thư mục chứa nhiều video:
python train.py --video-dir data/raw --epochs 30 --batch 4

# 3. Huấn luyện từ danh sách video bất kỳ bên ngoài:
python train.py --videos "D:/videos/cam1.mp4" "D:/videos/cam2.mp4" --epochs 30

# 4. Đánh giá kiểm chuẩn mô hình sau huấn luyện:
python train.py --val-only --model weights/best.pt
```

---

## 📁 5. Cấu Trúc Thư Mục Dự Án

```
traffic-congestion-cv/
├── configs/
│   ├── toadovideo.py                 # Toạ độ 4 điểm góc biến đổi BEV và đa giác ROI từng video
│   └── traffic_dataset.yaml          # Cấu hình tập dữ liệu 4 nhóm phương tiện PCU
├── data/
│   ├── raw/                          # Video camera giao thông thực tế TP.HCM (.mp4)
│   └── processed/                    # Thư mục lưu trữ ảnh Canvas nghiệm thu (sạch sẽ)
├── demo/
│   └── app.py                        # Mã nguồn ứng dụng Web Dashboard Streamlit
├── evaluation/
│   └── metrics.py                    # Bộ công cụ đánh giá định lượng phân loại & MAE vận tốc
├── modules/
│   ├── __init__.py
│   ├── congestion_evaluator.py       # Bộ tính toán chỉ số TCI tổng hợp & phân cấp mức độ
│   ├── preprocessing.py              # Tiền xử lý ảnh (CLAHE) & Nắn góc BEV
│   ├── segmentation.py               # Phân đoạn thích nghi Auto-ROI & Hybrid Occupancy
│   ├── optical_flow.py               # Ước lượng vận tốc qua Farneback Optical Flow
│   └── detection.py                  # Phát hiện phương tiện YOLOv8 & Tính tải trọng PCU
├── config.py                         # Cấu hình trọng số TCI, hệ số PCU và ngưỡng vận tốc
├── main.py                           # Điểm khởi chạy pipeline chính trên Desktop (OpenCV CLI)
├── train.py                          # Bộ công cụ huấn luyện & fine-tuning đa video YOLOv8
├── requirements.txt                  # Danh mục thư viện phụ thuộc của dự án
└── README.md                         # Báo cáo thuyết minh & Tài liệu hướng dẫn sử dụng
```

---


## 📊 6. Kết Quả Thực Nghiệm & Đánh Giá Định Lượng (TV6)
* Mô hình phát hiện phương tiện YOLOv8 kết hợp kỹ thuật Two-pass Slicing Inference nhận diện hiệu quả các cụm xe máy mật độ cao và phương tiện ở xa chân trời.
* Tích hợp vùng ROI mặt đường giúp chỉ số chiếm dụng đường phản ánh chính xác diện tích mặt cắt ngang giao thông, không bị nhiễu bởi vỉa hè hoặc tán cây.
* Sự kết hợp giữa Optical Flow và tải trọng PCU giúp phân biệt rõ ràng giữa hai trường hợp: **Đường vắng đi nhanh (TCI thấp)** và **Đường kẹt xe đứng yên (Vận tốc triệt tiêu $\rightarrow$ TCI cao)**.

---
*Bản quyền học phần Xử lý ảnh & Thị giác máy tính - Khoa Công nghệ Thông tin - Trường ĐH Giao thông Vận tải TP.HCM (UTH).*
