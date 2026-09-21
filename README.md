# Hệ Thống Ước Lượng & Đánh Giá Mức Độ Ùn Tắc Giao Thông Từ Camera (UTH)

**Học phần:** Xử Lý Ảnh và Thị Giác Máy Tính (Mã HP: 121036 | 3 Tín chỉ)  
**Trường:** Đại học Giao thông Vận tải TP. Hồ Chí Minh (UTH)  
**Khoa:** Công nghệ Thông tin  
**Quy mô dự án:** Nhóm 7 thành viên (TV1 – TV7)  
**Kho lưu trữ GitHub:** [https://github.com/Duy2025-1/traffic-congestion-cv](https://github.com/Duy2025-1/traffic-congestion-cv)

---

## 📌 1. Phát Biểu Mục Tiêu & Giả Thuyết Dự Án (Bắt Buộc Theo Chuẩn Đề Bài UTH)

* **Vấn đề cụ thể:** Đánh giá định lượng và phân loại chính xác mức độ ùn tắc giao thông theo thời gian thực từ camera quan sát cố định trên 3 kịch bản video thực tế đa dạng (`traffic_free_flow.mp4`, `traffic_congested.mp4`, `traffic_traffic_light.mp4`). Thách thức gồm: biến thiên ánh sáng (mưa mù, chênh lệch nắng-bóng râm), các vật thể ngoại vi (vỉa hè, cây xanh, tòa nhà, làn ngược chiều), rung lắc camera và hiện tượng *"Background Absorption"* (xe dừng kẹt đứng yên lâu bị mô hình trừ nền học nhầm thành ảnh nền).
* **Giả thuyết kiểm chứng được:** 
  1. *Phân đoạn & ROI (Ch.4):* Việc áp dụng **Vùng ROI hình thang phối cảnh (Perspective Trapezoidal ROI)** bám khít mép mặt đường thực tế kết hợp cơ chế **Hybrid Occupancy Fusion** $(Mask_{\text{MOG2}} \cup Mask_{\text{YOLO}}) \cap Mask_{\text{ROI}}$ sẽ loại bỏ $100\%$ phương tiện ngược chiều/ngoại cảnh và bù đắp hoàn toàn diện tích xe đứng yên bị trừ nền học nhầm, duy trì Occupancy $> 60\%$ khi kẹt xe.
  2. *Chuyển động (Ch.3):* Thuật toán **Gunnar Farneback Dense Optical Flow** với ngưỡng lọc nhiễu $noise\_threshold = 1.0\text{ px/frame}$ sẽ triệt tiêu hoàn toàn rung lắc camera và gợn nước mưa mà vẫn đo chính xác vận tốc dòng xe ($v \approx 60-70\text{ km/h}$ khi thông thoáng và $v < 10\text{ km/h}$ khi kẹt cứng).
  3. *Hợp nhất đa đặc trưng TCI (TV1):* Mô hình TCI kết hợp 3 đặc trưng vật lý $[Occupancy, Vận tốc, Tải trọng PCU]$ với bộ trọng số $(w_1=0.40, w_2=0.40, w_3=0.20)$ sẽ cho độ chính xác phân loại mức độ ùn tắc vượt trội hơn ít nhất $20\%$ so với phương pháp chỉ đếm số lượng xe truyền thống.
* **Tiêu chí thành công (định lượng):**
  - Nhận diện và phân loại chính xác 4 nhóm phương tiện theo chuẩn PCU với $mAP@0.5 \ge 85\%$.
  - Tỷ lệ nhận diện nhầm ngoài làn đường giảm về $0\%$.
  - Độ chính xác phân loại mức độ ùn tắc tổng thể (4 cấp độ) đạt $Accuracy \ge 90\%$; sai số vận tốc $MAE < 5.0\text{ km/h}$.
  - Tốc độ xử lý pipeline đạt thời gian thực ($\ge 25\text{ FPS}$).

---

## 👥 2. Phân Công Nhiệm Vụ & Ánh Xạ Kỹ Thuật Chương Trình Học

Dự án áp dụng toàn diện kiến thức của cả **4 chương cốt lõi** trong chương trình học (vượt xa yêu cầu tối thiểu 3 kỹ thuật, trong đó 3/4 kỹ thuật thuộc nhóm chuyên sâu Chương 3, 4, 5):

| Thành viên | Trách nhiệm trọng tâm | Chương môn học | Kỹ thuật & Thuật toán triển khai | Tệp mã nguồn phụ trách |
| :--- | :--- | :---: | :--- | :--- |
| **TV1 (Leader)** | Kiến trúc Pipeline & Hợp nhất TCI | **Chương 1-5** | Chỉ số ùn tắc đa đặc trưng TCI, Exponential Moving Average (EMA), kiểm soát luồng đa luồng | [`modules/congestion_evaluator.py`](modules/congestion_evaluator.py), [`main.py`](main.py), [`config.py`](config.py) |
| **TV2** | Tiền xử lý & Nắn góc phối cảnh | **Chương 2** | Grayscale, Gaussian Blur khử nhiễu, Cân bằng sáng cục bộ thích nghi CLAHE, Bird's-Eye View (BEV) | [`modules/preprocessing.py`](modules/preprocessing.py) |
| **TV3** | Phân đoạn mặt đường & ROI phối cảnh | **Chương 4** | Trừ nền MOG2, Khử bóng đổ (detectShadows), Lọc hình thái học Opening/Closing, ROI đa giác hình thang phối cảnh, Cơ chế Hybrid Occupancy Fusion | [`modules/segmentation.py`](modules/segmentation.py), [`configs/toadovideo.py`](configs/toadovideo.py) |
| **TV4** | Ước lượng vận tốc dòng xe | **Chương 3** | Gunnar Farneback Dense Optical Flow, Bộ lọc nhiễu rung lắc, Bản đồ nhiệt năng lượng vận tốc (JET Heatmap) | [`modules/optical_flow.py`](modules/optical_flow.py) |
| **TV5** | Nhận diện xe & Quy đổi tải trọng | **Chương 5** | Mạng nơ-ron tích chập YOLOv8, Two-pass Slicing Inference, Per-Class NMS, Quy đổi hệ số tải trọng xe con chuẩn PCU, Bộ lọc đối tượng theo ROI | [`modules/detection.py`](modules/detection.py) |
| **TV6** | Đánh giá định lượng & Kiểm chuẩn | **Đánh giá** | Confusion Matrix, Accuracy, Precision, Recall, F1-Score, Tốc độ MAE, Phân tích kiểm chứng giả thuyết | [`evaluation/metrics.py`](evaluation/metrics.py) |
| **TV7** | Web Dashboard & Trực quan hóa | **Hệ thống** | Ứng dụng Streamlit tương tác thời gian thực, Lưới 4 khung hình đồng bộ (Camera, BEV, Occupancy Mask, Flow Heatmap), HUD thời gian thực | [`demo/app.py`](demo/app.py) |

---

## 📐 3. Cơ Sở Lý Thuyết & Mô Hình Toán Học

### 3.1. Chỉ số đánh giá ùn tắc tổng hợp TCI (Traffic Congestion Index)
Chỉ số $TCI \in [0, 1]$ được tính toán tức thời tại mỗi khung hình thông qua hàm kết hợp đa chiều:
$$TCI_{instant} = w_1 \cdot \mathcal{O} + w_2 \cdot \left(1 - \frac{v}{v_{free}}\right) + w_3 \cdot \frac{D_{PCU}}{D_{max}}$$

Trong đó:
* $\mathcal{O} \in [0, 1]$: Tỷ lệ diện tích chiếm dụng lòng đường (Occupancy Ratio) trong vùng ROI đa giác hình thang:
  $$\mathcal{O} = \frac{\sum_{(x, y) \in ROI} Mask_{\text{Hybrid}}(x, y)}{\text{Area}(ROI)}$$
* $v$: Vận tốc trung bình của các pixel chuyển động thực tế từ Optical Flow (TV4).
* $v_{free} = 40.0\text{ km/h}$: Vận tốc chuẩn lưu thông tự do của tuyến đường.
* $D_{PCU}$: Tổng tải trọng dòng xe quy đổi sang Đơn vị Xe con Tiêu chuẩn (Passenger Car Unit) của TV5.
* $D_{max} = 50.0\text{ PCU}$: Sức chứa tải trọng thiết kế tối đa của phân đoạn đường quan sát.
* **Bộ trọng số tối ưu (TV1):** $w_1 = 0.40$ (Chiếm dụng), $w_2 = 0.40$ (Vận tốc), $w_3 = 0.20$ (Tải trọng PCU).

Để triệt tiêu hiện tượng nhảy ngưỡng tức thời, TCI được làm mịn qua cửa sổ trượt:
$$TCI_{smoothed} = \frac{1}{K} \sum_{i=0}^{K-1} TCI_{t-i} \quad (K = 15\text{ frames})$$

### 3.2. Bảng quy đổi hệ số tải trọng PCU (Chương 5 - TV5)
* 🏍️ **Xe máy / Xe 2 bánh (Motorcycle):** $0.33\text{ PCU}$ (3 xe máy $\approx$ 1 ô tô con)
* 🚗 **Ô tô con / Taxi (Car):** $1.00\text{ PCU}$ (Đơn vị cơ sở chuẩn)
* 🚌 **Xe buýt (Bus):** $2.50\text{ PCU}$
* 🚛 **Xe tải / Xe container (Truck):** $3.00\text{ PCU}$

### 3.3. Thang phân cấp 4 mức độ ùn tắc giao thông đô thị
* **Mức 1 · Thông thoáng (Free Flow):** $TCI < 0.30$ (Màu hiển thị: Xanh lá `#34d399`)
* **Mức 2 · Bình thường (Moderate):** $0.30 \le TCI < 0.55$ (Màu hiển thị: Vàng `#facc15`)
* **Mức 3 · Ùn ứ (Heavy Congestion):** $0.55 \le TCI < 0.75$ (Màu hiển thị: Cam `#fb923c`)
* **Mức 4 · Tắc nghẽn nghiêm trọng (Severe Jam):** $TCI \ge 0.75$ (Màu hiển thị: Đỏ `#f87171`)

---

## 🎯 4. Vùng ROI Hình Thang Phối Cảnh Độc Lập Cho Từng Video (`configs/toadovideo.py`)

Hệ thống thiết lập tọa độ 4 đỉnh hình thang phối cảnh $[P_0, P_1, P_2, P_3]$ bám khít mép đường thực tế, được biến đổi chính xác qua ma trận Homography:

```python
VIDEO_CONFIG = {
    # 1. Cao tốc thông thoáng (1920x1080) - Thu hẹp đúng 2 làn chính, loại bỏ xe lề đường
    "traffic_free_flow.mp4": {
        "src_pts": np.float32([[977, 0], [1046, 0], [1609, 1078], [514, 1078]])
    },
    # 2. Phố New York ùn tắc (1920x1080) - Cạnh trên tại ngã tư xa, cạnh phải bám hàng xe đỗ
    "traffic_congested.mp4": {
        "src_pts": np.float32([[940, 285], [1071, 285], [1869, 1078], [260, 1078]])
    },
    # 3. Ngã tư Hà Nội đèn tín hiệu (1918x970) - Đỉnh tại vạch dừng ngã tư, loại 100% xe ngược chiều
    "traffic_traffic_light.mp4": {
        "src_pts": np.float32([[927, 455], [991, 456], [1685, 969], [485, 969]])
    }
}
```

* Phương thức `filter_detections_by_roi()` sử dụng hàm `cv2.pointPolygonTest()` loại bỏ triệt để mọi phương tiện có tâm nằm ngoài polygon ROI trước khi tính toán Occupancy và PCU.

---

## 💻 5. Cài Đặt & Hướng Dẫn Sử Dụng

### 5.1. Chuẩn bị môi trường
Yêu cầu hệ thống: Python 3.10 trở lên.
```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# 2. Cài đặt các thư viện tiêu chuẩn
pip install -r requirements.txt
```

---

### 5.2. Khởi chạy Web Dashboard Trực Quan (Streamlit)
```bash
streamlit run demo/app.py
```
Truy cập giao diện tại: **`http://localhost:8501`**

* **Tính năng chính trên Dashboard:**
  - **Lưới 4 khung hình đồng bộ:** Frame gốc nhận diện + ROI viền vàng (TV5), Ảnh nắn góc BEV (TV2), Mặt nạ chiếm dụng Hybrid Mask (TV3), Bản đồ nhiệt vận tốc JET Heatmap (TV4).
  - **Bảng điều khiển trung tâm (HUD):** Hiển thị trực tiếp FPS thời gian thực, TCI tức thời/làm mịn, % diện tích chiếm dụng, % suy giảm vận tốc, Tổng tải trọng PCU và Nhãn cấp độ màu.
  - **Khảo sát tham số trực tiếp (Interactive Parameter Tuning):** Điều chỉnh tức thì Confidence, IoU NMS, Ngưỡng lọc nhiễu vận tốc mà không cần tải lại mô hình (nhờ `@st.cache_resource`).

---

### 5.3. Khởi chạy Pipeline Desktop (OpenCV CLI)
Dành cho kiểm thử tốc độ cao hoặc triển khai server không có màn hình (headless):
```bash
# Chạy mặc định hiển thị giao diện OpenCV
python main.py

# Chạy với video tùy chọn
python main.py data/raw/traffic_congested.mp4

# Chế độ headless (kiểm thử hiệu năng)
python main.py --headless --max-frames=150
```

---

### 5.4. Huấn Luyện & Fine-Tuning YOLOv8 (`train.py`)
```bash
# 1. Trích xuất frame và tạo nhãn bán tự động từ video
python train.py --videos data/raw/traffic_congested.mp4 data/raw/traffic_free_flow.mp4 --prepare-data

# 2. Huấn luyện mô hình với cấu hình PCU
python train.py --video-dir data/raw --epochs 30 --batch 4

# 3. Đánh giá kiểm chuẩn mô hình
python train.py --val-only --model weights/best.pt
```

---

## 📓 6. Hệ Thống Sổ Ghi Chép Thí Nghiệm Lab (Jupyter Notebooks)

Hệ thống bao gồm đầy đủ **6 sổ ghi chép thí nghiệm** độc lập, đáp ứng trọn vẹn yêu cầu lab report (Phát biểu mục tiêu/giả thuyết, Ảnh trung gian, Parameter Sweep $\ge 3$ giá trị, Nhận xét phân tích):

```bash
# Khởi chạy Jupyter Notebook để xem toàn bộ thí nghiệm:
jupyter notebook
```

| Notebook | Tên thí nghiệm & Nội dung | Yêu cầu bài tập lớn đáp ứng |
| :--- | :--- | :--- |
| [`01_tci_parameter_sweep.ipynb`](notebooks/01_tci_parameter_sweep.ipynb) | **Khảo sát bộ trọng số TCI:** So sánh 3 bộ trọng số $(w_1, w_2, w_3)$ trên cả 3 video thực tế. | Khảo sát tham số hợp nhất đa đặc trưng TCI |
| [`02_preprocessing_sweep.ipynb`](notebooks/02_preprocessing_sweep.ipynb) | **Tiền xử lý ảnh (Ch.2):** Khảo sát tham số `clipLimit` của CLAHE (1.0 vs 2.0 vs 4.0) và kích thước Gaussian kernel. | Ảnh trung gian, Parameter Sweep Chương 2 |
| [`03_segmentation_roi_sweep.ipynb`](notebooks/03_segmentation_roi_sweep.ipynb) | **Phân đoạn ảnh & ROI (Ch.4):** Khảo sát `varThreshold` MOG2 (8.0 vs 16.0 vs 32.0), so sánh ROI chữ nhật vs ROI phối cảnh, kiểm chứng giải pháp Hybrid Occupancy chống Background Absorption. | Ảnh trung gian 5 bước, Parameter Sweep Chương 4 |
| [`04_optical_flow_tv4.ipynb`](notebooks/04_optical_flow_tv4.ipynb) | **Đo vận tốc Optical Flow (Ch.3):** Khảo sát `noise_threshold` (0.5 vs 1.0 vs 2.0 px/frame), Quiver Plot vector và JET Heatmap. | Ảnh trung gian, Parameter Sweep Chương 3 |
| [`05_detection_tv5.ipynb`](notebooks/05_detection_tv5.ipynb) | **Nhận dạng đối tượng (Ch.5):** Khảo sát `conf_threshold` và `iou_threshold`, Per-Class NMS, quy đổi PCU, lọc xe theo ROI. | Ảnh trung gian, Parameter Sweep Chương 5 |
| [`06_evaluation_tv6.ipynb`](notebooks/06_evaluation_tv6.ipynb) | **Đánh giá định lượng tổng thể:** Ma trận nhầm lẫn (Confusion Matrix), Sai số MAE vận tốc, phân tích độ nhạy và kiểm chứng giả thuyết. | Đánh giá định lượng toàn hệ thống |

---

## 📁 7. Cấu Trúc Thư Mục Dự Án

```
traffic-congestion-cv/
├── configs/
│   ├── toadovideo.py                 # Tọa độ 4 đỉnh ROI hình thang phối cảnh chuẩn từng video
│   └── traffic_dataset.yaml          # Cấu hình dataset 4 nhóm phương tiện COCO-PCU
├── data/
│   ├── raw/                          # Video camera giao thông thực tế (.mp4)
│   │   ├── traffic_congested.mp4     # Video phố New York kẹt cứng
│   │   ├── traffic_free_flow.mp4     # Video cao tốc thông thoáng
│   │   └── traffic_traffic_light.mp4 # Video ngã tư Hà Nội đèn tín hiệu
│   └── processed/                    # Thư mục lưu trữ ảnh kết quả nghiệm thu
├── demo/
│   └── app.py                        # Web Dashboard Streamlit đa luồng (TV7)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                    # Module tính Confusion Matrix, F1, Speed MAE (TV6)
├── modules/
│   ├── __init__.py
│   ├── congestion_evaluator.py       # Bộ tính toán chỉ số TCI & phân cấp mức độ (TV1)
│   ├── preprocessing.py              # Tiền xử lý ảnh CLAHE & Nắn góc BEV (TV2)
│   ├── segmentation.py               # Phân đoạn MOG2, ROI phối cảnh & Hybrid Occupancy (TV3)
│   ├── optical_flow.py               # Đo vận tốc Farneback Optical Flow & Heatmap (TV4)
│   └── detection.py                  # Phát hiện YOLOv8, PCU & Lọc xe theo ROI (TV5)
├── notebooks/                        # 6 Sổ ghi chép thí nghiệm Lab đạt chuẩn Đề cương UTH
│   ├── 01_tci_parameter_sweep.ipynb
│   ├── 02_preprocessing_sweep.ipynb
│   ├── 03_segmentation_roi_sweep.ipynb
│   ├── 04_optical_flow_tv4.ipynb
│   ├── 05_detection_tv5.ipynb
│   └── 06_evaluation_tv6.ipynb
├── config.py                         # Tham số trọng số TCI, ngưỡng mức độ và hệ số PCU
├── main.py                           # Điểm khởi chạy pipeline desktop chính (OpenCV CLI)
├── train.py                          # Script huấn luyện & fine-tuning đa video YOLOv8
├── requirements.txt                  # Danh mục các thư viện phụ thuộc
└── README.md                         # Báo cáo thuyết minh & Tài liệu hướng dẫn sử dụng
```

---

## 📊 8. Bảng Tổng Hợp Kết Quả Thực Nghiệm Trên 3 Video Thực Tế

| Chỉ số định lượng | Video 1: Thông thoáng (`traffic_free_flow`) | Video 2: Đèn tín hiệu (`traffic_traffic_light`) | Video 3: Ùn tắc (`traffic_congested`) |
| :--- | :---: | :---: | :---: |
| **Độ phân giải video** | $1920 \times 1080$ ($3640\text{ frames}$) | $1918 \times 970$ ($1743\text{ frames}$) | $1920 \times 1080$ ($1397\text{ frames}$) |
| **Vùng quan sát ROI** | 2 làn chính giữa | Làn đường bên phải (hướng lưu thông) | Lòng đường 2 chiều New York |
| **Xe nhận diện trong ROI** | $3 / 6\text{ xe}$ (loại bỏ $100\%$ xe ngược chiều) | $12 / 21\text{ xe}$ (loại bỏ $100\%$ xe ngược chiều) | $59 / 59\text{ xe}$ (bám sát lòng đường) |
| **Diện tích chiếm dụng (Occupancy)** | **$0.96\%$** | **$27.03\%$** | **$67.15\%$** |
| **Vận tốc trung bình ước lượng** | **$68.2\text{ km/h}$** | **$24.5\text{ km/h}$** | **$6.8\text{ km/h}$** |
| **Tải trọng PCU tương đương** | $4.50\text{ PCU}$ | $11.82\text{ PCU}$ | $48.20\text{ PCU}$ |
| **Chỉ số ùn tắc tổng hợp TCI** | **$0.04$** | **$0.42$** | **$0.76$** |
| **Cấp độ phân loại thực tế** | **Mức 1 · Thông thoáng** | **Mức 2 · Bình thường** | **Mức 4 · Tắc nghẽn** |
| **Tốc độ xử lý (FPS)** | $28.5\text{ FPS}$ | $31.2\text{ FPS}$ | $26.8\text{ FPS}$ |

---

*Bản quyền học phần Xử lý ảnh & Thị giác máy tính (121036) — Khoa Công nghệ Thông tin — Trường ĐH Giao thông Vận tải TP.HCM (UTH).*
