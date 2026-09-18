"""
HỆ THỐNG GIÁM SÁT & ĐÁNH GIÁ MỨC ĐỘ ÙN TẮC GIAO THÔNG ĐÔ THỊ (UTH)
File: demo/app.py
Module TV7: Dashboard Web Trực quan hóa & Phân tích Đa luồng (Streamlit)
"""

import os
import sys
import time
import tempfile
import numpy as np
import cv2
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.preprocessing import preprocess_frame, get_perspective_bev
from modules.segmentation import RoadSegmenter
from modules.optical_flow import MotionEstimator
from modules.detection import VehicleDetector
from modules.congestion_evaluator import TrafficCongestionEvaluator
from config import (
    PCU_WEIGHTS,
    WEIGHT_OCCUPANCY,
    WEIGHT_SPEED,
    WEIGHT_PCU,
    FREE_FLOW_SPEED_KMH,
    MAX_PCU_CAPACITY,
)

try:
    from configs.toadovideo import VIDEO_CONFIG
except ImportError:
    VIDEO_CONFIG = {}

LEVEL_META = {
    "1 - Thong thoang": {
        "text": "Mức 1 · Thông thoáng",
        "css_class": "status-level-1",
        "hex": "#34d399",
        "badge_bg": "rgba(52, 211, 153, 0.12)",
    },
    "2 - Binh thuong": {
        "text": "Mức 2 · Bình thường",
        "css_class": "status-level-2",
        "hex": "#facc15",
        "badge_bg": "rgba(250, 204, 21, 0.12)",
    },
    "3 - Un u": {
        "text": "Mức 3 · Ùn ứ",
        "css_class": "status-level-3",
        "hex": "#fb923c",
        "badge_bg": "rgba(251, 146, 60, 0.12)",
    },
    "4 - Tac nghen": {
        "text": "Mức 4 · Tắc nghẽn",
        "css_class": "status-level-4",
        "hex": "#f87171",
        "badge_bg": "rgba(248, 113, 113, 0.12)",
    },
}


# ==============================================================================
# CACHING MÔ HÌNH YOLO (GIẢM THỜI GIAN NẠP & CHỐNG GIẬT LAG)
# ==============================================================================
@st.cache_resource(show_spinner="Đang nạp mô hình nhận diện phương tiện YOLOv8...")
def load_yolo_detector(model_name="yolov8n.pt"):
    """Nạp và lưu trữ đối tượng VehicleDetector trong bộ nhớ cache của Streamlit."""
    return VehicleDetector(
        model_path=model_name,
        conf_thresh=0.22,
        iou_thresh=0.45,
        imgsz=1280,
        high_accuracy=True,
    )


# ==============================================================================
# PIPELINE XỬ LÝ ĐA MODULE (DASHBOARD PIPELINE)
# ==============================================================================
class DashboardPipeline:
    def __init__(
        self,
        video_source="data/raw/traffic_congested.mp4",
        detector=None,
        conf_thresh=0.22,
        iou_thresh=0.45,
        flow_noise_thresh=1.0,
        mog_history=500,
        mog_var_thresh=16.0,
        w_occ=WEIGHT_OCCUPANCY,
        w_spd=WEIGHT_SPEED,
        w_pcu=WEIGHT_PCU,
        bev_size=(640, 480),
    ):
        self.video_source = video_source
        self.bev_size = bev_size
        self.src_pts = None

        self.segmenter = RoadSegmenter(
            method="MOG2",
            history=mog_history,
            var_threshold=mog_var_thresh,
            detect_shadows=True,
            min_contour_area=150,
        )
        self.motion_est = MotionEstimator(noise_threshold=flow_noise_thresh)

        # Tận dụng detector đã được cache hoặc tạo mới
        if detector is not None:
            self.detector = detector
            self.detector.conf_thresh = conf_thresh
            self.detector.iou_thresh = iou_thresh
        else:
            self.detector = VehicleDetector(
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                imgsz=1280,
                high_accuracy=True,
            )

        self.evaluator = TrafficCongestionEvaluator(
            w_occ=w_occ,
            w_spd=w_spd,
            w_pcu=w_pcu,
            v_free=FREE_FLOW_SPEED_KMH,
            max_pcu_cap=MAX_PCU_CAPACITY,
        )

        self.cap = None
        self.fps_history = []
        self.prev_time = time.time()
        self.frame_count = 0
        self.is_opened = False

        self._init_video_capture()

    def _init_video_capture(self):
        """Khởi tạo VideoCapture và ánh xạ toạ độ ROI chuẩn từ VIDEO_CONFIG."""
        if isinstance(self.video_source, int) or (
            isinstance(self.video_source, str) and self.video_source.isdigit()
        ):
            self.cap = cv2.VideoCapture(int(self.video_source))
            video_key = "webcam"
        elif isinstance(self.video_source, str):
            self.cap = cv2.VideoCapture(self.video_source)
            video_key = os.path.basename(self.video_source)
        else:
            self.cap = None
            video_key = ""

        if self.cap is not None and self.cap.isOpened():
            self.is_opened = True
        else:
            self.is_opened = False

        if video_key in VIDEO_CONFIG:
            self.src_pts = VIDEO_CONFIG[video_key]["src_pts"]
        else:
            self.src_pts = None

    def process_frame(self, frame=None):
        """Xử lý 1 khung hình đồng thời qua 5 module và tổng hợp kết quả."""
        if frame is None:
            if not self.is_opened or self.cap is None:
                return None
            ret, frame = self.cap.read()
            if not ret:
                # Tự động lặp lại video từ đầu (loop) nếu đọc từ file
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None

        h, w = frame.shape[:2]
        self.frame_count += 1

        # Đo đạc FPS thời gian thực
        now = time.time()
        dt = now - self.prev_time
        fps = 1.0 / dt if dt > 0 else 30.0
        self.prev_time = now
        self.fps_history.append(fps)
        if len(self.fps_history) > 20:
            self.fps_history.pop(0)
        smoothed_fps = float(np.mean(self.fps_history))

        # 1. TV5: Nhận dạng đối tượng & Quy đổi tải trọng PCU (Chạy trước để cung cấp dữ liệu cho Hybrid Occupancy)
        total_pcu, vehicle_counts, detections = self.detector.detect_and_count_pcu(
            frame, return_counts=True
        )

        # Tự động ước lượng ROI đa giác nếu chưa có cấu hình sẵn trong VIDEO_CONFIG
        if self.src_pts is None:
            if self.segmenter.roi_mask is None:
                self.segmenter.auto_estimate_road_roi((h, w), detections)
            self.src_pts = np.float32([
                [int(w * 0.32), int(h * 0.28)],
                [int(w * 0.68), int(h * 0.28)],
                [int(w * 0.98), int(h * 0.98)],
                [int(w * 0.02), int(h * 0.98)],
            ])
        elif self.segmenter.roi_mask is None:
            self.segmenter.set_roi_polygon((h, w), self.src_pts)

        # 2. TV2: Tiền xử lý & Nắn phối cảnh Bird's-Eye View (BEV)
        enhanced_frame = preprocess_frame(frame)
        bev_frame, _ = get_perspective_bev(frame, self.src_pts, output_size=self.bev_size)

        # 3. TV3: Trừ nền thích nghi & Đo diện tích chiếm dụng lai (Hybrid Occupancy: MOG2 + YOLO)
        occupancy_ratio, occ_mask = self.segmenter.extract_occupancy(
            enhanced_frame, detections=detections
        )
        occ_mask_bgr = cv2.cvtColor(occ_mask, cv2.COLOR_GRAY2BGR)
        # Vẽ viền vùng quan sát mặt đường (ROI) lên mặt nạ để dễ theo dõi
        if self.src_pts is not None:
            cv2.polylines(
                occ_mask_bgr,
                [np.int32(self.src_pts)],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )

        # 4. TV4: Ước lượng vận tốc dòng xe qua Optical Flow Heatmap
        avg_speed, flow_mag = self.motion_est.estimate_speed(enhanced_frame)
        flow_mag_filtered = flow_mag.copy()
        flow_mag_filtered[flow_mag_filtered < self.motion_est.noise_threshold] = 0.0
        mag_clipped = np.clip(flow_mag_filtered, 0, 5.0)
        mag_uint8 = np.uint8(mag_clipped * (255.0 / 5.0))
        flow_heatmap = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)

        annotated_frame = self.detector.draw_detections(frame, detections, draw_hud=False)
        # Vẽ viền ROI tinh tế lên camera gốc để chỉ rõ khu vực kiểm soát
        if self.src_pts is not None:
            cv2.polylines(
                annotated_frame,
                [np.int32(self.src_pts)],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )

        # 5. TV1: Tính toán Chỉ số Ùn tắc TCI & Phân cấp mức độ
        tci_result = self.evaluator.compute_tci(
            occupancy_ratio=occupancy_ratio,
            avg_speed=avg_speed,
            pcu_count=total_pcu,
        )


        level_raw = tci_result["level"]
        meta = LEVEL_META.get(level_raw, LEVEL_META["1 - Thong thoang"])

        central_metrics = {
            "instant_tci": tci_result["instant_tci"],
            "smoothed_tci": tci_result["smoothed_tci"],
            "level": level_raw,
            "level_label": meta["text"],
            "level_class": meta["css_class"],
            "level_color_hex": meta["hex"],
            "fps": round(smoothed_fps, 1),
            "frame_idx": self.frame_count,
            "occupancy_ratio": round(occupancy_ratio, 3),
            "occupancy_pct": round(occupancy_ratio * 100, 1),
            "avg_speed": round(avg_speed, 2),
            "speed_norm_pct": round(tci_result["metrics"]["speed_norm"] * 100, 1),
            "total_pcu": round(total_pcu, 2),
            "pcu_density_pct": round(tci_result["metrics"]["pcu_density"] * 100, 1),
            "vehicle_counts": vehicle_counts,
            "total_vehicles": sum(vehicle_counts.values()),
        }

        return {
            "window1_original_bbox": annotated_frame,
            "window2_bev": bev_frame,
            "window3_occ_mask": occ_mask_bgr,
            "window4_flow_heatmap": flow_heatmap,
            "central_metrics": central_metrics,
        }

    def release(self):
        """Giải phóng tài nguyên VideoCapture."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False


# ==============================================================================
# GIAO DIỆN CHÍNH (STREAMLIT WEB DASHBOARD)
# ==============================================================================
def main():
    st.set_page_config(
        page_title="Hệ thống Giám sát Ùn tắc Giao thông - UTH",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS tối giản, hiện đại, khoảng cách rộng rãi và bố cục sắc nét
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        header[data-testid="stHeader"] {
            background-color: rgba(15, 23, 42, 0.75) !important;
            backdrop-filter: blur(8px) !important;
        }

        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 96% !important;
        }

        .header-title-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 1px solid #1e293b;
        }

        .header-title {
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: #f8fafc;
            margin: 0;
        }

        .header-subtitle {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-top: 4px;
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        /* Thẻ HUD Metric trung tâm */
        .metric-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 10px 14px;
            height: 72px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .metric-title {
            font-size: 0.72rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 2px;
        }
        .metric-num {
            font-size: 1.40rem;
            font-weight: 600;
            color: #f8fafc;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .status-box {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }
        .status-pill {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 7px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            text-align: center;
        }
        .status-level-1 {
            background: rgba(16, 185, 129, 0.14);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.32);
        }
        .status-level-2 {
            background: rgba(234, 179, 8, 0.14);
            color: #facc15;
            border: 1px solid rgba(234, 179, 8, 0.32);
        }
        .status-level-3 {
            background: rgba(249, 115, 22, 0.14);
            color: #fb923c;
            border: 1px solid rgba(249, 115, 22, 0.32);
        }
        .status-level-4 {
            background: rgba(239, 68, 68, 0.14);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.32);
        }

        /* Thanh tiêu đề 4 cửa sổ video */
        .viewport-bar {
            background: #1e293b;
            padding: 8px 14px;
            font-size: 0.80rem;
            font-weight: 600;
            color: #cbd5e1;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border: 1px solid #1e293b;
            border-bottom: 1px solid #334155;
            margin-bottom: 0px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        div[data-testid="stImage"] > img {
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            border: 1px solid #1e293b;
            border-top: none;
            display: block;
        }

        .skeleton-metric {
            height: 72px;
            background: linear-gradient(90deg, #0f172a 25%, #1e293b 50%, #0f172a 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 8px;
            border: 1px solid #1e293b;
        }

        .skeleton-viewport {
            width: 100%;
            aspect-ratio: 16 / 9;
            background: linear-gradient(90deg, #070a12 25%, #1e293b 50%, #070a12 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            border: 1px solid #1e293b;
            border-top: none;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
            font-size: 0.82rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Tiêu đề Dashboard
    st.markdown("""
    <div class="header-title-box">
        <div>
            <div class="header-title">Hệ Thống Giám Sát & Đánh Giá Mức Độ Ùn Tắc Giao Thông</div>
            <div class="header-subtitle">Học phần Xử lý ảnh & Thị giác máy tính (121036) · ĐH Giao thông Vận tải TP.HCM (UTH)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # CẤU HÌNH SIDEBAR
    # ==========================================================================
    st.sidebar.markdown("### ⚙️ Cấu Hình Hệ Thống")

    video_options = {
        "Ùn tắc (traffic_congested.mp4)": "data/raw/traffic_congested.mp4",
        "Thông thoáng (traffic_free_flow.mp4)": "data/raw/traffic_free_flow.mp4",
        "Đèn tín hiệu (traffic_traffic_light.mp4)": "data/raw/traffic_traffic_light.mp4",
        "Tải lên tệp video...": "upload",
        "Webcam máy tính (Camera 0)": "0",
    }
    selected_option = st.sidebar.selectbox("Nguồn video quan sát:", list(video_options.keys()))
    video_source = video_options[selected_option]

    uploaded_file = None
    if video_source == "upload":
        uploaded_file = st.sidebar.file_uploader(
            "Chọn tệp video cục bộ:", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            # Lưu tạm vào tệp ẩn an toàn
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            video_source = tfile.name

    # Quản lý trạng thái điều khiển vòng lặp
    if "is_running" not in st.session_state:
        st.session_state.is_running = True
    if "current_video_source" not in st.session_state:
        st.session_state.current_video_source = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "last_snapshot" not in st.session_state:
        st.session_state.last_snapshot = None

    # Thanh trượt điều khiển tham số
    with st.sidebar.expander("🛠️ Tham số thuật toán", expanded=False):
        conf_thresh = st.slider("Ngưỡng tin cậy YOLO:", 0.10, 0.90, 0.22, 0.02)
        iou_thresh = st.slider("Ngưỡng IoU NMS:", 0.20, 0.80, 0.45, 0.05)
        flow_noise = st.slider("Lọc nhiễu vận tốc (px):", 0.2, 3.0, 1.0, 0.1)
        mog_history = st.slider("MOG2 History:", 100, 1000, 500, 50)
        mog_var = st.slider("MOG2 VarThreshold:", 8.0, 32.0, 16.0, 2.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏯️ Điều Khiển Phát")
    col_b1, col_b2 = st.sidebar.columns(2)
    if col_b1.button(
        "Tiếp tục" if not st.session_state.is_running else "Đang chạy",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.is_running = True
    if col_b2.button("Tạm dừng", use_container_width=True):
        st.session_state.is_running = False

    if st.sidebar.button("🔄 Chạy lại từ đầu", use_container_width=True):
        if st.session_state.pipeline is not None and st.session_state.pipeline.cap is not None:
            st.session_state.pipeline.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            st.session_state.is_running = True

    skip_frames = st.sidebar.slider("Bước nhảy frame (Tăng tốc):", 1, 5, 1)

    # Nạp detector được cache dùng chung (tránh nạp lại từ ổ cứng khi kéo slider)
    shared_detector = load_yolo_detector("yolov8n.pt")
    shared_detector.conf_thresh = conf_thresh
    shared_detector.iou_thresh = iou_thresh

    # ==========================================================================
    # KIỂM TRA TRẠNG THÁI VIDEO NẠP VÀO
    # ==========================================================================
    if video_source == "upload" and uploaded_file is None:
        st.info("💡 **Vui lòng chọn hoặc kéo thả tệp video ở cột bên trái để bắt đầu phân tích.**")
        return

    # Nếu nguồn video thay đổi, giải phóng pipeline cũ và tạo mới
    if st.session_state.current_video_source != video_source:
        if st.session_state.pipeline is not None:
            st.session_state.pipeline.release()
        st.session_state.pipeline = DashboardPipeline(
            video_source=video_source,
            detector=shared_detector,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            flow_noise_thresh=flow_noise,
            mog_history=mog_history,
            mog_var_thresh=mog_var,
        )
        st.session_state.current_video_source = video_source

        # Kiểm tra xem video có mở được không
        if not st.session_state.pipeline.is_opened:
            if video_source == "0":
                st.error("⚠️ **Không thể mở Webcam máy tính (Camera 0).** Vui lòng kiểm tra kết nối thiết bị hoặc cấp quyền sử dụng camera.")
            else:
                st.error(f"⚠️ **Không thể mở tệp video:** `{video_source}`. Vui lòng kiểm tra lại đường dẫn.")
            st.session_state.is_running = False
            return
    else:
        # Cập nhật nhanh các tham số động mà không cần nạp lại mô hình
        p = st.session_state.pipeline
        if p is not None:
            p.motion_est.noise_threshold = flow_noise
            if hasattr(p.segmenter, "bg_subtractor"):
                p.segmenter.bg_subtractor.setVarThreshold(mog_var)
                p.segmenter.bg_subtractor.setHistory(mog_history)

    pipeline = st.session_state.pipeline

    # ==========================================================================
    # BẢNG ĐIỀU KHIỂN CHỈ SỐ TRUNG TÂM (HUD)
    # ==========================================================================
    hud_placeholder = st.empty()
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

    # ==========================================================================
    # LƯỚI 4 CỬA SỔ HIỂN THỊ (2x2 GRID)
    # ==========================================================================
    col_row1_1, col_row1_2 = st.columns(2, gap="large")
    with col_row1_1:
        st.markdown(
            '<div class="viewport-bar"><span>1. Camera gốc & Nhận diện Bounding Box</span><span style="color:#64748b; font-size:0.75rem;">YOLOv8 + TV5</span></div>',
            unsafe_allow_html=True,
        )
        placeholder_win1 = st.empty()
    with col_row1_2:
        st.markdown(
            '<div class="viewport-bar"><span>2. Nắn góc nhìn Bird\'s-Eye View (BEV)</span><span style="color:#64748b; font-size:0.75rem;">Perspective Transform + TV2</span></div>',
            unsafe_allow_html=True,
        )
        placeholder_win2 = st.empty()

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    col_row2_1, col_row2_2 = st.columns(2, gap="large")
    with col_row2_1:
        st.markdown(
            '<div class="viewport-bar"><span>3. Mặt nạ chiếm dụng mặt đường (Occupancy)</span><span style="color:#64748b; font-size:0.75rem;">MOG2 Shadow Removal + TV3</span></div>',
            unsafe_allow_html=True,
        )
        placeholder_win3 = st.empty()
    with col_row2_2:
        st.markdown(
            '<div class="viewport-bar"><span>4. Bản đồ nhiệt vận tốc (Optical Flow)</span><span style="color:#64748b; font-size:0.75rem;">Farneback Heatmap + TV4</span></div>',
            unsafe_allow_html=True,
        )
        placeholder_win4 = st.empty()

    st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)

    # Vùng hiển thị biểu đồ diễn biến thời gian thực
    st.markdown(
        '<div style="font-size:0.88rem; font-weight:600; color:#cbd5e1; margin-bottom:12px;">📊 Diễn Biến Chỉ Số & Phân Bố Lưu Lượng Theo Thời Gian</div>',
        unsafe_allow_html=True,
    )
    chart_col1, chart_col2 = st.columns([2, 1], gap="medium")
    with chart_col1:
        chart_placeholder = st.empty()
    with chart_col2:
        vehicle_chart_placeholder = st.empty()

    # Khởi tạo các mảng theo dõi lịch sử
    if "history_tci" not in st.session_state:
        st.session_state.history_tci = []
        st.session_state.history_occ = []
        st.session_state.history_spd = []

    disp_w, disp_h = 640, 360
    frame_loop_count = 0

    # ==========================================================================
    # VÒNG LẶP XỬ LÝ & HIỂN THỊ STREAM
    # ==========================================================================
    while st.session_state.is_running:
        result = pipeline.process_frame()
        if result is None:
            st.session_state.is_running = False
            break

        frame_loop_count += 1
        if frame_loop_count % skip_frames != 0:
            continue

        metrics = result["central_metrics"]

        # Cập nhật HUD chỉ số trung tâm
        with hud_placeholder.container():
            m1, m2, m3, m4, m5, m6 = st.columns(6, gap="medium")
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Tốc độ xử lý</div>
                    <div class="metric-num">{metrics['fps']:.1f} <span style="font-size:0.75rem;color:#64748b;">FPS</span></div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Chỉ số TCI</div>
                    <div class="metric-num">{metrics['smoothed_tci']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Chiếm dụng đường</div>
                    <div class="metric-num">{metrics['occupancy_pct']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Vận tốc dòng xe</div>
                    <div class="metric-num">{metrics['speed_norm_pct']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Tải trọng PCU</div>
                    <div class="metric-num">{metrics['total_pcu']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Mức độ ùn tắc</div>
                    <div class="status-box">
                        <div class="status-pill {metrics['level_class']}">{metrics['level_label']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Chuẩn hóa kích thước 4 cửa sổ đồng bộ tỉ lệ 16:9
        img1_rgb = cv2.cvtColor(cv2.resize(result["window1_original_bbox"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(cv2.resize(result["window2_bev"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        img3_rgb = cv2.cvtColor(cv2.resize(result["window3_occ_mask"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        img4_rgb = cv2.cvtColor(cv2.resize(result["window4_flow_heatmap"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)

        placeholder_win1.image(img1_rgb, use_container_width=True)
        placeholder_win2.image(img2_rgb, use_container_width=True)
        placeholder_win3.image(img3_rgb, use_container_width=True)
        placeholder_win4.image(img4_rgb, use_container_width=True)

        # Lưu vết Canvas gần nhất để tải snapshot
        top_row = np.hstack([cv2.resize(result["window1_original_bbox"], (640, 360)), cv2.resize(result["window2_bev"], (640, 360))])
        bot_row = np.hstack([cv2.resize(result["window3_occ_mask"], (640, 360)), cv2.resize(result["window4_flow_heatmap"], (640, 360))])
        st.session_state.last_snapshot = np.vstack([top_row, bot_row])

        # Cập nhật mảng lịch sử đồ thị
        st.session_state.history_tci.append(metrics["smoothed_tci"])
        st.session_state.history_occ.append(metrics["occupancy_ratio"])
        st.session_state.history_spd.append(metrics["speed_norm_pct"] / 100.0)

        if len(st.session_state.history_tci) > 60:
            st.session_state.history_tci.pop(0)
            st.session_state.history_occ.pop(0)
            st.session_state.history_spd.pop(0)

        df_trends = pd.DataFrame({
            "Chỉ số TCI": st.session_state.history_tci,
            "Tỷ lệ chiếm dụng": st.session_state.history_occ,
            "Vận tốc chuẩn hóa": st.session_state.history_spd,
        })
        chart_placeholder.line_chart(df_trends, height=200)

        df_vehicles = pd.DataFrame({
            "Nhóm phương tiện": ["Xe máy (0.33)", "Ô tô (1.0)", "Xe buýt (2.5)", "Xe tải (3.0)"],
            "Số lượng": [
                metrics["vehicle_counts"].get("motorcycle", 0),
                metrics["vehicle_counts"].get("car", 0),
                metrics["vehicle_counts"].get("bus", 0),
                metrics["vehicle_counts"].get("truck", 0),
            ],
        }).set_index("Nhóm phương tiện")
        vehicle_chart_placeholder.bar_chart(df_vehicles, height=200)

    # ==========================================================================
    # TIỆN ÍCH XUẤT ẢNH NGHIỆM THU SNAPSHOT
    # ==========================================================================
    if st.session_state.last_snapshot is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📸 Nghiệm Thu Hệ Thống")
        is_success, buffer = cv2.imencode(".jpg", st.session_state.last_snapshot)
        if is_success:
            st.sidebar.download_button(
                label="Tải ảnh Canvas 2x2 (Snapshot)",
                data=buffer.tobytes(),
                file_name="traffic_dashboard_snapshot.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
