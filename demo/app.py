"""
Hệ thống Giám sát & Đánh giá Mức độ Ùn tắc Giao thông Đô thị
File: demo/app.py
Module TV7: Dashboard Web Streamlit hiển thị Demo & Kiểm thử
"""

import os
import sys
import time
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.preprocessing import preprocess_frame, get_perspective_bev
from modules.segmentation import RoadSegmenter
from modules.optical_flow import MotionEstimator
from modules.detection import VehicleDetector
from modules.congestion_evaluator import TrafficCongestionEvaluator
from config import PCU_WEIGHTS, WEIGHT_OCCUPANCY, WEIGHT_SPEED, WEIGHT_PCU, FREE_FLOW_SPEED_KMH, MAX_PCU_CAPACITY

try:
    from configs.toadovideo import VIDEO_CONFIG
except ImportError:
    VIDEO_CONFIG = {}

LEVEL_META = {
    "1 - Thong thoang": {
        "text": "Mức 1 · Thông thoáng",
        "css_class": "status-level-1",
        "hex": "#34d399"
    },
    "2 - Binh thuong": {
        "text": "Mức 2 · Bình thường",
        "css_class": "status-level-2",
        "hex": "#facc15"
    },
    "3 - Un u": {
        "text": "Mức 3 · Ùn ứ",
        "css_class": "status-level-3",
        "hex": "#fb923c"
    },
    "4 - Tac nghen": {
        "text": "Mức 4 · Tắc nghẽn",
        "css_class": "status-level-4",
        "hex": "#f87171"
    }
}


# ==============================================================================
# PIPELINE LÕI (DASHBOARD PIPELINE)
# ==============================================================================
class DashboardPipeline:
    def __init__(
        self,
        video_source="data/raw/traffic_congested.mp4",
        conf_thresh=0.22,
        iou_thresh=0.45,
        flow_noise_thresh=1.0,
        mog_history=500,
        mog_var_thresh=16.0,
        w_occ=WEIGHT_OCCUPANCY,
        w_spd=WEIGHT_SPEED,
        w_pcu=WEIGHT_PCU,
        bev_size=(640, 480)
    ):
        self.video_source = video_source
        self.bev_size = bev_size
        self.src_pts = None

        self.segmenter = RoadSegmenter(
            method="MOG2",
            history=mog_history,
            var_threshold=mog_var_thresh,
            detect_shadows=True,
            min_contour_area=150
        )
        self.motion_est = MotionEstimator(noise_threshold=flow_noise_thresh)
        self.detector = VehicleDetector(
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            imgsz=1280,
            high_accuracy=True
        )
        self.evaluator = TrafficCongestionEvaluator(
            w_occ=w_occ,
            w_spd=w_spd,
            w_pcu=w_pcu,
            v_free=FREE_FLOW_SPEED_KMH,
            max_pcu_cap=MAX_PCU_CAPACITY
        )

        self.cap = None
        self.fps_history = []
        self.prev_time = time.time()
        self.frame_count = 0

        self._init_video_capture()

    def _init_video_capture(self):
        if isinstance(self.video_source, int) or (isinstance(self.video_source, str) and self.video_source.isdigit()):
            self.cap = cv2.VideoCapture(int(self.video_source))
            video_key = "webcam"
        elif isinstance(self.video_source, str):
            self.cap = cv2.VideoCapture(self.video_source)
            video_key = os.path.basename(self.video_source)
        else:
            self.cap = None
            video_key = ""

        if video_key in VIDEO_CONFIG:
            self.src_pts = VIDEO_CONFIG[video_key]["src_pts"]
        else:
            self.src_pts = None

    def process_frame(self, frame=None):
        if frame is None:
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if not ret:
                # Tự động lặp lại video (loop)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None

        h, w = frame.shape[:2]
        self.frame_count += 1

        now = time.time()
        dt = now - self.prev_time
        fps = 1.0 / dt if dt > 0 else 30.0
        self.prev_time = now
        self.fps_history.append(fps)
        if len(self.fps_history) > 20:
            self.fps_history.pop(0)
        smoothed_fps = float(np.mean(self.fps_history))

        src_pts = self.src_pts
        if src_pts is None:
            src_pts = np.float32([
                [int(w * 0.38), int(h * 0.35)],
                [int(w * 0.62), int(h * 0.35)],
                [int(w * 0.90), int(h * 0.95)],
                [int(w * 0.10), int(h * 0.95)]
            ])
            self.src_pts = src_pts

        # 1. TV2: BEV
        enhanced_frame = preprocess_frame(frame)
        bev_frame, _ = get_perspective_bev(frame, src_pts, output_size=self.bev_size)

        # 2. TV3: Occupancy Mask
        occupancy_ratio, occ_mask = self.segmenter.extract_occupancy(enhanced_frame)
        occ_mask_bgr = cv2.cvtColor(occ_mask, cv2.COLOR_GRAY2BGR)

        # 3. TV4: Optical Flow Heatmap
        avg_speed, flow_mag = self.motion_est.estimate_speed(enhanced_frame)
        flow_mag_filtered = flow_mag.copy()
        flow_mag_filtered[flow_mag_filtered < self.motion_est.noise_threshold] = 0.0
        mag_clipped = np.clip(flow_mag_filtered, 0, 5.0)
        mag_uint8 = np.uint8(mag_clipped * (255.0 / 5.0))
        flow_heatmap = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)

        # 4. TV5: Detection & PCU
        total_pcu, vehicle_counts, detections = self.detector.detect_and_count_pcu(frame, return_counts=True)
        annotated_frame = self.detector.draw_detections(frame, detections, draw_hud=False)

        # 5. TV1: TCI
        tci_result = self.evaluator.compute_tci(
            occupancy_ratio=occupancy_ratio,
            avg_speed=avg_speed,
            pcu_count=total_pcu
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
            "total_vehicles": sum(vehicle_counts.values())
        }

        return {
            "window1_original_bbox": annotated_frame,
            "window2_bev": bev_frame,
            "window3_occ_mask": occ_mask_bgr,
            "window4_flow_heatmap": flow_heatmap,
            "central_metrics": central_metrics
        }

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ==============================================================================
# GIAO DIỆN STREAMLIT (TỐI GIẢN - KHOẢNG CÁCH RỘNG RÃI - SKELETON SHIMMER)
# ==============================================================================
def main():
    import streamlit as st
    import pandas as pd

    st.set_page_config(
        page_title="Giám sát Ùn tắc Giao thông",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS tối giản, thanh lịch, tạo khoảng cách thông thoáng rõ ràng giữa các phần tử
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Tránh bị navbar Streamlit đè lên tiêu đề */
        header[data-testid="stHeader"] {
            background-color: rgba(15, 23, 42, 0.75) !important;
            backdrop-filter: blur(8px) !important;
        }

        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 96% !important;
        }

        /* Header gọn gàng với khoảng cách dưới rộng rãi */
        .header-title {
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: #f8fafc;
            margin: 0 0 24px 0;
            padding-bottom: 12px;
            border-bottom: 1px solid #1e293b;
        }

        /* Hiệu ứng Skeleton Shimmer */
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        /* Thẻ Metric tối giản - chiều cao đồng bộ 68px */
        .metric-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 10px 14px;
            height: 68px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .metric-title {
            font-size: 0.70rem;
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

        /* Skeleton Metric */
        .skeleton-metric {
            height: 68px;
            background: linear-gradient(90deg, #0f172a 25%, #1e293b 50%, #0f172a 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 8px;
            border: 1px solid #1e293b;
        }

        /* Status Pill */
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
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.28);
        }
        .status-level-2 {
            background: rgba(234, 179, 8, 0.12);
            color: #facc15;
            border: 1px solid rgba(234, 179, 8, 0.28);
        }
        .status-level-3 {
            background: rgba(249, 115, 22, 0.12);
            color: #fb923c;
            border: 1px solid rgba(249, 115, 22, 0.28);
        }
        .status-level-4 {
            background: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.28);
        }

        /* Thanh tiêu đề cửa sổ video */
        .viewport-bar {
            background: #1e293b;
            padding: 8px 14px;
            font-size: 0.80rem;
            font-weight: 500;
            color: #cbd5e1;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border: 1px solid #1e293b;
            border-bottom: 1px solid #334155;
            margin-bottom: 0px;
        }

        /* Khung hình hiển thị */
        div[data-testid="stImage"] > img {
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            border: 1px solid #1e293b;
            border-top: none;
            display: block;
        }

        /* Skeleton Viewport cho 4 ô video */
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

    # Tiêu đề chính
    st.markdown('<div class="header-title">Hệ thống Giám sát & Đánh giá Ùn tắc Giao thông</div>', unsafe_allow_html=True)

    # ==========================================================================
    # SIDEBAR: CẤU HÌNH GỌN
    # ==========================================================================
    video_options = {
        "Ùn tắc (traffic_congested.mp4)": "data/raw/traffic_congested.mp4",
        "Thông thoáng (traffic_free_flow.mp4)": "data/raw/traffic_free_flow.mp4",
        "Đèn tín hiệu (traffic_traffic_light.mp4)": "data/raw/traffic_traffic_light.mp4",
        "Tải lên tệp video...": "upload",
        "Webcam máy tính": "0"
    }
    selected_option = st.sidebar.selectbox("Nguồn video:", list(video_options.keys()))
    video_source = video_options[selected_option]

    uploaded_file = None
    if video_source == "upload":
        uploaded_file = st.sidebar.file_uploader("Chọn file video:", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_path = "data/raw/_temp_uploaded_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            video_source = temp_path

    with st.sidebar.expander("Tham số nâng cao", expanded=False):
        conf_thresh = st.slider("Confidence:", 0.10, 0.90, 0.22, 0.02)
        iou_thresh = st.slider("IoU NMS:", 0.20, 0.80, 0.45, 0.05)
        flow_noise = st.slider("Lọc nhiễu vận tốc:", 0.2, 3.0, 1.0, 0.1)
        mog_history = st.slider("MOG2 History:", 100, 1000, 500, 50)
        mog_var = st.slider("MOG2 VarThreshold:", 8.0, 32.0, 16.0, 2.0)

    st.sidebar.markdown("---")
    if "is_running" not in st.session_state:
        st.session_state.is_running = True

    col_b1, col_b2 = st.sidebar.columns(2)
    if col_b1.button("Tiếp tục" if not st.session_state.is_running else "Đang chạy", use_container_width=True, type="primary"):
        st.session_state.is_running = True
    if col_b2.button("Tạm dừng", use_container_width=True):
        st.session_state.is_running = False

    skip_frames = st.sidebar.slider("Bước nhảy frame:", 1, 5, 1)

    # ==========================================================================
    # BẢNG ĐIỀU KHIỂN TRUNG TÂM (KHOẢNG CÁCH GAP VỪA PHẢI)
    # ==========================================================================
    hud_placeholder = st.empty()

    # Khoảng cách giữa Bảng điều khiển và Lưới video
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # ==========================================================================
    # 4 CỬA SỔ HIỂN THỊ (2x2 GRID - KHOẢNG CÁCH RÕ RÀNG)
    # ==========================================================================
    # HÀNG 1: Camera gốc & BEV
    col_row1_1, col_row1_2 = st.columns(2, gap="large")
    with col_row1_1:
        st.markdown('<div class="viewport-bar">Camera gốc & Bounding box</div>', unsafe_allow_html=True)
        placeholder_win1 = st.empty()
    with col_row1_2:
        st.markdown('<div class="viewport-bar">Bird\'s-Eye View (BEV)</div>', unsafe_allow_html=True)
        placeholder_win2 = st.empty()

    # Khoảng cách giữa Hàng 1 và Hàng 2 (Tách biệt 28px)
    st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)

    # HÀNG 2: Occupancy Mask & Optical Flow
    col_row2_1, col_row2_2 = st.columns(2, gap="large")
    with col_row2_1:
        st.markdown('<div class="viewport-bar">Mặt nạ chiếm dụng mặt đường</div>', unsafe_allow_html=True)
        placeholder_win3 = st.empty()
    with col_row2_2:
        st.markdown('<div class="viewport-bar">Bản đồ nhiệt vận tốc</div>', unsafe_allow_html=True)
        placeholder_win4 = st.empty()

    # Khoảng cách trước vùng đồ thị
    st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem; font-weight:600; color:#cbd5e1; margin-bottom:12px;">Diễn biến chỉ số theo thời gian</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([2, 1], gap="medium")
    with chart_col1:
        chart_placeholder = st.empty()
    with chart_col2:
        vehicle_chart_placeholder = st.empty()

    # ==========================================================================
    # SKELETON LOADING KHI THAY ĐỔI CẤU HÌNH HOẶC LOAD TRANG
    # ==========================================================================
    current_config = {
        "source": video_source,
        "conf": conf_thresh,
        "iou": iou_thresh,
        "noise": flow_noise,
        "history": mog_history,
        "var": mog_var
    }

    if "last_config" not in st.session_state or st.session_state.last_config != current_config:
        st.session_state.last_config = current_config
        with hud_placeholder.container():
            s_cols = st.columns(6, gap="medium")
            for sc in s_cols:
                with sc:
                    st.markdown('<div class="skeleton-metric"></div>', unsafe_allow_html=True)

        placeholder_win1.markdown('<div class="skeleton-viewport"><span>Đang nạp luồng camera...</span></div>', unsafe_allow_html=True)
        placeholder_win2.markdown('<div class="skeleton-viewport"><span>Đang nắn phối cảnh BEV...</span></div>', unsafe_allow_html=True)
        placeholder_win3.markdown('<div class="skeleton-viewport"><span>Đang khởi tạo mặt nạ nền...</span></div>', unsafe_allow_html=True)
        placeholder_win4.markdown('<div class="skeleton-viewport"><span>Đang đo Optical Flow...</span></div>', unsafe_allow_html=True)

    # ==========================================================================
    # VÒNG LẶP XỬ LÝ VIDEO
    # ==========================================================================
    if st.session_state.is_running:
        if video_source == "upload" and uploaded_file is None:
            st.info("Vui lòng tải lên tệp video.")
            return

        pipeline = DashboardPipeline(
            video_source=video_source,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            flow_noise_thresh=flow_noise,
            mog_history=mog_history,
            mog_var_thresh=mog_var
        )

        history_tci = []
        history_occ = []
        history_spd = []
        frame_idx = 0

        # Kích thước cố định chuẩn 16:9 đảm bảo 4 ô hoàn toàn bằng nhau
        disp_w, disp_h = 640, 360

        while st.session_state.is_running:
            result = pipeline.process_frame()
            if result is None:
                st.session_state.is_running = False
                break

            frame_idx += 1
            if frame_idx % skip_frames != 0:
                continue

            metrics = result["central_metrics"]

            # Cập nhật HUD chỉ số với khoảng cách gap="medium"
            with hud_placeholder.container():
                m1, m2, m3, m4, m5, m6 = st.columns(6, gap="medium")
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Tốc độ</div>
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
                        <div class="metric-title">Chiếm dụng</div>
                        <div class="metric-num">{metrics['occupancy_pct']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Vận tốc</div>
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
                        <div class="metric-title">Mức độ</div>
                        <div class="status-box">
                            <div class="status-pill {metrics['level_class']}">{metrics['level_label']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Cập nhật 4 Cửa sổ hiển thị - Đồng bộ kích thước chính xác 640x360
            img1_rgb = cv2.cvtColor(cv2.resize(result["window1_original_bbox"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(cv2.resize(result["window2_bev"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
            img3_rgb = cv2.cvtColor(cv2.resize(result["window3_occ_mask"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
            img4_rgb = cv2.cvtColor(cv2.resize(result["window4_flow_heatmap"], (disp_w, disp_h)), cv2.COLOR_BGR2RGB)

            placeholder_win1.image(img1_rgb, use_container_width=True)
            placeholder_win2.image(img2_rgb, use_container_width=True)
            placeholder_win3.image(img3_rgb, use_container_width=True)
            placeholder_win4.image(img4_rgb, use_container_width=True)

            # Biểu đồ thời gian thực
            history_tci.append(metrics["smoothed_tci"])
            history_occ.append(metrics["occupancy_ratio"])
            history_spd.append(metrics["speed_norm_pct"] / 100.0)

            if len(history_tci) > 60:
                history_tci.pop(0)
                history_occ.pop(0)
                history_spd.pop(0)

            df_trends = pd.DataFrame({
                "TCI": history_tci,
                "Chiếm dụng": history_occ,
                "Vận tốc": history_spd
            })
            chart_placeholder.line_chart(df_trends, height=190)

            df_vehicles = pd.DataFrame({
                "Nhóm xe": ["Xe máy", "Ô tô", "Xe buýt", "Xe tải"],
                "Số lượng": [
                    metrics["vehicle_counts"].get("motorcycle", 0),
                    metrics["vehicle_counts"].get("car", 0),
                    metrics["vehicle_counts"].get("bus", 0),
                    metrics["vehicle_counts"].get("truck", 0)
                ]
            }).set_index("Nhóm xe")
            vehicle_chart_placeholder.bar_chart(df_vehicles, height=190)

        pipeline.release()


if __name__ == "__main__":
    main()
