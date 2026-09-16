import cv2
import numpy as np
import time
from modules.preprocessing import preprocess_frame, get_perspective_bev
from modules.segmentation import RoadSegmenter
from modules.optical_flow import MotionEstimator
from modules.detection import VehicleDetector
from modules.congestion_evaluator import TrafficCongestionEvaluator

def draw_hud(frame, result, fps):
    """Vẽ bảng thông tin đo đạc trực quan lên màn hình"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (420, 190), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    metrics = result.get("metrics", {})
    cv2.putText(frame, "HE THONG GIAM SAT GIAO THONG - UTH", (35, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Muc do: {result['level']}", (35, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, result["color_bgr"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"Chi so TCI: {result['smoothed_tci']:.2f} (Tuc thoi: {result['instant_tci']:.2f})", 
                (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Chiem dung mat duong: {metrics.get('occupancy', 0.0)*100:.1f}%", 
                (35, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Ty le toc do: {metrics.get('speed_norm', 0.0)*100:.1f}% | FPS: {fps:.1f}", 
                (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

import sys
import os

# Cấu hình encoding UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    """
    Chạy pipeline giám sát giao thông
    Hỗ trợ webcam (0), file video cục bộ, hoặc link YouTube.
    """
    video_source = 0
    max_frames = None
    headless = "--headless" in sys.argv or not sys.stdin.isatty()

    for arg in sys.argv[1:]:
        if arg.startswith("--max-frames="):
            max_frames = int(arg.split("=")[1])
        elif not arg.startswith("-"):
            video_source = int(arg) if arg.isdigit() else arg
        
    is_vidgear = False
    if isinstance(video_source, str) and ('youtube.com' in video_source or 'youtu.be' in video_source):
        try:
            from vidgear.gears import CamGear
            print(f"[THONG TIN] Dang tai luong video tu YouTube: {video_source}")
            options = {"STREAM_RESOLUTION": "720p"}
            stream = CamGear(source=video_source, stream_mode=True, logging=False, **options).start()
            is_vidgear = True
        except ImportError:
            print("[LOI] Vui long cai dat vidgear: pip install vidgear yt-dlp")
            return
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            default_video = os.path.join("data", "raw", "traffic_congested.mp4")
            if os.path.exists(default_video):
                print(f"[THONG BAO] Nguon video {video_source} khong kha dung. Chuyen sang video mac dinh: {default_video}")
                video_source = default_video
                cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"[LOI] Khong the mo nguon video: {video_source}")
                return

    # Khởi tạo các module từ TV2 -> TV5 và bộ tính TCI của TV1
    segmenter = RoadSegmenter()
    motion_est = MotionEstimator()
    detector = VehicleDetector()
    evaluator = TrafficCongestionEvaluator()

    prev_time = time.time()

    while True:
        if is_vidgear:
            frame = stream.read()
            if frame is None:
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # Tính FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 1. Tiền xử lý (TV2)
        enhanced_frame = preprocess_frame(frame)

        # 2. Phân đoạn & đo diện tích chiếm dụng mặt đường (TV3)
        occupancy_ratio, occ_mask = segmenter.extract_occupancy(enhanced_frame)

        # 3. Ước lượng vận tốc qua Optical Flow (TV4)
        avg_speed, flow_mag = motion_est.estimate_speed(enhanced_frame)

        # 4. Phát hiện đối tượng & quy đổi tải trọng PCU (TV5)
        pcu_count, detections = detector.detect_and_count_pcu(frame)

        # 5. Đánh giá mức độ ùn tắc TCI (TV1)
        result = evaluator.compute_tci(
            occupancy_ratio=occupancy_ratio,
            avg_speed=avg_speed,
            pcu_count=pcu_count
        )

        # Vẽ Bounding Box nhận diện xe (TV5) và HUD đánh giá ùn tắc (TV1)
        frame = detector.draw_detections(frame, detections)
        draw_hud(frame, result, fps)

        # Tạo Heatmap hiển thị Optical Flow (TV4)
        flow_mag[flow_mag < motion_est.noise_threshold] = 0
        mag_clipped = np.clip(flow_mag, 0, 5.0)
        mag_uint8 = np.uint8(mag_clipped * (255.0 / 5.0))
        heatmap = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)

        frame_idx = getattr(main, "_frame_idx", 0) + 1
        setattr(main, "_frame_idx", frame_idx)

        # Lưu ảnh snapshot đại diện
        if frame_idx == 30 or getattr(main, "_saved_snapshot", False) is False:
            os.makedirs("data/processed", exist_ok=True)
            cv2.imwrite("data/processed/main_pipeline_result.jpg", frame)
            setattr(main, "_saved_snapshot", True)

        if not headless:
            cv2.imshow("Traffic Congestion Pipeline - UTH", frame)
            cv2.imshow("Occupancy Mask (Ch.4)", occ_mask)
            cv2.imshow("Optical Flow Heatmap (TV4)", heatmap)

            # Nhấn phím 'q' trên màn hình hiển thị để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if frame_idx % 20 == 0 or frame_idx == 1:
                print(f"[Pipeline Frame {frame_idx:03d}] TCI: {result['smoothed_tci']:.2f} ({result['level']}) | PCU: {pcu_count:.1f} | FPS: {fps:.1f}")
            if max_frames and frame_idx >= max_frames:
                print(f"[INFO] Da xu ly {max_frames} frames thanh cong!")
                break

    if is_vidgear:
        stream.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()