import cv2
import numpy as np
import sys
import os

# Thêm đường dẫn thư mục gốc vào sys.path để import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình encoding console UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modules.optical_flow import MotionEstimator

def test_optical_flow(video_source=0, max_frames=80):
    print(f"Bắt đầu test Optical Flow (TV4) với video source: {video_source}")
    
    is_vidgear = False
    if isinstance(video_source, str) and ('youtube.com' in video_source or 'youtu.be' in video_source):
        try:
            from vidgear.gears import CamGear
            print("Đang tải luồng video từ YouTube bằng vidgear (bền bỉ hơn)...")
            options = {"STREAM_RESOLUTION": "720p"}
            stream = CamGear(source=video_source, stream_mode=True, logging=False, **options).start()
            is_vidgear = True
        except ImportError:
            print("Lỗi: Vui lòng cài đặt thư viện bằng lệnh: pip install vidgear yt-dlp")
            return
    else:
        cap = cv2.VideoCapture(video_source)
        if cap is None or not cap.isOpened():
            print("Không thể mở nguồn video!")
            return

    # Khởi tạo module ước lượng vận tốc với threshold = 1.0
    estimator = MotionEstimator(noise_threshold=1.0)
    print("Nhấn 'q' để thoát.")

    while True:
        if is_vidgear:
            frame = stream.read()
            if frame is None:
                print("Hết luồng video hoặc lỗi kết nối.")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame.")
                break
            
        frame = cv2.resize(frame, (640, 480))
        
        # Gọi hàm tính toán vector chuyển động TV4
        avg_speed, magnitude = estimator.estimate_speed(frame)
        
        # Vẽ bản đồ nhiệt (Heatmap) cho các vector chuyển động
        # Loại bỏ nhiễu nhỏ cho việc hiển thị rõ hơn
        magnitude[magnitude < estimator.noise_threshold] = 0
        # Cố định ngưỡng max (clip) ở mức 5.0 để các xe ở xa (chuyển động nhỏ) vẫn hiển thị được màu cam/đỏ
        mag_clipped = np.clip(magnitude, 0, 5.0)
        mag_uint8 = np.uint8(mag_clipped * (255.0 / 5.0))
        heatmap = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)
        
        # Hiển thị vận tốc trung bình lên frame gốc
        text = f"Avg Speed (pixels/frame): {avg_speed:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        # Hiển thị
        is_interactive = sys.stdin.isatty() and "--headless" not in sys.argv
        if is_interactive:
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Optical Flow Heatmap (TV4)", heatmap)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        frame_count = getattr(test_optical_flow, "_frame_count", 0) + 1
        setattr(test_optical_flow, "_frame_count", frame_count)

        if frame_count % 30 == 0 or frame_count == 1:
            print(f"[Frame {frame_count:03d}] Avg Speed: {avg_speed:5.2f} px/frame")

        if frame_count == 40 or getattr(test_optical_flow, "_best_heatmap", None) is None:
            os.makedirs("data/processed", exist_ok=True)
            vis_combined = np.hstack([frame, heatmap])
            cv2.imwrite("data/processed/tv4_test_result.jpg", vis_combined)

        if not is_interactive and frame_count >= max_frames:
            print(f"[INFO] Da xu ly {max_frames} frames trong che do tu dong.")
            break
            
    if is_vidgear:
        stream.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("[OK] Da luu frame Optical Flow mau vao: data/processed/tv4_test_result.jpg")
    print("[THANH CONG] Kiem thu TV4 hoan tat!")

def test_optical_flow_wrapper(video_source=None, max_frames=80):
    if video_source is None:
        congested_path = os.path.join("data", "raw", "traffic_congested.mp4")
        if os.path.exists(congested_path):
            video_source = congested_path
        else:
            video_source = 0
    test_optical_flow(video_source, max_frames=max_frames)

if __name__ == "__main__":
    v_source = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        v_source = sys.argv[1]
    test_optical_flow_wrapper(v_source)
