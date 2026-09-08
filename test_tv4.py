import cv2
import numpy as np
import sys
import os

# Thêm đường dẫn thư mục gốc vào sys.path để import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.optical_flow import MotionEstimator

def test_optical_flow(video_source=0):
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
        # Loại bỏ nhiễu nhỏ cho việc hiển thị rõ hơn (tùy chọn)
        magnitude[magnitude < estimator.noise_threshold] = 0
        mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag_uint8 = np.uint8(mag_normalized)
        heatmap = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)
        
        # Hiển thị vận tốc trung bình lên frame gốc
        text = f"Avg Speed (pixels/frame): {avg_speed:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        # Hiển thị
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Optical Flow Heatmap (TV4)", heatmap)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 0 
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    test_optical_flow(video_path)
