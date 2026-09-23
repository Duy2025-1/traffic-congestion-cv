"""
SCRIPT KIỂM THỬ TỰ ĐỘNG CHO MODULE TV7: DASHBOARD DEMO & KIỂM THỬ
Phụ trách: TV7
Học phần: Xử lý ảnh và Thị giác máy tính (121036) - UTH

Nội dung kiểm thử:
1. Kiểm tra khởi tạo và đồng bộ 5 module (TV1, TV2, TV3, TV4, TV5) trong DashboardPipeline.
2. Kiểm tra tính toàn vẹn của 4 cửa sổ hiển thị:
   - Cửa sổ 1: Video gốc kèm bounding box nhận diện TV5 và viền đa giác ROI
   - Cửa sổ 2: Ảnh nắn góc nhìn Bird's-Eye View (BEV) TV2
   - Cửa sổ 3: Mặt nạ nhị phân Occupancy Mask TV3
   - Cửa sổ 4: Bản đồ nhiệt vận tốc dòng xe Optical Flow Heatmap TV4
3. Kiểm tra tính hợp lệ của Bảng điều khiển trung tâm (TCI, nhãn mức độ, FPS, tải trọng PCU).
4. Kiểm tra hàm tạo Canvas 2x2 tích hợp và xuất ảnh nghiệm thu (Snapshot).
"""

import os
import sys
import time
import numpy as np
import cv2

# Cấu hình in tiếng Việt trên console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from demo.app import DashboardPipeline


def test_tv7_dashboard_pipeline():
    """Kiểm thử tích hợp pipeline 4 cửa sổ và bảng điều khiển trung tâm TV7."""
    sample_video = os.path.join("data", "raw", "traffic_congested.mp4")
    assert os.path.exists(sample_video), f"Khong tim thay video mau: {sample_video}"

    print("\n==================================================================")
    print("  KIEM THU TV7: DASHBOARD HIEN THI DEMO & KIEM THU")
    print(f"  Video kiem thu: {sample_video}")
    print("==================================================================")

    # 1. Khởi tạo Pipeline TV7
    pipeline = DashboardPipeline(
        video_source=sample_video,
        conf_thresh=0.22,
        iou_thresh=0.45,
        flow_noise_thresh=1.0,
        bev_size=(640, 480)
    )

    processed_frames = 0
    max_test_frames = 25
    last_result = None

    start_time = time.time()
    while processed_frames < max_test_frames:
        result = pipeline.process_frame()
        if result is None:
            break

        processed_frames += 1
        last_result = result

        # 2. Kiểm tra định dạng và tính toàn vẹn của 4 cửa sổ hiển thị
        assert "window1_original_bbox" in result, "Thieu Cua so 1 (TV5 BBox)"
        assert "window2_bev" in result, "Thieu Cua so 2 (TV2 BEV)"
        assert "window3_occ_mask" in result, "Thieu Cua so 3 (TV3 Occupancy Mask)"
        assert "window4_flow_heatmap" in result, "Thieu Cua so 4 (TV4 Optical Flow Heatmap)"
        assert "central_metrics" in result, "Thieu Bang dieu khien trung tam"

        w1 = result["window1_original_bbox"]
        w2 = result["window2_bev"]
        w3 = result["window3_occ_mask"]
        w4 = result["window4_flow_heatmap"]

        assert isinstance(w1, np.ndarray) and len(w1.shape) == 3, "Cua so 1 phai la anh mau 3 kenh"
        assert isinstance(w2, np.ndarray) and w2.shape[:2] == (480, 640), "Cua so 2 BEV phai dung kich thuoc (480, 640)"
        assert isinstance(w3, np.ndarray) and len(w3.shape) == 3, "Cua so 3 phai la anh 3 kenh"
        assert isinstance(w4, np.ndarray) and len(w4.shape) == 3, "Cua so 4 phai la Heatmap 3 kenh"

        # 3. Kiểm tra chỉ số Bảng điều khiển trung tâm
        m = result["central_metrics"]
        assert 0.0 <= m["instant_tci"] <= 1.0, f"Gia tri TCI tuc thoi nam ngoai khoang [0, 1]: {m['instant_tci']}"
        assert 0.0 <= m["smoothed_tci"] <= 1.0, f"Gia tri TCI lam min nam ngoai khoang [0, 1]: {m['smoothed_tci']}"
        assert m["level"] in ["1 - Thong thoang", "2 - Binh thuong", "3 - Un u", "4 - Tac nghen"], f"Nhan muc do khong hop le: {m['level']}"
        assert m["fps"] >= 0.0, "Chi so FPS phai >= 0"
        assert m["total_pcu"] >= 0.0, "Tong PCU phai >= 0"

        if processed_frames % 5 == 0 or processed_frames == 1:
            print(f"  [Frame {processed_frames:02d}/{max_test_frames}] "
                  f"TCI: {m['smoothed_tci']:.2f} | Muc do: {m['level']} | "
                  f"Chiem dung: {m['occupancy_pct']}% | PCU: {m['total_pcu']:.1f} | FPS: {m['fps']:.1f}")

    pipeline.release()
    total_time = time.time() - start_time
    avg_pipeline_fps = processed_frames / total_time if total_time > 0 else 0

    print(f"\n[OK] Da xu ly thanh cong {processed_frames} frames (Toc do trung binh: {avg_pipeline_fps:.1f} FPS)")

    # 4. Kiểm tra tạo Canvas ghép 2x2 hoàn chỉnh và lưu ảnh nghiệm thu
    w1 = cv2.resize(last_result["window1_original_bbox"], (640, 360))
    w2 = cv2.resize(last_result["window2_bev"], (640, 360))
    w3 = cv2.resize(last_result["window3_occ_mask"], (640, 360))
    w4 = cv2.resize(last_result["window4_flow_heatmap"], (640, 360))
    row1 = np.hstack([w1, w2])
    row2 = np.hstack([w3, w4])
    composite_canvas = np.vstack([row1, row2])
    assert composite_canvas.shape == (720, 1280, 3), f"Kich thuoc Canvas khong dung: {composite_canvas.shape}"

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, "dashboard_tv7_snapshot.jpg")
    cv2.imwrite(snapshot_path, composite_canvas)

    assert os.path.exists(snapshot_path), f"Loi khong the tao file snapshot: {snapshot_path}"
    print(f"[OK] Da xuat anh Canvas nghiem thu TV7 tai: {snapshot_path}")
    print("==================================================================")
    print("  KET QUA: MODULE TV7 DASHBOARD DA VUOT QUA TAT CA CAC KIEM THU!")
    print("==================================================================\n")


if __name__ == "__main__":
    test_tv7_dashboard_pipeline()
