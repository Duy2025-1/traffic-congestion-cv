import os
import sys
import cv2
import numpy as np

from modules.preprocessing import (
    preprocess_frame,
    get_perspective_bev
)

from configs.toadovideo import VIDEO_CONFIG


print("===== DANH SACH VIDEO =====")

video_list = list(VIDEO_CONFIG.keys())

for idx, video_name in enumerate(video_list, start=1):
    print(f"{idx}. {video_name}")

video_name = video_list[0]
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    arg_val = sys.argv[1]
    if arg_val.isdigit() and 1 <= int(arg_val) <= len(video_list):
        video_name = video_list[int(arg_val) - 1]
    elif arg_val in video_list:
        video_name = arg_val
    elif os.path.basename(arg_val) in video_list:
        video_name = os.path.basename(arg_val)
elif sys.stdin.isatty():
    try:
        choice = input("\nChon video (1-3) [Mac dinh 1]: ").strip()
        if choice and choice.isdigit() and 1 <= int(choice) <= len(video_list):
            video_name = video_list[int(choice) - 1]
    except (EOFError, KeyboardInterrupt):
        pass
else:
    print(f"[INFO] Che do non-interactive: Tu dong chon video 1 ({video_name})")

print(f"\nDang mo: {video_name}")

video_path = f"data/raw/{video_name}"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Khong mo duoc video: {video_path}")
    exit()

# ==========================================
# NHAY DEN FRAME CO XE
# ==========================================
if video_name == "traffic_congested.mp4":
    cap.set(cv2.CAP_PROP_POS_MSEC, 3000)   # 3 giay
else:
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)

ret, frame = cap.read()

if not ret:
    print("Khong doc duoc frame!")
    exit()

# ==========================================
# TOA DO 4 GOC
# ==========================================
src_pts = VIDEO_CONFIG[video_name]["src_pts"]

print("\n===== 4 GOC TOA DO =====")
print(src_pts)

# ==========================================
# PREPROCESS
# ==========================================
processed = preprocess_frame(frame)

# ==========================================
# BIRD EYE VIEW
# ==========================================
bev_frame, matrix = get_perspective_bev(
    processed,
    src_pts,
    (640, 480)
)

# ==========================================
# THONG TIN
# ==========================================
print("\n===== THONG TIN =====")
print("Video:", video_name)
print("Original Shape :", frame.shape)
print("Processed Shape:", processed.shape)
print("BEV Shape      :", bev_frame.shape)

print("\n===== PERSPECTIVE MATRIX =====")
print(matrix)

# ==========================================
# LUU KET QUA VA HIEN THI
# ==========================================
os.makedirs("data/processed", exist_ok=True)
save_path = "data/processed/tv2_test_result.jpg"

# Ghép 3 ảnh so sánh trực quan
h_out, w_out = 360, 480
f_orig_resized = cv2.resize(frame, (w_out, h_out))
f_proc_resized = cv2.cvtColor(cv2.resize(processed, (w_out, h_out)), cv2.COLOR_GRAY2BGR)
f_bev_resized = cv2.cvtColor(cv2.resize(bev_frame, (w_out, h_out)), cv2.COLOR_GRAY2BGR)

cv2.putText(f_orig_resized, "1. Original", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(f_proc_resized, "2. CLAHE + Blur", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(f_bev_resized, "3. Bird Eye View", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

comparison = np.hstack([f_orig_resized, f_proc_resized, f_bev_resized])
cv2.imwrite(save_path, comparison)
print(f"\n[OK] Da luu anh so sanh TV2 vao: {save_path}")

is_interactive = sys.stdin.isatty() and "--headless" not in sys.argv
if is_interactive:
    cv2.imshow("TV2 Preprocessing Comparison", comparison)
    print("\nNhan phim bat ky de dong...")
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
print("[THANH CONG] Kiem thu TV2 hoan tat!")