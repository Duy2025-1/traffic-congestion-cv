import cv2

from modules.preprocessing import (
    preprocess_frame,
    get_perspective_bev
)

from configs.toadovideo import VIDEO_CONFIG


print("===== DANH SACH VIDEO =====")

video_list = list(VIDEO_CONFIG.keys())

for idx, video_name in enumerate(video_list, start=1):
    print(f"{idx}. {video_name}")

choice = input("\nChon video (1-3): ")

try:
    video_name = video_list[int(choice) - 1]
except (ValueError, IndexError):
    print("Lua chon khong hop le!")
    exit()

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
# HIEN THI
# ==========================================
cv2.imshow("Original", frame)
cv2.imshow("Processed", processed)
cv2.imshow("Bird Eye View", bev_frame)

print("\nNhan phim bat ky de dong...")

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()