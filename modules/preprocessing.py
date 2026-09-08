import cv2
import numpy as np


def preprocess_frame(
    frame,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    blur_ksize=(5, 5)
):
    """
    Tiền xử lý frame:
    Grayscale → Gaussian Blur → CLAHE.
    """

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Khử nhiễu
    blurred = cv2.GaussianBlur(
        gray,
        blur_ksize,
        0
    )

    # Cân bằng sáng bằng CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced = clahe.apply(blurred)

    return enhanced


def get_perspective_bev(
    frame,
    src_pts,
    output_size=(640, 480)
):
    """
    Chuyển vùng đường từ góc nhìn camera
    sang Bird's-Eye View.
    """

    w, h = output_size

    # 4 điểm đích tạo thành hình chữ nhật
    dst_pts = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    # Tính ma trận biến đổi phối cảnh
    matrix = cv2.getPerspectiveTransform(
        np.float32(src_pts),
        dst_pts
    )

    # Biến đổi sang BEV
    bev_frame = cv2.warpPerspective(
        frame,
        matrix,
        output_size
    )

    return bev_frame, matrix