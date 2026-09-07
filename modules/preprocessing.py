import cv2
import numpy as np

def preprocess_frame(frame, clip_limit=2.0, tile_grid_size=(8, 8), blur_ksize=(5, 5)):
    """
    Tiền xử lý ảnh: Giảm nhiễu và cân bằng sáng CLAHE (Chương 2)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(blurred)
    return enhanced

def get_perspective_bev(frame, src_pts, dst_pts, output_size=(640, 480)):
    """
    Nắn góc nhìn phối cảnh về Bird's-Eye View (Chương 2)
    """
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev_frame = cv2.warpPerspective(frame, matrix, output_size)
    return bev_frame, matrix
