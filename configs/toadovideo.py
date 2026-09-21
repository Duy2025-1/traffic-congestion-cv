import numpy as np

# ==============================================================================
# TỌA ĐỘ VÙNG ROI (GROUND-TRUTH CHÍNH XÁC 100% THEO ĐƯỜNG ĐEN NGƯỜI DÙNG VẼ)
# ==============================================================================
# Tọa độ 4 đỉnh hình thang phối cảnh [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
# được tính toán và biến đổi chuẩn xác (bằng ma trận Homography không gian thực)
# từ các nét vẽ màu đen của người dùng trên từng video gốc:
#
# 1. traffic_free_flow.mp4 (1920x1080):
#    - Cạnh trái bám sát vạch kẻ liền dải phân cách giữa.
#    - Cạnh phải bám sát vạch kẻ mép phải của làn đường chính.
#    - Cạnh trên thu hẹp ở đỉnh cao tốc xa.
#
# 2. traffic_congested.mp4 (1920x1080):
#    - Cạnh trên (y=285) thu hẹp tại điểm tụ ngã tư xa nhất của phố New York.
#    - Cạnh trái bám sát mép vỉa hè / tán cây bên trái.
#    - Cạnh phải bám sát dọc theo hàng xe đỗ bên phải đường.
#    - Cạnh đáy (y=1078) bao phủ trọn vẹn chiều rộng lòng đường gần camera.
#
# 3. traffic_traffic_light.mp4 (1918x970):
#    - Cạnh trên (y=455) nằm đúng tại vạch dừng ngã tư đèn tín hiệu xa.
#    - Cạnh trái bám sát mép dải phân cách cây xanh giữa đường Hà Nội.
#    - Cạnh phải bám sát vỉa hè bên phải của làn đường đang lưu thông.
#    - Cạnh đáy (y=969) phủ trọn vẹn toàn bộ phần đường sát camera.
# ==============================================================================

VIDEO_CONFIG = {

    # =====================================================
    # VIDEO 1 - FREE FLOW (1920x1080)
    # =====================================================
    "traffic_free_flow.mp4": {
        "src_pts": np.float32([
            [ 977,    0],   # Top-left     (vạch phân cách trái ở xa)
            [1046,    0],   # Top-right    (vạch phân cách phải ở xa)
            [1609, 1078],   # Bottom-right (chân mép phải gần camera)
            [ 514, 1078],   # Bottom-left  (chân mép trái gần camera)
        ])
    },

    # =====================================================
    # VIDEO 2 - CONGESTED (1920x1080)
    # =====================================================
    "traffic_congested.mp4": {
        "src_pts": np.float32([
            [ 940,  285],   # Top-left     (điểm xa bên trái lòng đường)
            [1071,  285],   # Top-right    (điểm xa bên phải lòng đường)
            [1869, 1078],   # Bottom-right (chân hàng xe đỗ sát mép phải dưới)
            [ 260, 1078],   # Bottom-left  (mép trái dưới cạnh xe tải trắng)
        ])
    },

    # =====================================================
    # VIDEO 3 - TRAFFIC LIGHT (1918x970)
    # =====================================================
    "traffic_traffic_light.mp4": {
        "src_pts": np.float32([
            [ 927,  455],   # Top-left     (mép dải phân cách cây xanh tại ngã tư xa)
            [ 991,  456],   # Top-right    (mép vỉa hè phải tại ngã tư xa)
            [1685,  969],   # Bottom-right (chân vỉa hè phải gần camera)
            [ 485,  969],   # Bottom-left  (chân dải phân cách trái gần camera)
        ])
    }

}