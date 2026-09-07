import numpy as np

class TrafficCongestionEvaluator:
    def __init__(self, w_occ=0.4, w_spd=0.4, w_pcu=0.2, v_free=40.0, max_pcu_cap=50.0):
        assert abs((w_occ + w_spd + w_pcu) - 1.0) < 1e-4, "Tong trong so phai bang 1.0"
        self.w_occ = w_occ
        self.w_spd = w_spd
        self.w_pcu = w_pcu
        self.v_free = v_free
        self.max_pcu_cap = max_pcu_cap
        self.tci_history = []
        self.window_size = 15

    def compute_tci(self, occupancy_ratio, avg_speed, pcu_count):
        o_norm = np.clip(occupancy_ratio, 0.0, 1.0)
        v_norm = np.clip(avg_speed / self.v_free, 0.0, 1.0)
        d_pcu_norm = np.clip(pcu_count / self.max_pcu_cap, 0.0, 1.0)
        
        speed_penalty = 1.0 - v_norm
        instant_tci = (self.w_occ * o_norm) + (self.w_spd * speed_penalty) + (self.w_pcu * d_pcu_norm)
        instant_tci = float(np.clip(instant_tci, 0.0, 1.0))
        
        self.tci_history.append(instant_tci)
        if len(self.tci_history) > self.window_size:
            self.tci_history.pop(0)
            
        smoothed_tci = float(np.mean(self.tci_history))
        level_label, level_color = self._classify_level(smoothed_tci)
        
        return {
            "instant_tci": round(instant_tci, 3),
            "smoothed_tci": round(smoothed_tci, 3),
            "level": level_label,
            "color_bgr": level_color,
            "metrics": {
                "occupancy": round(o_norm, 3),
                "speed_norm": round(v_norm, 3),
                "pcu_density": round(d_pcu_norm, 3)
            }
        }

    def _classify_level(self, tci):
        if tci < 0.30:
            return "1 - Thong thoang", (0, 255, 0)
        elif tci < 0.55:
            return "2 - Binh thuong", (0, 255, 255)
        elif tci < 0.75:
            return "3 - Un u", (0, 165, 255)
        else:
            return "4 - Tac nghen", (0, 0, 255)