import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from generate_expert_data import MapGenerator as _Base


class ScenarioGenerator(_Base):
    def __init__(self, cfg_or_size=20.0, resolution=0.05):
        if hasattr(cfg_or_size, 'MAP_SIZE'):
            super().__init__(cfg_or_size.MAP_SIZE, cfg_or_size.MAP_RESOLUTION)
            self._local_px = cfg_or_size.LOCAL_MAP_PIXELS  # 80
        else:
            super().__init__(cfg_or_size, resolution)
            self._local_px = 80

    def extract_local_esdf(self, global_esdf, x, y, yaw):
        """Extract (local_px × local_px) ESDF patch centred on robot, rotated to robot frame."""
        cx = x / self.resolution + self.offset
        cy = y / self.resolution + self.offset
        half = self._local_px / 2
        rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(-yaw), 1.0)
        rot[0, 2] += half - cx
        rot[1, 2] += half - cy
        return cv2.warpAffine(
            global_esdf.astype(np.float32), rot,
            (self._local_px, self._local_px),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0,
        )
