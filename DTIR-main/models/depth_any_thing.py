import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingWrapper:
    def __init__(self, encoder, model_path, device="cuda"):
        self.device = device
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = encoder
        self.model_path = model_path
        depth_anything_model = DepthAnythingV2(**self.model_configs[self.encoder])
        depth_anything_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = depth_anything_model.to(self.device).eval()
    
    def process_image(self, image_path):
        """Process an image and update depth information for items"""
        cv2_image = cv2.imread(image_path)
        depth_map = self.model.infer_image(cv2_image)
        
        # Normalize depth map to 0-255 range
        depth_map = np.array(depth_map)
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
        normalized_depth = np.round(normalized_depth, 2)
        normalized_depth = 255 - normalized_depth
        return normalized_depth.tolist()

