import argparse
import copy
import json
import base64
import os
from bitarray import bitarray
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from pathlib import Path
from utils.utils import extract_object_with_mask, mask_coverage, load_image
from utils.log import log_info, log_verbose, set_log_level



class DigitalTwinsLoader:
    def decode_mask(self, mask_base64):
        mask_bytes = base64.b64decode(mask_base64)
        mask = bitarray()
        mask.frombytes(mask_bytes)
        rows = len(mask) // 8
        mask = np.array([mask[i*8:(i+1)*8].tolist() for i in range(rows)], dtype=bool)
        mask = mask.flatten()
        return mask

    def expand_mask_to_full_image(self, mask, x, y, w, h, image_height, image_width):
        full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask
        return full_mask

    def load_digital_twins(self, dt_path, dt_mask_path, load_mask=True):
        with open(dt_path, 'r') as f:
            json_data_object = json.load(f)
        with open(dt_mask_path, 'r') as f:
            json_data_mask = json.load(f)

        image_info = json_data_object['image_info']
        image_h = image_info['height']
        image_w = image_info['width']

        obj_mask_map = {}
        for mask in json_data_mask['objects_mask_info']:
            obj_mask_map[mask['id']] = mask['mask']

        digital_twins_info = []
        for obj in json_data_object['objects_info']:
            if load_mask:
                obj['mask'] = obj_mask_map[obj['id']]
                x, y, w, h = obj['box']
                x, y, w, h = int(x), int(y), int(w), int(h)
                mask_size = w * h
                mask = self.decode_mask(obj['mask'])
                mask = mask[:mask_size]
                mask = mask.reshape(h, w)
                obj['mask'] = self.expand_mask_to_full_image(mask, x, y, w, h, image_h, image_w)
            digital_twins_info.append(obj)

        return digital_twins_info, json_data_object['image_info']
    