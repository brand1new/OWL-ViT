import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor
import torch
import os
import shutil
from .log import log_info, log_verbose



def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    return image_pil, image_np


def extract_images_from_boxes(image, items_info):
    extracted_images = []
    for item in items_info:
        box = item['box']
        x, y, w, h = box
        cropped_image = image.crop([x, y, x+w, y+h])
        extracted_images.append(cropped_image)
    return extracted_images


def draw_boxes_on_image(input_image, boxes_list, color="red"):
    box_images_list = []
    for box in boxes_list:
        image = input_image.copy()
        draw = ImageDraw.Draw(image)

        width, height = image.size
        image_diagonal = (width**2 + height**2)**0.5
        line_thickness = max(1, int(image_diagonal * 1 / 500))
        
        if isinstance(color, str):
            try:
                color = ImageColor.getrgb(color)
            except:
                color = (255, 0, 0)
        
        x, y, w, h = box
        if x > 5:
            x = x - 5
        if y > 5:
            y = y - 5
        if x + w < width - 5:
            w = w + 5
        if y + h < height - 5:
            h = h + 5
        draw.rectangle([x, y, x+w, y+h], outline=color, width=line_thickness)
        box_images_list.append(image)

    return box_images_list


def images_to_numpy(image):
    return np.asarray(image)


def extract_object_with_mask(image_np, items_info):
    extracted_objects = []
    for item in items_info:
        mask = item["mask"]
        mask_x, mask_y, mask_w, mask_h = map(int, item["box"])
        cropped_mask = mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]
        cropped_image = image_np[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]
        object_image = np.full([mask_h, mask_w, 3], 255, dtype=np.uint8)
        object_image[cropped_mask!=0] = cropped_image[cropped_mask!=0]
        extracted_objects.append(Image.fromarray(object_image))
    
    return extracted_objects


def make_json_serializable(item):
    if isinstance(item, dict):
        return {k: make_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [make_json_serializable(elem) for elem in item]
    elif isinstance(item, (torch.Tensor, np.ndarray)):
        return item.tolist()
    elif isinstance(item, torch.Tensor):
        return item.item() if item.numel() == 1 else item.tolist()
    elif hasattr(item, 'item'):
        return item.item()
    else:
        return item


def print_cuda_memory_summary(device=None):
    if device is None:
        device = torch.cuda.current_device()
    
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - allocated_memory

    col_width = 20

    log_info("============================")
    log_info(f"{'Total memory:':<{col_width}} {total_memory / 1024**2:.2f} MB")
    log_info(f"{'Allocated memory:':<{col_width}} {allocated_memory / 1024**2:.2f} MB")
    log_info(f"{'Reserved memory:':<{col_width}} {reserved_memory / 1024**2:.2f} MB")
    log_info(f"{'Free memory:':<{col_width}} {free_memory / 1024**2:.2f} MB")
    log_info("============================")


def convert_to_int(s):
    if s.isdigit():
        return True, int(s)
    else:
        return False, 0
    

def mask_coverage(mask_a, mask_b):
    overlap = np.logical_and(mask_a, mask_b)
    sum_overlap = np.sum(overlap)
    sum_mask_b = np.sum(mask_b)
    if sum_mask_b == 0:
        return 0
    coverage = sum_overlap / sum_mask_b
    return coverage


def list_convert_to_int(element):
    if isinstance(element, list):
        return [list_convert_to_int(sub_element) for sub_element in element]
    else:
        return int(element)


def mask_box_iou(box, mask):
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    sub_item_mask = mask[y1:y2, x1:x2]
    sub_item_area = np.sum(sub_item_mask)
    return sub_item_area / box_area


def get_bounding_box_of_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return (cmin, rmin, cmax, rmax)


def box_over_convert_percent(x1, y1, x2, y2, image_width, image_height):
    rect_width = abs(x2 - x1)
    rect_height = abs(y2 - y1)
    rect_area = rect_width * rect_height
    image_total_area = image_width * image_height
    return rect_area / image_total_area


def create_directory(dir_path, delete_if_exist = False):
    if delete_if_exist and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def str_convert_to_int_array(s):
    cleaned = s.replace("'", "").replace('"', '').strip()
    if not cleaned:
        return []

    parts = cleaned.split(',')

    result = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            return []
        if not re.fullmatch(r'^[-+]?\d+$', stripped):
            return []
        result.append(int(stripped))

    return result