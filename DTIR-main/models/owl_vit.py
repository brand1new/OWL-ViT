import sys
import torch

import jax
import functools
import numpy as np
import skimage
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models
from utils.log import log_info


BOX_OVERLAP_IOU_THRESHOLD = 0.80

class OWLVitWrapper:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self.model, self.variables, self.config = self.load_model(checkpoint_path)

        self.image_embedder = jax.jit(
            functools.partial(self.model.apply, self.variables, train=False, method=self.model.image_embedder))
        self.objectness_predictor = jax.jit(
            functools.partial(self.model.apply, self.variables, method=self.model.objectness_predictor))
        self.box_predictor = jax.jit(
            functools.partial(self.model.apply, self.variables, method=self.model.box_predictor))


    def load_model(self, checkpoint_path):
        config = configs.owl_v2_clip_l14.get_config(init_mode='canonical_checkpoint')
        config.init_from.checkpoint_path = checkpoint_path

        model = models.TextZeroShotDetectionModule(
            body_configs=config.model.body,
            objectness_head_configs=config.model.objectness_head,
            normalize=config.model.normalize,
            box_bias=config.model.box_bias)
        variables = model.load_variables(config.init_from.checkpoint_path)

        if jax.default_backend() == 'gpu':
            log_info(f"Load OWL Vit to GPU, JAX device: {jax.devices()}")
            variables = jax.device_put(variables, jax.devices('gpu')[0])

        return model, variables, config


    def object_detection(self, image_np_list, top_k=0, verify_area_threshold=0.0):
        batch_processed_image = self.pad_and_resize_image(image_np_list, self.config.dataset_configs.input_size)
        batch_processed_image = jax.device_put(batch_processed_image, jax.devices('gpu')[0])

        batch_feature_map = self.image_embedder(batch_processed_image)
        b, h, w, d = batch_feature_map.shape
        batch_image_features = batch_feature_map.reshape(b, h * w, d)

        batch_image_features = jax.device_put(batch_image_features, jax.devices('gpu')[0])
        batch_objectnesses = self.objectness_predictor(batch_image_features)['objectness_logits']
        batch_feature_map = jax.device_put(batch_feature_map, jax.devices('gpu')[0])
        batch_raw_boxes = self.box_predictor(image_features=batch_image_features, feature_map=batch_feature_map)['pred_boxes']

        boxes_for_all_images = []
        for image, objectnesses, raw_boxes in zip(image_np_list, batch_objectnesses, batch_raw_boxes):
            objectnesses = np.array(objectnesses)
            if top_k > 0:
                candidates = min(top_k * 3, len(objectnesses))
                indices = np.argpartition(objectnesses, -candidates)[-candidates:]
                indices = indices[np.argsort(-objectnesses[indices])]
                
                candidate_boxes = [raw_boxes[i] for i in indices]
                # candidate_scores = [objectnesses[i] for i in indices]
            else:
                indices = np.argsort(-objectnesses)
                candidate_boxes = [raw_boxes[i] for i in indices]
                # candidate_scores = [objectnesses[i] for i in indices]
            
            candidate_boxes = self.rescale_detection_box(candidate_boxes, image)
            final_result = []
            if len(candidate_boxes) > 0:
                boxes_np = np.array(candidate_boxes)

                for i in range(len(boxes_np)):
                    if top_k > 0 and len(final_result) >= top_k:
                        break
                    
                    box = boxes_np[i]
                    has_overlap = False
                    for selected_box in final_result:
                        iou = self.calculate_iou(box, np.array(selected_box))
                        if iou > BOX_OVERLAP_IOU_THRESHOLD:
                            has_overlap = True
                            break
                    
                    if not has_overlap:
                        final_result.append(box.tolist())

            boxes_for_all_images.append(final_result)

        return boxes_for_all_images
        
        
    def calculate_iou(self, box1, box2):
        # Convert xywh to xyxy for easier intersection calculation
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = x1_1 + box1[2], y1_1 + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = x1_2 + box2[2], y1_2 + box2[3]
        
        # Calculate intersection coordinates
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # Early return if no intersection
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        # Calculate areas directly from width and height
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        # Calculate intersection area
        intersect_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        union_area = box1_area + box2_area - intersect_area
        
        # Avoid division by zero
        if union_area <= 0:
            return 0.0
        
        return intersect_area / union_area


    def pad_and_resize_image(self, image_list, model_input_size):
        processed_image_list = []
        for image in image_list:
            h, w, d = image.shape
            size = max(h, w)
            image_padded = np.pad(image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)
            image_resized = skimage.transform.resize(
                image_padded,
                (model_input_size, model_input_size),
                anti_aliasing=True,
            )
            processed_image_list.append(image_resized)
        return np.array(processed_image_list, dtype=np.float32)


    def rescale_detection_box(self, boxes, image):
        h_img, w_img, _ = image.shape
        size = max(h_img, w_img)

        recovered_boxes = []
        for box in boxes:
            box = [array.tolist() for array in box]

            cx, cy, w, h = box
            cx = cx * size
            cy = cy * size
            w = w * size
            h = h * size

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if y1 < 0:
                y1 = 0
            if y2 < 0:
                y2 = 0
            if x1 > w_img:
                x1 = w_img
            if x2 > w_img:
                x2 = w_img
            if y1 > h_img:
                y1 = h_img
            if y2 > h_img:
                y2 = h_img
            recovered_boxes.append([x1, y1, x2 - x1, y2 - y1])
        return recovered_boxes