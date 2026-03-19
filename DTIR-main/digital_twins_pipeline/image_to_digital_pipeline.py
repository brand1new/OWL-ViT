import multiprocessing
import os
import time
import numpy as np
import base64
from bitarray import bitarray
from pathlib import Path
import torch
import json
import re
import concurrent
from utils.log import log_info, log_verbose, set_log_level
from utils.utils import (
    draw_boxes_on_image,
    load_image,
    mask_coverage,
    get_bounding_box_of_mask
)
from models.qwenvl import QWenVLWrapper
from models.owl_vit import OWLVitWrapper
from models.sam import SAMWrapper
from models.depth_any_thing import DepthAnythingWrapper
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from models.internvl import InternVLWrapper
from PIL import Image
import torch.nn.functional as F


# Data structures for intermediate artifacts
@dataclass
class BoundingBox:
    """Information about a single object's bounding box"""
    id: int
    box: List[int]  # [x, y, width, height]
    area: int


@dataclass
class DepthMap:
    """Depth map information"""
    shape: List[int]
    dtype: str
    depth_array: List[List[float]]


@dataclass
class Mask:
    """Information about a single object's mask"""
    id: int
    original_box: List[int]
    mask_box: List[int]
    mask: List[List[bool]]
    mask_shape: List[int]
    mask_encoded: str = ""  # Base64 encoded mask, set during saving
    area: int = 0


@dataclass
class SemanticInfo:
    """Semantic information for a single object"""
    id: int
    box: List[int]
    description: str



LLM_TEMPERATURE = 0.0
MAX_OWLVIT_BATCH_NUM = 4
MAX_OWLVIT_OBJECT_NUM = 20
OWLVIT_OBJECT_BOX_AREA_THRESHOLD = 0.0
MAX_SAM_BATCH_SIZE = 20
MAX_QWENVL_BATCH_NUM = 2
GLOBAL_SEMANTIC_MAX_NEW_TOKENS = 4096
CAPTION_PROMPT = \
'''
<image>
Provide a one-sentence caption for the provided image.
'''
SEMANTIC_OBJECT_PROMPT = \
'''
Based on the image and the highlighted red bounding box, describe the object inside the box in a single, natural sentence.
Your description should include:
-The object's category and color
-Its current state or action
-Its spatial relationship or interaction with nearby objects or the surrounding environment

Use the following output format:
[Thought]: <your reasoning about what the object is, its state, and its context>  
[Semantic]: [color] [object category] that is/are [action or state], [spatial relation or interaction with nearby objects/environment].  

Example:
[Thought]: The red box highlights a woman wearing a black top and dark jeans, mid-motion as she releases a bowling ball down the lane, indicating she is actively participating in a bowling game.
[Semantic]: A woman in a black top and dark jeans is releasing a red bowling ball, leaning forward in a bowling posture while standing near the lane, interacting with the bowling alley environment as part of the gameplay.
'''
SEMANTIC_GLOBAL_PROMPT = \
'''
Based on the entire image, describe the scene in detail, breaking it down into up to three layers: foreground, midground, and background. If there is no midground, describe only the foreground and background.
For each group in any layer, provide the following information:
- General Features:
  - Appearance: Describe the color, shape, and number of the objects, animals, or people.
  - Internal Actions: Describe any actions or behaviors happening within the group. For example:
    - "A couple is sitting on a park bench, holding hands and chatting."
    - "Two dogs are playfully chasing each other, while a third dog watches from the side."
  - Interaction with Other Groups: Describe how this group interacts with or relates to other groups in the scene. For example:
    - "The children are running near a group of adults sitting on the grass, occasionally looking at them."
  - Human Characteristics (if applicable):
    - Gender: Specify the gender of the person.
    - Approximate Age: Provide the person's approximate age based on their facial features and clothing style (e.g., child, young adult, middle-aged, elderly).
    - Physical Features: Describe their appearance, including clothing, hairstyle, body shape, and notable accessories.
  - Quantity:
    - For fewer than 10 items, specify the exact number (e.g., "3 dogs").
    - For more than 10 items, use "multiple" (e.g., "multiple trees").
- Position: Indicate where the group is located in the image (e.g., "foreground center", "midground right lower corner", "background right").
- Background Details:
  - Environment: Describe the environment or scene, specifying whether it's indoor or outdoor.
  - If the scene is outdoors, include:
    - Whether it is day or night.
    - Weather conditions (e.g., sunny, cloudy, raining).
- Avoid any descriptions of atmosphere, such as "the overall mood is cheerful."

Output Format:
[Thought]:
Foreground, <group1's position> <group1's description>, <group2's position> <group2's description>, ...
In the midground(if applicable), <group1's position> <group1's description>, <group2's position> <group2's description>, ...
In the background, <group1's position> <group1's description>, <group2's position> <group2's description>, ....
Environment: <detailed environment information>
[Semantic]:
<Provide a detailed elaboration of every element outlined in the [Thought] section, expanding on each group’s appearance, actions, interactions, human characteristics (if any), quantity, position, and environmental context.>
'''

class ImageToDigitalTwinsPipeline:
    def __init__(self, device="cuda", owlvit_checkpoint_path=None, depth_anything_checkpoint_path=None, sam_config_path=None, sam_checkpoint_path=None):
        self.device = device
        self.owlvit_checkpoint_path = owlvit_checkpoint_path
        self.depth_anything_checkpoint_path = depth_anything_checkpoint_path
        self.sam_config_path = sam_config_path
        self.sam_checkpoint_path = sam_checkpoint_path

    def image_to_digital_twins(self, img_path_list, dt_dir):
        """
        Process images to create digital twins.
        
        Args:
            img_path_list: List of image paths
            dt_dir: Directory to save digital twin files
            
        Returns:
            Dictionary with processing results
        """
        log_info(f"Starting digital twin creation process for {len(img_path_list)} images")
        
        # Create output directories
        os.makedirs(dt_dir, exist_ok=True)
        
        # Create intermediate directories - always generate intermediate files
        intermediate_dir = os.path.join(dt_dir, "intermediate")
        bbox_dir = os.path.join(intermediate_dir, "bbox")
        mask_dir = os.path.join(intermediate_dir, "mask")
        depth_dir = os.path.join(intermediate_dir, "depth")
        semantic_dir = os.path.join(intermediate_dir, "semantic")
        caption_dir = os.path.join(intermediate_dir, "caption")
        image_semantic_dir = os.path.join(intermediate_dir, "image_semantic")
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(bbox_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(semantic_dir, exist_ok=True)
        os.makedirs(caption_dir, exist_ok=True)
        os.makedirs(image_semantic_dir, exist_ok=True)
        
        ctx = multiprocessing.get_context('spawn')
        
        # Step 1: Object detection process
        object_detection_process= ctx.Process(
            target=self._run_object_detection,
            args=(img_path_list, bbox_dir, self.device)
        )
        object_detection_process.start()
        object_detection_process.join()
        log_info("Object detection completed")
        
        # Step 2: Depth computation process
        depth_computation_process = ctx.Process(
            target=self._run_depth_computation,
            args=(img_path_list, depth_dir, self.device)
        )
        depth_computation_process.start()
        depth_computation_process.join()
        log_info("Depth computation completed")
        
        # Step 3: Mask generation process (depends on bbox)
        mask_generation_process = ctx.Process(
            target=self._run_mask_generation,
            args=(img_path_list, bbox_dir, mask_dir, self.device)
        )
        mask_generation_process.start()
        mask_generation_process.join()
        log_info("Mask generation completed")
        
        # Step 4: Semantic information extraction process (depends on bbox)
        semantic_extraction_process = ctx.Process(
            target=self._run_semantic_extraction,
            args=(img_path_list, bbox_dir, semantic_dir, self.device)
        )
        semantic_extraction_process.start()
        semantic_extraction_process.join()
        log_info("Semantic extraction completed")
        
        # Step 5: Image caption extraction process
        image_caption_process = ctx.Process(
            target=self._run_image_caption_extraction,
            args=(img_path_list, caption_dir, self.device)
        )
        image_caption_process.start()
        image_caption_process.join()
        log_info("Image caption extraction completed")
        
        # Step 6: Whole image semantic extraction process
        image_semantic_process = ctx.Process(
            target=self._run_image_semantic_extraction_local_model,
            args=(img_path_list, image_semantic_dir, self.device)
        )
        image_semantic_process.start()
        image_semantic_process.join()
        log_info("Whole image semantic extraction completed")
        
        # Step 7: Save final digital twin files (main process)
        log_info("Saving final digital twin files")
        def process_image_save(img_path, bbox_dir, mask_dir, depth_dir, semantic_dir, caption_dir, image_semantic_dir, dt_dir):
            image_file_path = Path(img_path)
            base_filename = image_file_path.stem
            
            # Load image to get dimensions
            pil_image, _ = load_image(img_path)
            image_w, image_h = pil_image.size
            
            # Create image info
            image_info = {
                'width': image_w,
                'height': image_h
            }
            
            file_paths = {
                'bbox': os.path.join(bbox_dir, f"bbox_{base_filename}.json"),
                'mask': os.path.join(mask_dir, f"masks_{base_filename}.json"),
                'depth': os.path.join(depth_dir, f"depth_{base_filename}.json"),
                'semantic': os.path.join(semantic_dir, f"semantic_{base_filename}.json"),
                'caption': os.path.join(caption_dir, f"caption_{base_filename}.json"),
                'image_semantic': os.path.join(image_semantic_dir, f"image_semantic_{base_filename}.json"),
            }
            
            boxes, raw_boxes = (None, []) if not os.path.exists(file_paths['bbox']) else self._load_boxes_from_file(file_paths['bbox'])
            masks = [] if not os.path.exists(file_paths['mask']) else self._load_masks_from_file(file_paths['mask'])
            depth_map = None if not os.path.exists(file_paths['depth']) else self._load_depth_from_file(file_paths['depth'])
            semantics = [] if not os.path.exists(file_paths['semantic']) else self._load_semantics_from_file(file_paths['semantic'])
            image_caption = "" if not os.path.exists(file_paths['caption']) else self._load_image_caption_from_file(file_paths['caption'])
            image_semantic = "" if not os.path.exists(file_paths['image_semantic']) else self._load_image_semantics_from_file(file_paths['image_semantic'])
            
            # Add image semantic and caption to image_info
            image_info['semantic'] = image_semantic
            image_info['caption'] = image_caption
            
            semantic_map = {s.id: s for s in semantics}
            mask_map = {m.id: m for m in masks}
            
            depth_tensor = None
            if depth_map is not None:
                depth_tensor = torch.tensor(depth_map.depth_array)
            
            objects_info = []
            for box_idx, box in enumerate(boxes):
                # Initialize object info with box data
                x, y, w, h = box.box
                obj_info = {
                    'id': int(box_idx),
                    'box': [int(x), int(y), int(w), int(h)],
                    'area': 0,
                    'description': "",
                    'depth': 0.0,
                }
                
                # Add mask information if available
                mask_info = mask_map.get(box_idx)
                if mask_info:
                    obj_info['area'] = int(mask_info.area)
                
                # Add semantic information if available
                semantic_info = semantic_map.get(box_idx)
                if semantic_info:
                    obj_info['description'] = semantic_info.description
                
                # Calculate depth if available
                if depth_tensor is not None and mask_info and mask_info.mask is not None:
                    mask_tensor = torch.tensor(mask_info.mask).bool()
                    mask_sum = mask_tensor.sum().item()
                    if mask_sum > 0:
                        obj_depth = float(
                            torch.where(
                                mask_tensor,
                                depth_tensor,
                                torch.tensor(0.0)
                            ).sum() / mask_sum
                        )
                        obj_info['depth'] = round(obj_depth, 2)
                
                objects_info.append(obj_info)
            
            objects_info = sorted(objects_info, key=lambda x: x['area'], reverse=True)
            
            dt_data = {
                'image_info': image_info,
                'objects_info': objects_info
            }
            
            output_paths = {
                'main': os.path.join(dt_dir, f"{base_filename}.json"),
                'coarse': os.path.join(dt_dir, f"{base_filename}_coarse_grained.json"),
                'mask': os.path.join(dt_dir, f"{base_filename}_mask.json")
            }
            
            with open(output_paths['main'], 'w') as f:
                json.dump(dt_data, f, indent=2)
            
            # Save coarse-grained version
            coarse_grained_items_info = self.small_object_filter(objects_info, masks)
            coarse_grained_data = {
                'image_info': image_info,
                'objects_info': coarse_grained_items_info
            }
            with open(output_paths['coarse'], 'w') as f:
                json.dump(coarse_grained_data, f, indent=2)
            
            # Save mask information
            if masks:
                masks_list = []
                for mask_info in masks:
                    # Only encode if not already encoded
                    if not mask_info.mask_encoded:
                        x, y, w, h = map(int, mask_info.original_box)
                        flat_mask = np.array(mask_info.mask)[y:y+h, x:x+w]
                        mask_bytes = bitarray(flat_mask.flatten().tolist()).tobytes()
                        mask_info.mask_encoded = base64.b64encode(mask_bytes).decode('utf-8')
                    
                    masks_list.append({
                        'id': int(mask_info.id),
                        'mask': mask_info.mask_encoded,
                    })
                
                mask_data = {
                    'image_info': image_info,
                    'objects_mask_info': masks_list
                }
                with open(output_paths['mask'], 'w') as f:
                    json.dump(mask_data, f, indent=2)
            
            return img_path
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, len(img_path_list))) as executor:
            futures = []
            for img_path in img_path_list:
                futures.append(
                    executor.submit(
                        process_image_save, 
                        img_path, 
                        bbox_dir, 
                        mask_dir, 
                        depth_dir, 
                        semantic_dir, 
                        caption_dir, 
                        image_semantic_dir,
                        dt_dir
                    )
                )
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    img_path = future.result()
                    log_info(f"Processed [{i+1}/{len(futures)}] Saved digital twin files for: {img_path}")
                except Exception as e:
                    log_info(f"Error processing image: {e}")
        
        log_info(f"Completed digital twin creation for {len(img_path_list)} images")

    def _save_boxes_to_file(self, img_path, bbox_list, save_dir, save_prefix):
        """Save bounding boxes to an intermediate file"""
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file name from image path
        img_name = Path(img_path).stem
        save_path = os.path.join(save_dir, f"{save_prefix}_{img_name}.json")
        
        # Convert to dictionary for JSON serialization
        output_dict = {
            'boxes': [
                {
                    'id': box.id,
                    'box': box.box,
                    'area': box.area
                } for box in bbox_list
            ]
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        log_verbose(f"Saved {len(bbox_list)} bounding boxes to {save_path}")
    
    def _load_boxes_from_file(self, file_path):
        """Load bounding boxes from an intermediate file
        
        Args:
            file_path: Path to the JSON file containing bounding boxes
            
        Returns:
            tuple: (list of BoundingBox objects, list of raw box)
        """
        log_verbose(f"Loading bounding boxes from {file_path}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to structured data
        boxes_list = []
        raw_boxes = []
        
        for box_data in data.get('boxes', []):
            bbox = BoundingBox(
                id=box_data['id'],
                box=box_data['box'],
                area=box_data['area']
            )
            boxes_list.append(bbox)
            raw_boxes.append(box_data['box'])
        
        log_verbose(f"Loaded {len(boxes_list)} bounding boxes")
        return boxes_list, raw_boxes

    def _save_depth_to_file(self, img_path, depth_data, save_dir, save_prefix):
        """Save depth map to an intermediate file"""
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)        
        # Create file name from image path
        img_name = Path(img_path).stem
        # Save as JSON with encoded data
        json_save_path = os.path.join(save_dir, f"{save_prefix}_{img_name}.json")
        
        # Convert depth map to base64 for storage (using float16 to save space)
        depth_bytes = np.array(depth_data.depth_array, dtype=np.float16).flatten().tobytes()
        encoded_depth = base64.b64encode(depth_bytes).decode('utf-8')
        
        # Convert to dictionary for JSON serialization
        output_dict = {
            'depth_map': {
                'shape': depth_data.shape,
                'dtype': depth_data.dtype,
                'encoded': encoded_depth
            }
        }
        
        # Save to file
        with open(json_save_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        log_verbose(f"Saved depth map to {json_save_path}")
        
    def _load_depth_from_file(self, file_path):
        """Load depth map from an intermediate file
        
        Args:
            file_path: Path to the JSON file containing depth map
            
        Returns:
            tuple: (DepthMap structure, numpy array of depth values)
        """
        log_verbose(f"Loading depth map from {file_path}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        depth_data = data.get('depth_map', {})
        
        # Extract data
        shape = depth_data.get('shape', [])
        dtype = depth_data.get('dtype', 'float16')
        encoded = depth_data.get('encoded', '')
        
        # Decode depth map from base64
        depth_bytes = base64.b64decode(encoded)
        depth_array = np.frombuffer(depth_bytes, dtype=np.dtype(dtype)).reshape(shape).tolist()
        
        # Create structured output
        depth_map = DepthMap(shape, dtype, depth_array)
        
        log_verbose(f"Loaded depth map with shape {shape}")
        return depth_map

    def _save_masks_to_file(self, img_path, masks, save_dir, save_prefix):
        """Save masks to an intermediate file"""
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file name from image path
        img_name = Path(img_path).stem
        save_path = os.path.join(save_dir, f"{save_prefix}_{img_name}.json")

        # Convert to structured data
        masks_list = []
        for mask_info in masks:
            # Convert mask to base64 for storage
            x, y, w, h = map(int, mask_info.original_box)
            flat_mask = np.array(mask_info.mask)[y:y+h, x:x+w]
            mask_bytes = bitarray(flat_mask.flatten().tolist()).tobytes()
            mask_info.mask_encoded = base64.b64encode(mask_bytes).decode('utf-8')
            masks_list.append(mask_info)
        
        # Convert to dictionary for JSON serialization
        output_dict = {
            'masks': [
                {
                    'id': mask.id,
                    'original_box': mask.original_box,
                    'mask_box': mask.mask_box,
                    'mask_encoded': mask.mask_encoded,
                    'mask_shape': mask.mask_shape,
                    'area': mask.area
                } for mask in masks_list
            ]
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        log_verbose(f"Saved {len(masks)} masks to {save_path}")

    def expand_mask_to_full_image(self, mask, x, y, w, h, image_height, image_width):
        full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask
        return full_mask

    def _load_masks_from_file(self, file_path):
        """Load masks from an intermediate file
        
        Args:
            file_path: Path to the JSON file containing masks
            
        Returns:
            list: List of Mask objects
        """
        log_verbose(f"Loading masks from {file_path}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to structured data
        masks_list = []
        
        for mask_data in data.get('masks', []):
            # Decode mask from base64
            encoded_mask = mask_data['mask_encoded']
            mask_shape = mask_data['mask_shape']
            mask_bytes = base64.b64decode(encoded_mask)
            ba = bitarray()
            ba.frombytes(mask_bytes)
            rows = len(ba) // 8
            bool_array = np.array([ba[i*8:(i+1)*8].tolist() for i in range(rows)], dtype=bool)
            bool_array = bool_array.flatten()
            
            x, y, w, h = map(int, mask_data['original_box'])
            bool_array = bool_array[:w*h].reshape(h, w)
            mask_array = self.expand_mask_to_full_image(bool_array, x, y, w, h, mask_shape[0], mask_shape[1])
            mask_array = mask_array.tolist()


            # Reshape to original dimensions
            # mask_array = bool_array[:np.prod(mask_shape)].reshape(mask_shape).tolist()
            
            mask = Mask(
                id=mask_data['id'],
                original_box=mask_data['original_box'],
                mask_box=mask_data['mask_box'],
                mask=mask_array,
                mask_encoded=encoded_mask,
                mask_shape=mask_shape,
                area=mask_data['area']
            )
            masks_list.append(mask)
        
        log_verbose(f"Loaded {len(masks_list)} masks")
        return masks_list

    def _save_semantics_to_file(self, img_path, boxes, semantics, save_dir, save_prefix):
        """Save semantic information to an intermediate file"""
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file name from image path
        img_name = Path(img_path).stem
        save_path = os.path.join(save_dir, f"{save_prefix}_{img_name}.json")
        
        # Convert to dictionary for JSON serialization
        output_dict = {
            'semantics': [
                {
                    'id': int(sem.id),
                    'box': [int(x) for x in sem.box],
                    'description': sem.description,
                } for sem in semantics
            ]
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        log_verbose(f"Saved semantic information for {len(semantics)} objects to {save_path}")
    
    def _load_semantics_from_file(self, file_path):
        """Load semantic information from an intermediate file
        
        Args:
            file_path: Path to the JSON file containing semantic information
            
        Returns:
            list: List of SemanticInfo objects
        """
        log_verbose(f"Loading semantic information from {file_path}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to structured data
        semantics_list = []
        
        for semantic_data in data.get('semantics', []):
            sem_info = SemanticInfo(
                id=int(semantic_data['id']),
                box=[int(x) for x in semantic_data['box']],
                description=semantic_data['description'],
            )
            semantics_list.append(sem_info)
        
        log_verbose(f"Loaded semantic information for {len(semantics_list)} objects")
        return semantics_list

    def small_object_filter(self, items_info, masks):
        # Early exit if no masks or items
        if not masks or not items_info:
            return items_info
            
        filtered_items_info = []
        absorbable_threshold = 0.7
        to_remove = set()
        
        # Pre-process masks by object ID for faster lookup
        masks_by_id = {}
        for m in masks:
            if m.mask is not None:
                masks_by_id[m.id] = np.array(m.mask)
        
        # Skip filter if masks not found for most objects
        if len(masks_by_id) < len(items_info) / 2:
            return items_info

        for i, obj_a in enumerate(items_info):
            if i in to_remove:
                continue
                
            filtered_items_info.append(obj_a)
            
            # Get object ID and check if mask exists
            obj_a_id = obj_a.get("id")
            if obj_a_id not in masks_by_id:
                continue
                
            mask_a = masks_by_id[obj_a_id]
            
            # Optimize inner loop to avoid unnecessary comparisons
            for j, obj_b in enumerate(items_info):
                if i < j and j not in to_remove:
                    obj_b_id = obj_b.get("id")
                    
                    # Skip if mask doesn't exist
                    if obj_b_id not in masks_by_id:
                        continue
                        
                    mask_b = masks_by_id[obj_b_id]
                    if mask_coverage(mask_a, mask_b) >= absorbable_threshold:
                        to_remove.add(j)

        return filtered_items_info

    def _run_object_detection(self, img_path_list, bbox_dir, device):
        """Run object detection in a separate process, processing images in batches"""
        log_info("Starting object detection process")
            
        # Initialize model once for all images
        owl_vit = OWLVitWrapper(
            self.owlvit_checkpoint_path,
            device
        )
        
        # Process images in batches
        for i in range(0, len(img_path_list), MAX_OWLVIT_BATCH_NUM):
            batch_img_paths = img_path_list[i:i+MAX_OWLVIT_BATCH_NUM]
            log_info(f"Running object detection on batch of {len(batch_img_paths)} images")
            
            # Load images
            batch_img = [load_image(path)[1] for path in batch_img_paths]
            
            # Run object detection on batch
            batch_item_boxes = owl_vit.object_detection(
                image_np_list=batch_img, 
                top_k=MAX_OWLVIT_OBJECT_NUM*2, 
                verify_area_threshold=OWLVIT_OBJECT_BOX_AREA_THRESHOLD
            )
            
            # Store results and save
            for j, boxes in enumerate(batch_item_boxes):
                img_path = batch_img_paths[j]

                # Sort boxes by area (largest first) and keep only the top MAX_OWLVIT_OBJECT_NUM boxes
                if boxes:
                    boxes_np = np.array(boxes)
                    areas = (boxes_np[:, 2]) * (boxes_np[:, 3])
                    if len(areas) > MAX_OWLVIT_OBJECT_NUM:
                        top_indices = np.argsort(-areas)[:MAX_OWLVIT_OBJECT_NUM]
                        boxes = boxes_np[top_indices].tolist()

                # Convert to structured data
                bbox_list = []
                for box_idx, box in enumerate(boxes):
                    x, y, w, h = box
                    bbox = BoundingBox(
                        id=box_idx,
                        box=[int(x), int(y), int(w), int(h)],
                        area=int(w * h)
                    )
                    bbox_list.append(bbox)
                
                # Save to file
                self._save_boxes_to_file(img_path, bbox_list, bbox_dir, "bbox")
            
        log_info(f"Completed object detection for {len(img_path_list)} images")
        
        # Clear memory
        del owl_vit

    def _run_depth_computation(self, img_path_list, depth_dir, device):
        """Run depth computation in a separate process, processing images one by one"""
        log_info("Starting depth computation process")
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize model once for all images
        depth_model = DepthAnythingWrapper(
            "vitl", 
            self.depth_anything_checkpoint_path,
            device
        )
        
        # Process images one by one
        for img_path in img_path_list:
            log_info(f"Computing depth for image: {img_path}")
            
            # Generate depth map for the image
            depth_map = depth_model.process_image(img_path)
            
            # Create structured output
            depth_data = DepthMap(
                shape=[len(depth_map), len(depth_map[0])],
                dtype='float16',
                depth_array=depth_map
            )
            
            # Save intermediate result
            self._save_depth_to_file(img_path, depth_data, depth_dir, "depth")
        
        # Clear memory
        del depth_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log_info(f"Completed depth computation for {len(img_path_list)} images")
        
    def _run_mask_generation(self, img_path_list, bbox_dir, mask_dir, device):
        """Run mask generation in a separate process, processing images in batches"""
        log_info("Starting mask generation process")
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize model once for all images
        sam_wrapper = SAMWrapper(
            self.sam_config_path,
            self.sam_checkpoint_path,
            device
        )
        
        # Collect valid image paths and bounding boxes for batch processing
        valid_img_indices = []
        valid_img_paths = []
        valid_boxes_list = []
        
        for idx, img_path in enumerate(img_path_list):
            img_name = Path(img_path).stem
            bbox_file = os.path.join(bbox_dir, f"bbox_{img_name}.json")
            
            if not os.path.exists(bbox_file):
                log_info(f"[Error] No bounding box file found for {img_path}, skipping")
                continue
                
            _, boxes = self._load_boxes_from_file(bbox_file)
            
            if not boxes:
                log_info(f"No boxes found for {img_path}, skipping")
                continue
                
            # Convert boxes from xywh to xyxy
            converted_boxes = []
            for box in boxes:
                x, y, w, h = box
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                converted_boxes.append([x1, y1, x2, y2])
            boxes = converted_boxes

            valid_img_indices.append(idx)
            valid_img_paths.append(img_path)
            valid_boxes_list.append(boxes)
            
        # Process valid images in batches
        valid_img_list = []
        for img_path in valid_img_paths:
            _, image_np = load_image(img_path)
            valid_img_list.append(image_np)
            
        masks_batch = []
        batch_img_paths = []
        batch_boxes_list = []
        save_batch_count = 10  # Save after accumulating 10 batches
        current_batch_count = 0
        processed_count = 0
        
        for batch_start in range(0, len(valid_img_list), MAX_SAM_BATCH_SIZE):
            batch_end = min(batch_start + MAX_SAM_BATCH_SIZE, len(valid_img_list))
            
            # Get current batch
            batch_imgs = valid_img_list[batch_start:batch_end]
            batch_boxes = valid_boxes_list[batch_start:batch_end]
            
            log_info(f"Generating masks for batch of {len(batch_imgs)} images")
            
            # Generate masks for current batch
            batch_masks = sam_wrapper.predict_batch_by_boxes(batch_imgs, batch_boxes)
            
            # Collect batch data
            masks_batch.extend(batch_masks)
            batch_img_paths.extend(valid_img_paths[batch_start:batch_end])
            batch_boxes_list.extend(batch_boxes)
            
            current_batch_count += 1
            processed_count += len(batch_imgs)
            
            # Save after every save_batch_count batches or when all images are processed
            if current_batch_count >= save_batch_count or batch_end >= len(valid_img_list):
                log_info(f"Saving masks for {len(masks_batch)} images (processed {processed_count}/{len(valid_img_paths)} images)")
                
                # Process and save masks for accumulated batches
                for idx, (img_path, masks, boxes) in enumerate(zip(batch_img_paths, masks_batch, batch_boxes_list)):
                    img_masks = []
                    for box_idx, (box, mask) in enumerate(zip(boxes, masks)):
                        mask_box = get_bounding_box_of_mask(mask)
                        x1, y1, x2, y2 = mask_box
                        mask_box = [int(x1), int(y1), int(x2-x1), int(y2-y1)]

                        org_x1, org_y1, org_x2, org_y2 = box
                        
                        mask_info = Mask(
                            id=int(box_idx),
                            original_box=[int(org_x1), int(org_y1), int(org_x2-org_x1), int(org_y2-org_y1)],
                            mask_box=mask_box,
                            mask=mask,
                            mask_encoded="",
                            mask_shape=[len(mask), len(mask[0])],
                            area=int(np.sum(mask))
                        )
                        img_masks.append(mask_info)
                    
                    # Save masks to file
                    self._save_masks_to_file(img_path, img_masks, mask_dir, "masks")
                
                # Reset the accumulated batches
                masks_batch = []
                batch_img_paths = []
                batch_boxes_list = []
                current_batch_count = 0
        
        # Clear memory
        del sam_wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log_info(f"Completed mask generation for {len(img_path_list)} images")
        
    def _run_semantic_extraction(self, img_path_list, bbox_dir, semantic_dir, device):
        """Run semantic extraction in a separate process, processing images in batches where possible"""
        log_info("Starting semantic extraction process")
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize model once for all images
        qwenvl = QWenVLWrapper("Qwen/Qwen2.5-VL-7B-Instruct", device)
        prompt = SEMANTIC_OBJECT_PROMPT
        
        # Collect valid image paths and bounding boxes
        valid_img_paths = []
        valid_boxes_list = []
        
        for img_path in img_path_list:
            # Load bounding boxes for this image
            img_name = Path(img_path).stem
            bbox_file = os.path.join(bbox_dir, f"bbox_{img_name}.json")
            
            if not os.path.exists(bbox_file):
                log_info(f"[Error] No bounding box file found for {img_path}, skipping")
                continue
                
            _, boxes = self._load_boxes_from_file(bbox_file)
            
            if not boxes:
                log_info(f"No boxes found for {img_path}, skipping")
                continue
                
            valid_img_paths.append(img_path)
            valid_boxes_list.append(boxes)
        
        # Process each valid image 
        for img_idx, (img_path, boxes) in enumerate(zip(valid_img_paths, valid_boxes_list)):
            log_info(f"Extracting semantics for {img_path} with {len(boxes)} objects")
            
            # Load image
            pil_image, _ = load_image(img_path)
            
            # Extract object images based on bounding boxes
            object_images = draw_boxes_on_image(pil_image, boxes)
            
            # Process objects in batches
            obj_semantics = []
            for i in range(0, len(object_images), MAX_QWENVL_BATCH_NUM):
                batch_images = object_images[i:i + MAX_QWENVL_BATCH_NUM]
                batch_outputs = qwenvl.generate_text_from_image(batch_images, prompt)
                
                for j, model_output in enumerate(batch_outputs):
                    if i + j >= len(boxes):  # Safety check
                        continue
                        
                    semantic = "Nothing"
                    
                    match_semantic = re.search(r'\[Semantic\]:\s*(.*)', model_output)
                    if match_semantic:
                        semantic = match_semantic.group(1)
                    else:
                        log_info(f"No match semantic format, model output text: {model_output}")
                    
                    sem_info = SemanticInfo(
                        id=i + j,
                        box=[int(x) for x in boxes[i + j]],
                        description=semantic,
                    )
                    obj_semantics.append(sem_info)
                    
                    log_verbose(f"Get semantic for object[{i + j}],Semantic: {semantic}")
            
            # Save semantics to file
            self._save_semantics_to_file(img_path, boxes, obj_semantics, semantic_dir, "semantic")
        
        # Clear memory
        del qwenvl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log_info(f"Completed semantic extraction for {len(valid_img_paths)} images")

    def _save_image_caption_to_file(self, img_path, caption_text, save_dir, save_prefix):
        """Save image caption information to an intermediate file"""
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file name from image path
        img_name = Path(img_path).stem
        save_path = os.path.join(save_dir, f"{save_prefix}_{img_name}.json")
        
        # Convert to dictionary for JSON serialization
        output_dict = {
            'image_caption': {
                'caption': caption_text
            }
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        log_verbose(f"Saved image caption information to {save_path}")
    
    def _load_image_caption_from_file(self, file_path):
        """Load image caption information from an intermediate file
        
        Args:
            file_path: Path to the JSON file containing image caption information
            
        Returns:
            str: The caption of the image
        """
        log_verbose(f"Loading image caption information from {file_path}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract caption
        caption_text = data.get('image_caption', {}).get('caption', "")
        
        log_verbose(f"Loaded image caption: {caption_text}")
        return caption_text

    def _run_image_caption_extraction(self, img_path_list, caption_dir, device):
        """Run image caption extraction in a separate process"""
        log_info("Starting image caption extraction process")

        internvl_model = InternVLWrapper("OpenGVLab/InternVL2_5-8B-MPO")
        # Process each image
        for img_path in img_path_list:
            start_time = time.time()
            log_verbose(f"Extracting caption for image: {img_path}")
            
            caption_text = internvl_model.generate_text_from_image(img_path, CAPTION_PROMPT)

            log_verbose(f"Caption for image {img_path}: {caption_text}, cost time: {time.time() - start_time} seconds")
            # Save caption to file
            self._save_image_caption_to_file(img_path, caption_text, caption_dir, "caption")
            
        internvl_model = None
        log_info(f"Completed image caption extraction for {len(img_path_list)} images")

    def _save_image_semantics_to_file(self, img_path, semantic_text, save_dir, save_prefix):
        """Save whole image semantic information to an intermediate file"""
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file name from image path
        img_name = Path(img_path).stem
        save_path = os.path.join(save_dir, f"{save_prefix}_{img_name}.json")
        
        # Convert to dictionary for JSON serialization
        output_dict = {
            'image_semantic': {
                'description': semantic_text
            }
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        log_verbose(f"Saved whole image semantic information to {save_path}")
    
    def _load_image_semantics_from_file(self, file_path):
        """Load whole image semantic information from an intermediate file
        
        Args:
            file_path: Path to the JSON file containing image semantic information
            
        Returns:
            str: The semantic description of the whole image
        """
        log_verbose(f"Loading image semantic information from {file_path}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract semantic description
        semantic_text = data.get('image_semantic', {}).get('description', "")
        
        # Convert to string if not already a string
        if not isinstance(semantic_text, str):
            semantic_text = str(semantic_text)
        
        log_verbose(f"Loaded whole image semantic description in {file_path}")
        return semantic_text

    def _run_image_semantic_extraction_local_model(self, img_path_list, image_semantic_dir, device):
        """Run whole image semantic extraction in a separate process"""
        log_info("Starting whole image semantic extraction process")

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize model once for all images
        qwenvl = QWenVLWrapper("Qwen/Qwen2.5-VL-7B-Instruct", device)
        
        for i in range(0, len(img_path_list), MAX_QWENVL_BATCH_NUM):
            batch_images_list = img_path_list[i:i + MAX_QWENVL_BATCH_NUM]
            batch_size = len(batch_images_list)
            log_str_list = [""] * batch_size
            batch_images = []
            history_list = [[] for _ in range(batch_size)]
            semantic_list = [""] * batch_size
            for idx, img_path in enumerate(batch_images_list):
                pil_image, _ = load_image(img_path)
                batch_images.append(pil_image)
                history_list[idx].append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SEMANTIC_GLOBAL_PROMPT}]})
                
            # First round of semantic extraction
            batch_outputs = qwenvl.generate_text_from_image_with_multi_messages(batch_images, qwenvl.generate_messages_from_history(history_list), max_new_tokens=GLOBAL_SEMANTIC_MAX_NEW_TOKENS, temperature=LLM_TEMPERATURE)
            for idx, (img_path, model_output) in enumerate(zip(batch_images_list, batch_outputs)):
                semantic = "Nothing"
                match_semantic = re.search(r'\[Semantic\]:\s*(.*)', model_output, flags=re.DOTALL)
                if match_semantic:
                    semantic = match_semantic.group(1)
                else:
                    log_info(f"[Error]: No match for semantic format in whole image first round, image path: {img_path}, model output text: {model_output}")
                
                semantic_list[idx] = semantic
                history_list[idx].append({"role": "assistant", "content": [{"type": "text", "text": model_output}]})
                log_str_list[idx] = f"Image semantic for image {img_path}\n\n" + "<-----First round----->\nmodel output: \n" + model_output + "\nsemantic: " + semantic

            for idx, (img_path, semantic) in enumerate(zip(batch_images_list, semantic_list)):
                self._save_image_semantics_to_file(img_path, semantic, image_semantic_dir, "image_semantic")
                log_info(f"Processed image {img_path}, semantic: {semantic}")

            for log_str in log_str_list:
                log_info(log_str)
        
        # Clear memory
        del qwenvl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log_info(f"Completed whole image semantic extraction for {len(img_path_list)} images")
