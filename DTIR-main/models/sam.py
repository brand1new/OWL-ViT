import torch
from typing import List

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor

class SAMWrapper:
    def __init__(self, sam_model_cfg, sam_checkpoint, device="cuda"):
        self.device = device
        self.sam_model_cfg = sam_model_cfg
        self.sam_checkpoint = sam_checkpoint
        if torch.cuda.is_available():
            torch.autocast(self.device, dtype=torch.bfloat16).__enter__()
        else:
            print("cuda is not available!")

        sam2 = build_sam2(self.sam_model_cfg, self.sam_checkpoint, device=self.device, apply_postprocessing=False)
        self.sam_mask_generator = SAM2AutomaticMaskGenerator(sam2)
        self.sam_mask_predictor = SAM2ImagePredictor(sam2)

    def predict_batch_by_boxes(self, image_list, boxes_for_all_img_list):
        """Run batch prediction with SAM
        
        Args:
            image_list: List of images to process
            box_list: List of bounding boxes for each image
            
        Returns:
            masks_batch: Batch of predicted masks
        """
        self.sam_mask_predictor.set_image_batch(image_list)
        masks_batch, _, _ = self.sam_mask_predictor.predict_batch(
            None,
            None,
            box_batch=boxes_for_all_img_list,
            multimask_output=False
        )
        # Convert masks_batch to a 2D list of bools
        bool_masks_batch = []
        for masks in masks_batch:
            bool_masks = []
            for mask in masks:
                if mask.ndim == 3:
                    mask = mask[0]
                bool_mask = mask.astype(bool).tolist()
                bool_masks.append(bool_mask)
            bool_masks_batch.append(bool_masks)

        return bool_masks_batch

