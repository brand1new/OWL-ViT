import argparse
import os
import sys
import json
from utils.log import log_info, log_verbose, set_log_level
from image_to_digital_pipeline import ImageToDigitalTwinsPipeline
from check_dt_valid import handle_check_dt_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--query_info', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_digital_twins_dir', type=str)
    parser.add_argument('--owlvit_checkpoint_path', type=str)
    parser.add_argument('--depth_anything_checkpoint_path', type=str)
    parser.add_argument('--sam_config_path', type=str)
    parser.add_argument('--sam_checkpoint_path', type=str)
    args = parser.parse_args()

    if args.image_dir is None or args.output_digital_twins_dir is None or args.query_info is None:
        log_info(f"Parameter error: image_dir, output_digital_twins_dir and query_info must be set")
        sys.exit(0)

    set_log_level(True)
    os.makedirs(args.output_digital_twins_dir, exist_ok=True)

    image_path_list = []
    with open(args.query_info, 'r') as f:
        query_info = json.load(f)
    for query_info_item in query_info:
        image_path = os.path.join(args.image_dir, query_info_item["image"])
        image_path_list.append(image_path)
    
    if len(image_path_list) <= 0:
        log_info(f"Parameter error: image_path_list is empty")
        sys.exit(0)

    log_info(f"Image list len: {len(image_path_list)}, {image_path_list}")

    # Initialize pipeline
    pipeline = ImageToDigitalTwinsPipeline(
        owlvit_checkpoint_path=args.owlvit_checkpoint_path,
        depth_anything_checkpoint_path=args.depth_anything_checkpoint_path,
        sam_config_path=args.sam_config_path,
        sam_checkpoint_path=args.sam_checkpoint_path
    )

    # Set maximum retry attempts to avoid infinite loops
    MAX_RETRY_ATTEMPTS = 3
    retry_count = 0

    while retry_count < MAX_RETRY_ATTEMPTS:
        # Generate digital twins for current image list
        log_info(f"Attempt {retry_count + 1}/{MAX_RETRY_ATTEMPTS} - Generating digital twins for {len(image_path_list)} images")
        pipeline.image_to_digital_twins(image_path_list, args.output_digital_twins_dir)

        # Check validity of generated digital twins
        log_info(f"Checking validity of generated digital twins...")
        invalid_images = handle_check_dt_valid(args.query_info, args.output_digital_twins_dir)

        # If no invalid images found, we're done
        if not invalid_images:
            log_info("All digital twins are valid. Generation complete.")
            break

        # Prepare for next iteration
        retry_count += 1
        if retry_count < MAX_RETRY_ATTEMPTS:
            # Update image_path_list to only include invalid images
            image_path_list = [os.path.join(args.image_dir, img) for img in invalid_images]
            log_info(f"Retrying generation for {len(image_path_list)} invalid images...")
        else:
            log_info(f"Maximum retry attempts ({MAX_RETRY_ATTEMPTS}) reached. Some digital twins may still be invalid.")
            log_info(f"Invalid images remaining: {invalid_images}")

    