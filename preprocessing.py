"""
Human Image Preprocessing Pipeline

This script processes human images by performing:
- Background removal using rembg and optionally SAM
- Depth and normal map estimation using Sapiens
- 2D keypoint detection using MediaPipe
- Optional image captioning using BLIP2
- Image recentering and resizing
"""

import os
import sys
import cv2
import argparse
import numpy as np
import logging
import mediapipe as mp
from typing import Optional, Dict, Tuple
from tqdm import tqdm

import torch
from PIL import Image
from ultralytics import YOLO

from extern.sapiens_inference import SapiensPredictor, SapiensConfig, SapiensDepthType, SapiensNormalType
from segment_anything import SamPredictor, sam_model_registry


class BLIP2:
    """BLIP2 model for image captioning."""
    
    def __init__(self, device='cuda'):
        """Initialize BLIP2 model.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            revision="51572668da0eb669e01a189dc22abe6088589a24",
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16, 
            revision="51572668da0eb669e01a189dc22abe6088589a24",
        ).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(
            self.device, torch.float16
        )
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text


def get_keypoints(image: np.ndarray) -> Dict[str, torch.Tensor]:
    """Extract 2D keypoints from image using MediaPipe.
    
    Args:
        image: RGB image array
        
    Returns:
        Dictionary containing body and face keypoints
    """
    def collect_xyv(x, body=True):
        lmk = x.landmark
        all_lmks = []
        for i in range(len(lmk)):
            visibility = lmk[i].visibility if body else 1.0
            all_lmks.append(torch.Tensor([lmk[i].x, lmk[i].y, lmk[i].z, visibility]))
        return torch.stack(all_lmks).view(-1, 4)

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
    ) as holistic:
        results = holistic.process(image)

    fake_kps = torch.zeros(33, 4)

    result = {}
    result["body"] = collect_xyv(results.pose_landmarks) if results.pose_landmarks else fake_kps
    result["face"] = collect_xyv(
        results.face_landmarks, False
    ) if results.face_landmarks else fake_kps

    return result


def detect_human_bbox(image: np.ndarray, yolo_model) -> Optional[np.ndarray]:
    """Detect human bounding box using YOLO11.
    
    Args:
        image: RGB image array
        yolo_model: YOLO model instance
        
    Returns:
        Bounding box as [x_min, y_min, x_max, y_max] or None if no person detected
    """
    results = yolo_model.predict(
        image,
        save=False,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        classes=[0],  # only person class
        verbose=False
    )
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get the largest bounding box (highest confidence or largest area)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 1:
            # Select box with largest area
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            bbox = boxes[largest_idx]
        else:
            bbox = boxes[0]
        return bbox.astype(int)
    return None


def preprocess_single_image(img_path: str, args: argparse.Namespace, 
                           yolo_model,
                           sam_predictor,
                           sapiens_predictor: Optional[SapiensPredictor] = None,
                           blip2: Optional[BLIP2] = None) -> None:
    """Process a single image through the preprocessing pipeline.
    
    Args:
        img_path: Path to input image
        args: Command line arguments
        sam_predictor: SAM predictor instance (optional)
        sapiens_predictor: Sapiens predictor instance
        blip2: BLIP2 captioning model (optional)
    """
    if not os.path.exists(img_path):
        logging.error(f"Image file not found: {img_path}")
        return
        
    if args.output_path:
        out_dir = args.output_path
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.dirname(img_path)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    out_fg = os.path.join(out_dir, f"{base_name}_fg.png")
    out_mask = os.path.join(out_dir, f"{base_name}_mask.npy")
    out_depth = os.path.join(out_dir, f"{base_name}_depth.png")
    out_normal = os.path.join(out_dir, f"{base_name}_normal.png")
    out_landmarks = os.path.join(out_dir, f"{base_name}_landmarks.npy")
    out_caption = os.path.join(out_dir, f"{base_name}_caption.txt")

    # Load image
    logging.info(f'Loading image {img_path}...')
    
    try:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            logging.error(f"Failed to load image: {img_path}")
            return
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            # image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    except Exception as e:
        logging.error(f"Error loading image {img_path}: {e}")
        return
    
    # Detect human bounding box using YOLO11
    logging.info("Detecting human bounding box with YOLO11...")
    bbox = detect_human_bbox(img_path, yolo_model)
    if bbox is None:
        logging.error(f"No human detected in image: {img_path}")
        return
    
    # Background removal using SAM with YOLO11 bbox
    logging.info("Removing background with SAM...")
    sam_predictor.set_image(image)
    masks_bbox, scores_bbox, logits_bbox = sam_predictor.predict(
        box=bbox,
        multimask_output=False
    )
    mask = masks_bbox.squeeze()  # [H, W]
    
    # Apply mask to image
    masked_image = image.copy()
    masked_image[~mask] = 255

    # Predict normal and depth using sapiens
    logging.info('Estimating depth and normal maps using Sapiens...')
    normal_map, depth_map = sapiens_predictor(image, mask=mask)
    
    # Image captioning
    if args.enable_captioning and blip2 is not None:
        logging.info('Generating caption...')
        caption = blip2(masked_image)
        with open(out_caption, 'w') as f:
            f.write(caption)
    
    # Recenter the image if requested
    if args.recenter:
        logging.info('Recentering image...')
        image_fg = np.ones((args.size, args.size, 3), dtype=np.uint8) * 255
        image_normal = np.zeros((args.size, args.size, 3), dtype=np.uint8)
        image_depth = np.zeros((args.size, args.size), dtype=np.uint8)
        
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(args.size * (1 - args.border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (args.size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (args.size - w2) // 2
        y2_max = y2_min + w2
        image_fg[x2_min:x2_max, y2_min:y2_max] = cv2.resize(masked_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        image_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal_map[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        image_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth_map[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

        # Recenter the mask as well
        fg_mask = np.zeros((args.size, args.size), dtype=np.bool_)
        alpha_threshold = 0.5
        resized_mask = cv2.resize(mask[x_min:x_max, y_min:y_max].astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
        fg_mask[x2_min:x2_max, y2_min:y2_max] = (resized_mask > alpha_threshold)
        
    else:
        image_fg = masked_image
        fg_mask = mask
    
    # Save outputs
    logging.info('Saving outputs...')
    np.save(out_mask, fg_mask)
    cv2.imwrite(out_fg, cv2.cvtColor(image_fg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_normal, image_normal if args.recenter else normal_map)
    cv2.imwrite(out_depth, image_depth if args.recenter else depth_map)
    
    # Extract 2D keypoints using MediaPipe
    logging.info('Extracting 2D keypoints...')
    landmarks = get_keypoints(image_fg)['body']
    
    # Coordinate transformation
    landmarks[:, :2] = 2 * landmarks[:, :2] - 1
    landmarks[:, 1] = -landmarks[:, 1]
    np.save(out_landmarks, landmarks)
    
    logging.info(f'Processing completed for {img_path}')


def load_sam_predictor(sam_checkpoint: str = "pretrained_models/sam/sam_vit_h_4b8939.pth",
                      model_type: str = "vit_h",
                      device: str = "cuda"):
    """Load SAM predictor.
    
    Args:
        sam_checkpoint: Path to SAM checkpoint
        model_type: SAM model type
        device: Device to run on
        
    Returns:
        SAM predictor instance or None if failed
    """
    if not os.path.exists(sam_checkpoint):
        logging.error(f"SAM checkpoint not found at {sam_checkpoint}")
        return None
        
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
        return SamPredictor(sam)
    except Exception as e:
        logging.error(f"Failed to load SAM: {e}")
        return None


def load_sapiens_predictor(normal_model: str='2B', depth_model: str='2B') -> SapiensPredictor:
    """Load Sapiens predictor.
    
    Args:
        normal_model: Sapiens normal model type
        depth_model: Sapiens depth model type
    Returns:
        Sapiens predictor instance
    """
    config = SapiensConfig()
    config.normal_type = eval(f"SapiensNormalType.NORMAL_{normal_model.upper()}")
    config.depth_type = eval(f"SapiensDepthType.DEPTH_{depth_model.upper()}")
    config.use_precomp_mask = True  # no need to predict mask (not accurate anyway)
    return SapiensPredictor(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Human Image Preprocessing Pipeline")
    parser.add_argument('path', type=str, help="Path to image file or directory")
    parser.add_argument('--output_path', type=str, help="Output directory path")
    parser.add_argument('--size', default=1024, type=int, help="Output resolution")
    parser.add_argument('--border_ratio', default=0.1, type=float, help="Output border ratio")
    parser.add_argument('--recenter', action='store_true', help="Recenter image")
    parser.add_argument('--enable_captioning', action='store_true', help="Generate text captions")
    parser.add_argument('--sapiens_normal_model', default='2b', type=str, help="Sapiens normal model type")
    parser.add_argument('--sapiens_depth_model', default='2b', type=str, help="Sapiens depth model type")
    parser.add_argument('--sam_checkpoint', default='pretrained_models/seg/sam_vit_h_4b8939.pth', 
                       type=str, help="Path to SAM checkpoint")
    parser.add_argument('--sam_model_type', default='vit_h', type=str, help="SAM model type")
    parser.add_argument('--yolo_model', default='pretrained_models/seg/yolo11x.pt', type=str, help="YOLO model name or path")
    parser.add_argument('--device', default='cuda', type=str, help="Device to run models on")
    
    args = parser.parse_args()
    
    # Initialize YOLO11, SAM, and Sapiens models
    logging.info("Loading YOLO11 model...")
    yolo_model = YOLO(args.yolo_model)
    
    logging.info("Loading SAM model...")
    sam_predictor = load_sam_predictor(
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.device
    )
    if sam_predictor is None:
        logging.error("Failed to load SAM model")
        sys.exit(1)

    logging.info("Loading Sapiens model...")
    sapiens_predictor = load_sapiens_predictor(args.sapiens_normal_model, args.sapiens_depth_model)

    blip2 = None
    if args.enable_captioning:
        logging.info("Loading BLIP2 model...")
        blip2 = BLIP2(device=args.device)
    
    # Process images
    if os.path.isdir(args.path):  # Process all images in directory
        logging.info(f"Processing images in directory: {args.path}")
        img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        img_list = []
        
        for root, dirs, files in os.walk(args.path):
            for fname in files:
                if any(fname.lower().endswith(ext) for ext in img_extensions):
                    # Skip already processed files
                    if not any(fname.endswith(suffix) for suffix in ['_fg.png', '_depth.png', '_normal.png']):
                        img_list.append(os.path.join(root, fname))
        
        img_list.sort()
        logging.info(f"Found {len(img_list)} images to process")
        
        for img_path in tqdm(img_list, desc="Processing images"):
            try:
                preprocess_single_image(img_path, args, yolo_model, sam_predictor, sapiens_predictor, blip2)
            except Exception as e:
                logging.error(f"Failed to process {img_path}: {e}")
                continue
    
    else:  # Process single image
        if not os.path.exists(args.path):
            logging.error(f"Image file not found: {args.path}")
            sys.exit(1)
            
        try:
            preprocess_single_image(args.path, args, yolo_model, sam_predictor, sapiens_predictor, blip2)
        except Exception as e:
            logging.error(f"Failed to process {args.path}: {e}")
            sys.exit(1)
    
    logging.info("Processing completed successfully!")