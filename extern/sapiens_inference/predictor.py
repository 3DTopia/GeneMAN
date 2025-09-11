from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .depth import SapiensDepth, SapiensDepthType, draw_depth_map
from .segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map
from .normal import SapiensNormal, SapiensNormalType, draw_normal_map
# from .detector import Detector, DetectorConfig


@dataclass
class SapiensConfig:
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_1B
    normal_type: SapiensNormalType = SapiensNormalType.OFF
    depth_type: SapiensDepthType = SapiensDepthType.OFF
    # detector_config: DetectorConfig = DetectorConfig()
    minimum_person_height: int = 0.5  # 50% of the image height
    use_precomp_mask: bool = False  # whether to use pre-computed mask

    def __str__(self):
        return f"SapiensConfig(dtype={self.dtype}\n" \
               f"device={self.device}\n" \
               f"segmentation_type={self.segmentation_type}\n" \
               f"normal_type={self.normal_type}\n" \
               f"depth_type={self.depth_type}\n" \
               f"minimum_person_height={self.minimum_person_height * 100}% of the image height"


def filter_small_boxes(boxes: np.ndarray, img_height: int, height_thres: float = 0.1) -> np.ndarray:
    person_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        person_height = y2 - y1
        if person_height < height_thres * img_height:
            continue
        person_boxes.append(box)
    return np.array(person_boxes)


def expand_boxes(boxes: np.ndarray, img_shape: tuple[int, int], padding: int = 50) -> np.ndarray:
    expanded_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_shape[1], x2 + padding)
        y2 = min(img_shape[0], y2 + padding)
        expanded_boxes.append([x1, y1, x2, y2])
    return np.array(expanded_boxes)


class SapiensPredictor:
    def __init__(self, config: SapiensConfig):
        self.has_normal = config.normal_type != SapiensNormalType.OFF
        self.has_depth = config.depth_type != SapiensDepthType.OFF
        # self.minimum_person_height = config.minimum_person_height

        # create normal/depth model
        self.normal_predictor = SapiensNormal(
            config.normal_type, config.device, config.dtype
        ) if self.has_normal else None
        self.depth_predictor = SapiensDepth(
            config.depth_type, config.device, config.dtype
        ) if self.has_depth else None
        
        self.use_precomp_mask = config.use_precomp_mask
        self.segmentation_predictor = SapiensSegmentation(
            config.segmentation_type, config.device, config.dtype
        ) if not self.use_precomp_mask else None
        self.detector = None

    def __call__(self, img: np.ndarray, mask=None) -> np.ndarray:
        return self.predict(img, mask=mask)

    def predict(self, img: np.ndarray, mask=None) -> np.ndarray:
        if not self.use_precomp_mask:
            segmentation_map = self.segmentation_predictor(img)
        else:
            segmentation_map = None
        
        normal_map = self.normal_predictor(img)
        depth_map = self.depth_predictor(img)
        return self.draw_map(img, mask, segmentation_map, normal_map, depth_map)

    # TODO: check here
    def draw_map(self, img, precomp_mask, segmentation_map, normal_map, depth_map):
        # compute subejct mask
        mask = (segmentation_map > 0) if precomp_mask is None else precomp_mask
        
        draw_img = []
        bg_value = 0  # black background
        
        if segmentation_map is not None:
            segmentation_draw = draw_segmentation_map(segmentation_map)
            segmentation_draw[~mask] = bg_value
            draw_img.append(segmentation_draw)

        if self.has_normal:
            normal_draw = draw_normal_map(normal_map)
            normal_draw[~mask] = bg_value
            # draw_img.append(normal_draw)

        if self.has_depth:
            depth_map[~mask] = 0
            depth_draw = draw_depth_map(depth_map)
            depth_draw[~mask] = bg_value
            # draw_img.append(depth_draw)

        return normal_draw, depth_draw
