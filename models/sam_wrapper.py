"""
sam_wrapper.py

Thin wrapper around Meta's Segment Anything Model (SAM).

We use SAM-ViT-H (the largest variant) for best quality,
but ViT-L or ViT-B can be used for faster inference.

SAM takes point prompts (foreground + background) and returns
binary segmentation masks ranked by confidence score.
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional
import torch


# Checkpoint filenames and their download URLs (for reference)
SAM_CHECKPOINTS = {
    "vit_h": "sam_vit_h_4b8939.pth",   # ~2.4 GB — best quality
    "vit_l": "sam_vit_l_0b3195.pth",   # ~1.2 GB — good balance
    "vit_b": "sam_vit_b_01ec64.pth",   # ~375 MB — fastest
}

SAM_DOWNLOAD_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


class SAMWrapper:
    """
    Wraps SAM's SamPredictor to accept our custom point prompts.

    Usage
    -----
        sam = SAMWrapper(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth")
        mask, score = sam.predict_with_points(image, fg_points, bg_points)
    """

    def __init__(
        self,
        model_type:      str = "vit_h",
        checkpoint_path: str = "sam_vit_h_4b8939.pth",
        device:          str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        from segment_anything import sam_model_registry, SamPredictor

        print(f"[SAM] Loading SAM {model_type} from {checkpoint_path} on {device} ...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device)
        self.predictor = SamPredictor(sam_model)
        self.device = device
        print("[SAM] Model loaded.")

    def predict_with_points(
        self,
        image:            Image.Image,
        fg_points:        np.ndarray,                    # (K, 2)  (x, y) pixel coords
        bg_points:        Optional[np.ndarray] = None,   # (M, 2)
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run SAM with foreground and (optionally) background point prompts.

        SAM's multimask mode produces 3 candidate masks per call, each with
        a confidence score. We pick the one with the highest score.

        Args:
            image:            PIL.Image.Image (RGB)
            fg_points:        (K, 2) array of foreground point coords (x, y)
            bg_points:        (M, 2) array of background point coords (x, y)
            multimask_output: if True, SAM returns 3 masks; we pick the best

        Returns:
            best_mask:  np.ndarray (H, W) bool  — best binary mask
            best_score: float                   — SAM confidence score
        """
        img_np = np.array(image.convert("RGB"))
        self.predictor.set_image(img_np)

        # SAM uses a single point_coords array with a parallel label array:
        # label 1 = foreground, label 0 = background
        all_points = list(fg_points)
        all_labels = [1] * len(fg_points)

        if bg_points is not None and len(bg_points) > 0:
            all_points += list(bg_points)
            all_labels += [0] * len(bg_points)

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        # masks:  (num_masks, H, W) bool
        # scores: (num_masks,)

        best_idx   = int(np.argmax(scores))
        best_mask  = masks[best_idx]
        best_score = float(scores[best_idx])

        return best_mask, best_score

    def predict_batch(
        self,
        images:  List[Image.Image],
        prompts: List[Dict],            # list of {'fg_points': ..., 'bg_points': ...}
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Run SAM on a list of images with corresponding prompts.

        Returns:
            list of (mask, score) tuples
        """
        results = []
        for img, prompt in zip(images, prompts):
            mask, score = self.predict_with_points(
                image     = img,
                fg_points = prompt["fg_points"],
                bg_points = prompt.get("bg_points", None),
            )
            results.append((mask, score))
        return results