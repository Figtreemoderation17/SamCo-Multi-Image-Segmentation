"""
cosegmentation.py

Full SAMCo Co-segmentation Pipeline

Ties together:
  1. DINOv2FeatureExtractor    — dense patch features
  2. SemanticConsensusPrompter — cross-image consensus -> SAM prompts  (novel)
  3. SAMWrapper                — precision segmentation masks
  4. Iterative Refinement      — mask-guided prompt improvement        (novel)

Input : N images sharing a common foreground object (any number N >= 2)
Output: N binary segmentation masks isolating the common object
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import os

from models.feature_extractor   import DINOv2FeatureExtractor
from models.consensus_prompting import SemanticConsensusPrompter
from models.sam_wrapper          import SAMWrapper


class SAMCo:
    """
    SAMCo: Automatic Co-segmentation via Semantic Consensus Prompting.

    Parameters
    ----------
    sam_model_type  : SAM variant — 'vit_h' | 'vit_l' | 'vit_b'
    sam_checkpoint  : path to SAM .pth checkpoint file
    n_fg_points     : foreground prompt points per image
    n_bg_points     : background prompt points per image
    top_k_ratio     : fraction of cross-image patches used for consensus
    n_refine_iter   : iterations of mask-guided prompt refinement (0 = no refine)
    device          : 'cuda', 'cpu', or 'auto'
    """

    def __init__(
        self,
        sam_model_type: str   = "vit_h",
        sam_checkpoint: str   = "sam_vit_h_4b8939.pth",
        n_fg_points:    int   = 5,
        n_bg_points:    int   = 3,
        top_k_ratio:    float = 0.30,
        n_refine_iter:  int   = 2,
        device:         str   = "auto",
    ):
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise all three sub-modules
        self.feat_extractor = DINOv2FeatureExtractor(device=device)
        self.prompter       = SemanticConsensusPrompter(
            n_fg_points   = n_fg_points,
            n_bg_points   = n_bg_points,
            top_k_ratio   = top_k_ratio,
            n_refine_iter = n_refine_iter,
        )
        self.sam            = SAMWrapper(
            model_type      = sam_model_type,
            checkpoint_path = sam_checkpoint,
            device          = device,
        )
        self.n_refine_iter  = n_refine_iter

    def segment(
        self,
        images:  List[Image.Image],
        verbose: bool = True,
    ) -> List[np.ndarray]:
        """
        Run the full SAMCo pipeline on a list of co-segmentation images.

        Args:
            images:  list of PIL.Image.Image (RGB), N >= 2
            verbose: print progress messages

        Returns:
            masks: list of N binary np.ndarray of shape (H, W) matching each image
        """
        assert len(images) >= 2, "Co-segmentation requires at least 2 images."

        orig_sizes = [(img.height, img.width) for img in images]

        if verbose:
            print("[SAMCo] Step 1/4 — Extracting DINOv2 patch features ...")
        all_feats = self.feat_extractor.extract_batch(images)

        if verbose:
            print("[SAMCo] Step 2/4 — Computing cross-image consensus saliency ...")
        saliency_maps = self.prompter.compute_consensus_saliency(all_feats)

        if verbose:
            print("[SAMCo] Step 3/4 — Generating Semantic Consensus Prompts ...")
        prompts = self.prompter.generate_prompts(
            saliency_maps, self.feat_extractor, orig_sizes
        )

        if verbose:
            print("[SAMCo] Step 4/4 — Running SAM segmentation ...")
        results = self.sam.predict_batch(images, prompts)
        masks   = [mask  for (mask, _)  in results]
        scores  = [score for (_, score) in results]

        # Iterative refinement: reweight saliency using current masks, then re-prompt
        for iteration in range(self.n_refine_iter):
            if verbose:
                print(f"[SAMCo] Refinement iteration {iteration + 1}/{self.n_refine_iter} ...")
            prompts = self.prompter.refine_prompts_with_masks(
                prompts, masks, saliency_maps,
                self.feat_extractor, orig_sizes
            )
            results = self.sam.predict_batch(images, prompts)
            masks   = [mask  for (mask, _)  in results]
            scores  = [score for (_, score) in results]

        if verbose:
            avg_score = np.mean(scores)
            print(f"[SAMCo] Done. Average SAM confidence: {avg_score:.3f}")

        return masks

    def segment_from_paths(
        self,
        image_paths: List[str],
        verbose:     bool = True,
    ) -> List[np.ndarray]:
        """
        Convenience wrapper: load images from disk, run segment(), return masks.
        """
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.segment(images, verbose=verbose)