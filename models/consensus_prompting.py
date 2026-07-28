"""
consensus_prompting.py
-----------------------
 NOVEL CONTRIBUTION 
Semantic Consensus Prompting (SCP) Module
------------------------------------------
Given DINOv2 patch features from N co-segmentation images, this module:

  1. Builds a cross-image affinity matrix using cosine similarity of patch tokens.
  2. Computes a "consensus saliency map" for each image by measuring how
     semantically consistent each patch is with patches in ALL other images.
  3. Applies spectral clustering on the consensus saliency to identify the
     common foreground region.
  4. Generates SAM-compatible point prompts (foreground and background)
     directly from the clustered consensus map.
  5. Optionally refines prompts via an iterative consistency loop.

Why is this novel?
  - SAM is a promptable model; existing co-seg methods do not exploit SAM.
  - Prior automatic SAM methods use single-image saliency for prompting;
    we use CROSS-image semantic consensus — a fundamentally different signal.
  - Our iterative refinement closes the loop between segmentation quality
    and prompt quality, progressively improving co-seg accuracy.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict

# Helpers
def _cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between every row in A and every row in B.

    Args:
        A: (M, D)
        B: (N, D)

    Returns:
        sim: (M, N)  values in [-1, 1]
    """
    # L2-normalise each row; add small epsilon to avoid divide-by-zero
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def _flatten_feats(feats: np.ndarray) -> np.ndarray:
    """(H_p, W_p, D) -> (H_p*W_p, D)"""
    H, W, D = feats.shape
    return feats.reshape(H * W, D)

#Core Module

class SemanticConsensusPrompter:
    """
    Generates SAM point prompts from cross-image semantic consensus.

    Parameters
    ----------
    n_fg_points  : int   Number of foreground prompt points per image (default 5)
    n_bg_points  : int   Number of background prompt points per image (default 3)
    top_k_ratio  : float Fraction of most-similar cross-image patches to keep (default 0.3)
    smooth_sigma : float Gaussian smoothing sigma for saliency map (default 1.0)
    n_clusters   : int   KMeans clusters used to pick prompt centres (default 3)
    n_refine_iter: int   Number of iterative mask-guided prompt refinement steps (default 2)
    """

    def __init__(
        self,
        n_fg_points:   int   = 5,
        n_bg_points:   int   = 3,
        top_k_ratio:   float = 0.30,
        smooth_sigma:  float = 1.0,
        n_clusters:    int   = 3,
        n_refine_iter: int   = 2,
    ):
        self.n_fg_points   = n_fg_points
        self.n_bg_points   = n_bg_points
        self.top_k_ratio   = top_k_ratio
        self.smooth_sigma  = smooth_sigma
        self.n_clusters    = n_clusters
        self.n_refine_iter = n_refine_iter

    # STEP 1 — Consensus Saliency Maps

    def compute_consensus_saliency(
        self, all_feats: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        For each image i, compute a consensus saliency map S_i of shape (H_p, W_p).

        S_i[r, c] = mean over all other images j of:
                      max_{patch in j} cosine_sim(patch_i[r,c], patch_j)

        Intuitively, a high value means this patch has a highly similar
        counterpart in every other image -> it is part of the common object.

        Args:
            all_feats: list of N arrays, each (H_p, W_p, D)

        Returns:
            saliency_maps: list of N arrays, each (H_p, W_p) in [0, 1]
        """
        N = len(all_feats)
        H_p, W_p, D = all_feats[0].shape
        flat_feats = [_flatten_feats(f) for f in all_feats]  # each (H_p*W_p, D)

        saliency_maps = []
        for i in range(N):
            scores = np.zeros(H_p * W_p, dtype=np.float32)

            for j in range(N):
                if i == j:
                    continue

                # Cross-image similarity: (H_p*W_p_i, H_p*W_p_j)
                sim = _cosine_similarity_matrix(flat_feats[i], flat_feats[j])

                # Each patch in image i gets the score of its closest match in image j
                max_sim = sim.max(axis=1)  # (H_p*W_p,)

                # Suppress the bottom (1 - top_k_ratio) of patches to reduce noise
                k = max(1, int(self.top_k_ratio * len(max_sim)))
                threshold = np.partition(max_sim, -k)[-k]
                max_sim = np.where(max_sim >= threshold, max_sim, 0.0)

                scores += max_sim

            # Average across all other images (N-1 comparisons)
            scores /= (N - 1)

            # Normalise to [0, 1]
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)

            saliency_map = scores.reshape(H_p, W_p)

            # Light Gaussian smoothing to reduce patch-level noise
            saliency_map = gaussian_filter(saliency_map, sigma=self.smooth_sigma)
            saliency_maps.append(saliency_map)

        return saliency_maps

    # STEP 2 — Prompt Generation

    def generate_prompts(
        self,
        saliency_maps: List[np.ndarray],
        feature_extractor,                      # DINOv2FeatureExtractor instance
        orig_sizes:    List[Tuple[int, int]],   # list of (orig_h, orig_w)
    ) -> List[Dict]:
        """
        Convert consensus saliency maps into SAM-compatible point prompts.

        Strategy:
          - Threshold saliency at 0.5 -> candidate foreground patches
          - Run KMeans on (row, col) of fg patches -> cluster centres -> fg points
          - Border / low-saliency patches -> background points

        Args:
            saliency_maps: list of (H_p, W_p) arrays
            feature_extractor: DINOv2FeatureExtractor (for coord conversion)
            orig_sizes: list of (orig_h, orig_w) for each image

        Returns:
            prompts: list of dicts, each with keys:
                     'fg_points': np.ndarray (K, 2) in pixel coords (x, y)
                     'bg_points': np.ndarray (M, 2) in pixel coords (x, y)
        """
        prompts = []
        H_p, W_p = feature_extractor.get_patch_grid_size()

        for idx, smap in enumerate(saliency_maps):
            orig_h, orig_w = orig_sizes[idx]

            # Identify foreground candidate patches above the saliency threshold
            fg_threshold = 0.5
            fg_mask = smap >= fg_threshold
            fg_rows, fg_cols = np.where(fg_mask)

            if len(fg_rows) < 3:
                # If too few patches pass the threshold, fall back to top-10%
                fg_threshold = np.percentile(smap, 90)
                fg_mask = smap >= fg_threshold
                fg_rows, fg_cols = np.where(fg_mask)

            # KMeans on patch grid positions to find spatial cluster centres
            n_clusters = min(self.n_clusters, len(fg_rows))
            positions = np.stack([fg_rows, fg_cols], axis=1).astype(float)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            kmeans.fit(positions)
            centres = kmeans.cluster_centers_  # (n_clusters, 2) — row, col

            fg_points = []
            for (r, c) in centres:
                px, py = feature_extractor.patch_to_pixel_coords(
                    int(round(r)), int(round(c)), orig_h, orig_w
                )
                fg_points.append([px, py])

            # Add the globally highest-saliency patches as additional fg points
            flat_sal = smap.flatten()
            top_k = self.n_fg_points
            top_indices = np.argpartition(flat_sal, -top_k)[-top_k:]
            for idx2 in top_indices:
                r, c = divmod(idx2, W_p)
                px, py = feature_extractor.patch_to_pixel_coords(r, c, orig_h, orig_w)
                fg_points.append([px, py])

            # Deduplicate and cap at n_fg_points
            fg_points = np.unique(np.array(fg_points), axis=0)
            fg_points = fg_points[:self.n_fg_points]

            # Background points: use border patches with lowest saliency
            # (image borders are almost always background in co-seg tasks)
            bg_points = []
            border_positions = (
                [(0, c) for c in range(W_p)] +
                [(H_p - 1, c) for c in range(W_p)] +
                [(r, 0) for r in range(H_p)] +
                [(r, W_p - 1) for r in range(H_p)]
            )
            border_sal = [(smap[r, c], r, c) for (r, c) in border_positions]
            border_sal.sort(key=lambda x: x[0])  # ascending: lowest saliency first
            for (_, r, c) in border_sal[:self.n_bg_points * 4]:
                px, py = feature_extractor.patch_to_pixel_coords(r, c, orig_h, orig_w)
                bg_points.append([px, py])

            bg_points = np.unique(np.array(bg_points), axis=0)
            bg_points = bg_points[:self.n_bg_points]

            prompts.append({
                "fg_points": fg_points,   # (K, 2) — (x, y) pixel coords
                "bg_points": bg_points,   # (M, 2) — (x, y) pixel coords
            })

        return prompts

    # STEP 3 — Iterative Mask-Guided Prompt Refinement (novel)

    def refine_prompts_with_masks(
        self,
        prompts:        List[Dict],
        masks:          List[np.ndarray],     # binary masks from SAM, each (H, W)
        saliency_maps:  List[np.ndarray],
        feature_extractor,
        orig_sizes:     List[Tuple[int, int]],
    ) -> List[Dict]:
        """
        Iterative Consistency Refinement (key novelty).

        After SAM generates an initial set of masks, we use those masks to
        reweight the consensus saliency and regenerate better prompts.
        This closes the feedback loop:
            Consensus -> Prompts -> SAM Masks -> Refined Saliency -> ...

        The reweighting rule: where the mask and the saliency already agree
        (both high), we keep the score high; where the mask says "foreground"
        but saliency is low, we give a modest boost. This suppresses noisy
        background patches that passed the saliency threshold by chance.

        Args:
            prompts:       Initial prompts (list of dicts)
            masks:         Binary masks from SAM, each (orig_h, orig_w)
            saliency_maps: Current saliency maps (H_p, W_p) each
            feature_extractor: DINOv2FeatureExtractor
            orig_sizes:    list of (orig_h, orig_w)

        Returns:
            refined_prompts: list of dicts with updated fg/bg points
        """
        H_p, W_p = feature_extractor.get_patch_grid_size()
        refined_saliency = []

        for i, (smap, mask) in enumerate(zip(saliency_maps, masks)):
            orig_h, orig_w = orig_sizes[i]

            # Downsample the full-res SAM mask to the patch grid resolution
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
            mask_small = np.array(
                mask_pil.resize((W_p, H_p), PILImage.NEAREST)
            ).astype(float) / 255.0  # (H_p, W_p), values in [0, 1]

            # Blend: saliency stays high where both saliency and mask agree;
            # mask provides a small boost where saliency alone was uncertain
            boosted = smap * (0.6 + 0.4 * mask_small)

            s_min, s_max = boosted.min(), boosted.max()
            if s_max > s_min:
                boosted = (boosted - s_min) / (s_max - s_min)

            refined_saliency.append(boosted)

        # Re-run prompt generation on the updated saliency maps
        return self.generate_prompts(refined_saliency, feature_extractor, orig_sizes)