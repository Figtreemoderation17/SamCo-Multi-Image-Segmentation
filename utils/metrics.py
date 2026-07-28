"""
metrics.py

Comprehensive co-segmentation evaluation metrics for SAMCo.

Metrics

  - Jaccard Index (IoU)        intersection over union
  - Dice Coefficient (F1)      harmonic mean of precision & recall
  - Precision / Recall         standard TP-based metrics
  - Pixel Accuracy             fraction correctly classified
  - Boundary IoU (BIoU)        IoU computed on boundary band only
  - Mean Absolute Error (MAE)  pixel-level absolute error
  - F-measure curve            PR curve across thresholds
  - Mask Statistics            coverage, area, pairwise similarity

Plotting utilities (matplotlib) are at the bottom of this file.
"""

import numpy as np
import cv2
import warnings
from typing import List, Dict, Optional, Tuple


# Core Pixel-Level Metrics

def jaccard_index(pred: np.ndarray, gt: np.ndarray) -> float:
    """Jaccard Index (IoU) = |P∩G| / |P∪G|."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice Coefficient = 2|P∩G| / (|P|+|G|).

    Resizes gt to match pred if shapes differ (can happen when pred is
    at original resolution and gt was loaded at a different size).
    """
    if pred.shape != gt.shape:
        # cv2.resize expects (width, height); interpolation=NEAREST preserves binary values
        gt = cv2.resize(
            gt.astype(np.uint8),
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    pred  = pred.astype(bool)
    gt    = gt.astype(bool)
    inter = (pred & gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0
    return 2.0 * float(inter) / float(total)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Returns (precision, recall). Small epsilon prevents division by zero."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    TP = float((pred & gt).sum())
    FP = float((pred & ~gt).sum())
    FN = float((~pred & gt).sum())
    return TP / (TP + FP + 1e-8), TP / (TP + FN + 1e-8)


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of pixels correctly classified (both fg and bg)."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    return float((pred == gt).sum()) / float(pred.size)


def mean_absolute_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute error between 0/1 arrays (equivalent to pixel error rate)."""
    return float(np.abs(pred.astype(float) - gt.astype(float)).mean())


def _get_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """
    Extract boundary pixels of a binary mask via morphological erosion.

    The boundary width is set as a fraction of the mask diagonal,
    so it scales sensibly with image resolution.
    """
    try:
        h, w = mask.shape
        # Radius proportional to image diagonal
        radius = max(1, int(dilation_ratio * (h**2 + w**2)**0.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        m8     = mask.astype(np.uint8) * 255
        eroded = cv2.erode(m8, kernel, iterations=1)
        return ((m8 > 0) & ~(eroded > 0)).astype(bool)
    except ImportError:
        # Numpy-only fallback (less accurate but dependency-free)
        m      = mask.astype(bool)
        eroded = (m & np.roll(m,  1, 0) & np.roll(m, -1, 0)
                    & np.roll(m,  1, 1) & np.roll(m, -1, 1))
        return m & ~eroded


def boundary_iou(pred: np.ndarray, gt: np.ndarray,
                 dilation_ratio: float = 0.02) -> float:
    """
    Boundary IoU — IoU computed only on the boundary pixel band.

    This metric penalises inaccurate contours more than standard IoU,
    making it a better measure of segmentation sharpness.
    """
    pred_b = _get_boundary(pred.astype(np.uint8), dilation_ratio)
    gt_b   = _get_boundary(gt.astype(np.uint8),   dilation_ratio)
    inter  = (pred_b & gt_b).sum()
    union  = (pred_b | gt_b).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


# PR Curve and F-measure

def precision_recall_curve(
    pred_prob:    np.ndarray,
    gt:           np.ndarray,
    n_thresholds: int   = 50,
    beta:         float = 1.0,
) -> Dict[str, np.ndarray]:
    """Full precision-recall-fmeasure curves swept across n_thresholds."""
    thresholds = np.linspace(0, 1, n_thresholds)
    precs, recs, fms = [], [], []
    beta2 = beta ** 2
    for t in thresholds:
        p, r = precision_recall((pred_prob >= t), gt)
        fm   = (1 + beta2) * p * r / (beta2 * p + r + 1e-8)
        precs.append(p)
        recs.append(r)
        fms.append(fm)

    precs    = np.array(precs)
    recs     = np.array(recs)
    fms      = np.array(fms)
    sort_idx = np.argsort(recs)

    return {
        "thresholds":   thresholds,
        "precisions":   precs,
        "recalls":      recs,
        "fmeasures":    fms,
        "auc_pr":       float(np.trapz(precs[sort_idx], recs[sort_idx])),
        "max_fmeasure": float(fms.max()),
    }


# Group and Dataset Evaluation

def _resize_pred_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Resize pred to match gt shape if they differ (nearest-neighbour to keep binary)."""
    if pred.shape != gt.shape:
        from PIL import Image as PILImage
        p = PILImage.fromarray(pred.astype(np.uint8) * 255)
        p = p.resize((gt.shape[1], gt.shape[0]), PILImage.NEAREST)
        return np.array(p) > 127
    return pred


def evaluate_single(pred: np.ndarray, gt: Optional[np.ndarray] = None) -> Dict:
    """
    Compute all metrics for one (pred, gt) pair.
    If gt is None, only coverage and area are returned.
    """
    pred     = pred.astype(bool)
    coverage = float(pred.sum()) / float(pred.size)
    result   = {"coverage": coverage, "area_pixels": int(pred.sum())}

    if gt is not None:
        pred = _resize_pred_to_gt(pred, gt)
        prec, rec = precision_recall(pred, gt)
        iou       = jaccard_index(pred, gt)
        result.update({
            "iou":            iou,
            "dice":           dice_coefficient(pred, gt),
            "precision":      prec,
            "recall":         rec,
            "pixel_accuracy": pixel_accuracy(pred, gt),
            "boundary_iou":   boundary_iou(pred, gt),
            "mae":            mean_absolute_error(pred, gt),
            "f1":             2 * prec * rec / (prec + rec + 1e-8),
        })
    return result


def evaluate_group(
    pred_masks: List[np.ndarray],
    gt_masks:   Optional[List[np.ndarray]] = None,
) -> Dict:
    """Evaluate a group of images; returns per-image metrics and group means."""
    per_image = [
        evaluate_single(p, g if gt_masks else None)
        for p, g in zip(pred_masks, (gt_masks or [None] * len(pred_masks)))
    ]
    result = {"n_images": len(pred_masks), "per_image": per_image}

    if gt_masks:
        for k in ["iou", "dice", "precision", "recall",
                  "pixel_accuracy", "boundary_iou", "mae", "f1"]:
            vals = [m[k] for m in per_image if k in m]
            if vals:
                result.update({
                    f"mean_{k}": float(np.mean(vals)),
                    f"std_{k}":  float(np.std(vals)),
                    f"list_{k}": vals,
                })
    return result


def evaluate_dataset(
    all_pred: List[List[np.ndarray]],
    all_gt:   List[List[np.ndarray]],
) -> Dict:
    """Average metrics across all groups in the dataset."""
    groups = [evaluate_group(p, g) for p, g in zip(all_pred, all_gt)]
    result = {"n_groups": len(groups)}

    for k in ["mean_iou", "mean_dice", "mean_precision", "mean_recall",
              "mean_pixel_accuracy", "mean_boundary_iou", "mean_mae"]:
        vals = [g[k] for g in groups if k in g]
        if vals:
            result[f"dataset_{k}"] = float(np.mean(vals))

    return result


def mask_statistics(masks: List[np.ndarray]) -> Dict:
    """
    Descriptive statistics for a group of masks (no GT needed).
    Includes coverage, area, and pairwise Dice between all mask pairs.
    """
    coverages = [float(m.astype(bool).sum()) / float(m.size) for m in masks]
    N = len(masks)
    pairwise = [
        dice_coefficient(masks[i], masks[j])
        for i in range(N) for j in range(i+1, N)
    ]
    return {
        "n_masks":           N,
        "coverages":         coverages,
        "mean_coverage":     float(np.mean(coverages)),
        "std_coverage":      float(np.std(coverages)),
        "areas":             [int(m.astype(bool).sum()) for m in masks],
        "pairwise_dice":     pairwise,
        "mean_pairwise_dice": float(np.mean(pairwise)) if pairwise else 0.0,
    }


# Plotting Utilities

def plot_metrics_bar(group_result: Dict,
                     title: str = "Per-Image Segmentation Metrics",
                     save_path: Optional[str] = None):
    """Grouped bar chart of IoU, Dice, Precision, Recall, BIoU, Pixel Acc."""
    import matplotlib.pyplot as plt

    per_image = group_result.get("per_image", [])
    N = len(per_image)
    if N == 0:
        return None

    metrics = {
        "IoU":       [m.get("iou", 0)           for m in per_image],
        "Dice":      [m.get("dice", 0)           for m in per_image],
        "Precision": [m.get("precision", 0)      for m in per_image],
        "Recall":    [m.get("recall", 0)         for m in per_image],
        "BIoU":      [m.get("boundary_iou", 0)   for m in per_image],
        "Pix Acc":   [m.get("pixel_accuracy", 0) for m in per_image],
    }
    x     = np.arange(N)
    width = 0.13
    n_m   = len(metrics)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#af7aa1"]

    fig, ax = plt.subplots(figsize=(max(8, N*1.5), 5))
    for idx, (label, vals) in enumerate(metrics.items()):
        offset = (idx - n_m/2) * width + width/2
        ax.bar(x + offset, vals, width, label=label, color=colors[idx % 6], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Img {i+1}" for i in range(N)])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_coverage_pie(masks: List[np.ndarray],
                      title: str = "Mask Coverage per Image",
                      save_path: Optional[str] = None):
    """Donut chart showing foreground vs background coverage for each mask."""
    import matplotlib.pyplot as plt

    N = len(masks)
    if N == 0:
        return None

    coverages = [float(m.astype(bool).sum()) / float(m.size) * 100 for m in masks]
    fig, axes = plt.subplots(1, N, figsize=(3*N, 3))
    if N == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab10", N)
    for ax, cov, col in zip(axes, coverages, [cmap(i) for i in range(N)]):
        ax.pie(
            [cov, 100-cov], labels=["FG", "BG"],
            colors=[col, "#e8e8e8"], autopct="%1.1f%%", startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"}
        )
        ax.set_title(f"Img (cov={cov:.1f}%)", fontsize=9)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_saliency_histograms(saliency_maps: List[np.ndarray],
                              title: str = "Consensus Saliency Distributions",
                              save_path: Optional[str] = None):
    """Histogram of saliency values per image, with mean and 0.5 threshold marked."""
    import matplotlib.pyplot as plt

    N    = len(saliency_maps)
    cols = min(N, 4)
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes      = np.array(axes).flatten()
    colors    = plt.cm.plasma(np.linspace(0.2, 0.9, N))

    for i, (smap, ax) in enumerate(zip(saliency_maps, axes)):
        ax.hist(smap.flatten(), bins=40, color=colors[i], alpha=0.85, edgecolor="white")
        ax.axvline(smap.mean(), color="red", ls="--", lw=1.5, label=f"μ={smap.mean():.2f}")
        ax.axvline(0.5, color="black", ls=":", lw=1)  # 0.5 is the fg threshold in generate_prompts
        ax.set_title(f"Image {i+1}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[N:]:
        ax.set_visible(False)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_iou_dice_scatter(group_result: Dict, title: str = "IoU vs Dice",
                          save_path: Optional[str] = None):
    """Scatter of IoU vs Dice with the theoretical Dice=2J/(1+J) curve overlaid."""
    import matplotlib.pyplot as plt

    per_image = group_result.get("per_image", [])
    ious  = [m.get("iou",  None) for m in per_image]
    dices = [m.get("dice", None) for m in per_image]
    if None in ious or None in dices:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    j = np.linspace(0, 1, 200)
    # Theoretical relationship between IoU (J) and Dice (D): D = 2J / (1+J)
    ax.plot(j, 2*j/(1+j), "k--", lw=1.2, alpha=0.6, label="D=2J/(1+J)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(ious)))
    for idx, (iou, dice) in enumerate(zip(ious, dices)):
        ax.scatter(iou, dice, s=120, color=colors[idx], zorder=5,
                   edgecolors="white", lw=1.5, label=f"Img {idx+1}: {iou:.3f}")

    ax.set_xlabel("IoU")
    ax.set_ylabel("Dice")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confidence_scores(scores: List[float],
                           title: str = "SAM Confidence Scores",
                           save_path: Optional[str] = None):
    """Horizontal bar chart of SAM confidence scores, coloured red-yellow-green."""
    import matplotlib.pyplot as plt

    N      = len(scores)
    labels = [f"Image {i+1}" for i in range(N)]
    cmap   = plt.cm.RdYlGn
    colors = [cmap(s) for s in scores]

    fig, ax = plt.subplots(figsize=(6, max(3, N*0.7)))
    bars = ax.barh(labels, scores, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Confidence")
    ax.set_title(title, fontweight="bold")
    ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.6)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{s:.3f}", va="center", fontsize=10)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pairwise_similarity(stats: Dict,
                              title: str = "Pairwise Mask Similarity (Dice)",
                              save_path: Optional[str] = None):
    """Heatmap of pairwise Dice similarity between all masks in a group."""
    import matplotlib.pyplot as plt

    N        = stats.get("n_masks", 0)
    pairwise = stats.get("pairwise_dice", [])
    if N < 2:
        return None

    # Unpack the upper-triangle values back into a symmetric matrix
    mat = np.eye(N)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            mat[i, j] = mat[j, i] = pairwise[idx]
            idx += 1

    fig, ax = plt.subplots(figsize=(max(4, N), max(4, N)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"Img {i+1}" for i in range(N)])
    ax.set_yticklabels([f"Img {i+1}" for i in range(N)])

    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if mat[i, j] < 0.7 else "white")

    ax.set_title(title, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_summary_radar(group_result: Dict, image_idx: int = 0,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None):
    """Radar (spider) chart of all six metrics for a single image."""
    import matplotlib.pyplot as plt

    per_image = group_result.get("per_image", [])
    if image_idx >= len(per_image):
        return None

    m     = per_image[image_idx]
    names = ["IoU", "Dice", "Precision", "Recall", "Pix Acc", "BIoU"]
    keys  = ["iou", "dice", "precision", "recall", "pixel_accuracy", "boundary_iou"]
    vals  = [m.get(k, 0.0) for k in keys]

    angles  = np.linspace(0, 2*np.pi, len(names), endpoint=False).tolist()
    vals_p  = vals + [vals[0]]    # close the polygon
    angles  = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(angles, vals_p, "o-", lw=2, color="#4e79a7")
    ax.fill(angles, vals_p, alpha=0.25, color="#4e79a7")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.4)
    ax.set_title(title or f"Metric Radar — Image {image_idx+1}", fontsize=12, pad=18)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_all_plots(
    masks:         List[np.ndarray],
    saliency_maps: List[np.ndarray],
    scores:        List[float],
    gt_masks:      Optional[List[np.ndarray]] = None,
    output_dir:    str = "./plots",
) -> Dict[str, str]:
    """
    Generate and save the full suite of analysis plots.
    Returns a dict mapping plot name -> saved file path.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    saved = {}
    stats = mask_statistics(masks)
    group = evaluate_group(masks, gt_masks)

    # Plots that don't require GT
    for fname, fn, kwargs in [
        ("confidence_scores.png",  plot_confidence_scores,  {"scores":        scores}),
        ("saliency_histograms.png", plot_saliency_histograms, {"saliency_maps": saliency_maps}),
        ("coverage_pie.png",        plot_coverage_pie,        {"masks":         masks}),
        ("pairwise_similarity.png", plot_pairwise_similarity, {"stats":         stats}),
    ]:
        p = os.path.join(output_dir, fname)
        fn(save_path=p, **kwargs)
        saved[fname.replace(".png", "")] = p

    # GT-dependent plots (metrics bar, scatter, per-image radar)
    if gt_masks:
        for fname, fn, kwargs in [
            ("metrics_bar.png",      plot_metrics_bar,      {"group_result": group}),
            ("iou_dice_scatter.png", plot_iou_dice_scatter, {"group_result": group}),
        ]:
            p = os.path.join(output_dir, fname)
            fn(save_path=p, **kwargs)
            saved[fname.replace(".png", "")] = p

        for i in range(len(masks)):
            p = os.path.join(output_dir, f"radar_img{i+1}.png")
            plot_summary_radar(group, image_idx=i, save_path=p)
            saved[f"radar_img{i+1}"] = p

    return saved
