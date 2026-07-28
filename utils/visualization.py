"""
visualization.py

Visualization utilities for SAMCo co-segmentation results.

Functions

  overlay_mask_on_image    -- semi-transparent colour overlay on an image
  draw_points_on_image     -- foreground / background prompt dots
  saliency_to_heatmap      -- (H_p, W_p) float array -> PIL jet-coloured heatmap
  make_results_grid        -- N x 4 grid: original | prompts | saliency | mask
  save_masks               -- dump binary masks as PNG files
  save_metrics_report      -- self-contained HTML report with embedded plots
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for servers and scripts
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional, Dict


def overlay_mask_on_image(
    image: Image.Image,
    mask:  np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay a binary mask on an image with a semi-transparent colour.

    Args:
        image : PIL RGB image
        mask  : (H, W) bool / 0-1 array (resized automatically if needed)
        color : RGB overlay colour
        alpha : opacity of the overlay (0 = fully transparent, 1 = opaque)

    Returns:
        PIL RGB image with overlay applied
    """
    img      = image.convert("RGBA")
    # Resize mask to match image size in case they differ
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)
    mask_arr = np.array(mask_pil)

    # Build a transparent RGBA overlay and paint only the foreground pixels
    overlay_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    overlay_arr[mask_arr > 127] = (*color, int(alpha * 255))
    overlay_img = Image.fromarray(overlay_arr, "RGBA")

    return Image.alpha_composite(img, overlay_img).convert("RGB")


def draw_points_on_image(
    image:     Image.Image,
    fg_points: np.ndarray,
    bg_points: Optional[np.ndarray] = None,
    radius:    int = 6,
) -> Image.Image:
    """Draw green (foreground) and red (background) prompt dots on an image."""
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for (x, y) in fg_points:
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)],
                     fill=(0, 255, 0), outline=(0, 160, 0), width=2)

    if bg_points is not None:
        for (x, y) in bg_points:
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)],
                         fill=(255, 0, 0), outline=(160, 0, 0), width=2)
    return img


def saliency_to_heatmap(saliency: np.ndarray, size: tuple = None) -> Image.Image:
    """
    Convert a (H_p, W_p) saliency map to a jet-coloured PIL heatmap.

    Args:
        saliency : float array, any range (normalised to [0,1] internally)
        size     : optional (W, H) to resize the output to (e.g., match image thumbnail)

    Returns:
        PIL RGB heatmap image
    """
    s = saliency.astype(float)
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s = (s - s_min) / (s_max - s_min)

    # Apply jet colourmap: low saliency = blue, high = red
    heatmap = (cm.jet(s)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(heatmap)

    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img


def make_results_grid(
    images:        List[Image.Image],
    masks:         List[np.ndarray],
    saliency_maps: Optional[List[np.ndarray]] = None,
    prompts:       Optional[List[Dict]] = None,
    title:         str   = "SAMCo Results",
    thumb_size:    tuple = (256, 256),
) -> Image.Image:
    """
    Build a results grid: N rows x 4 cols
    (Original | Prompt Points | Saliency | Mask Overlay).

    Args:
        images        : list of N PIL images
        masks         : list of N binary masks
        saliency_maps : list of N saliency arrays (H_p, W_p)
        prompts       : list of N prompt dicts
        title         : title string drawn in the header
        thumb_size    : (W, H) thumbnail size for each cell

    Returns:
        PIL RGB grid image
    """
    N      = len(images)
    cols   = 4 if saliency_maps is not None else 2
    TW, TH = thumb_size
    M      = 4      # margin between cells in pixels
    HEADER = 32     # header row height

    grid_w = cols * TW + (cols + 1) * M
    grid_h = N * TH + (N + 1) * M + HEADER

    grid = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    draw.text((M, 8), title, fill=(30, 30, 30))

    col_labels = ["Original", "Prompts", "Saliency", "Mask"]
    for c, lbl in enumerate(col_labels[:cols]):
        draw.text((M + c*(TW+M) + TW//2 - 25, 14), lbl, fill=(80, 80, 80))

    for i in range(N):
        y = HEADER + i * (TH + M) + M
        thumb = images[i].convert("RGB").resize(thumb_size, Image.LANCZOS)

        # Column 0: original image
        grid.paste(thumb, (M, y))

        # Column 1: prompt points drawn on the thumbnail
        if prompts is not None and cols >= 2:
            # Scale prompt coords from original image space to thumbnail space
            scale  = np.array([TW / images[i].width, TH / images[i].height])
            fg_sc  = prompts[i]["fg_points"] * scale
            bg_sc  = (prompts[i].get("bg_points") * scale
                      if prompts[i].get("bg_points") is not None else None)
            pt_img = draw_points_on_image(thumb, fg_sc, bg_sc)
            grid.paste(pt_img, (M + TW + M, y))

        # Column 2: saliency heatmap
        if saliency_maps is not None and cols >= 3:
            sal = saliency_to_heatmap(saliency_maps[i], size=thumb_size)
            grid.paste(sal, (M + 2*(TW+M), y))

        # Column 3 (or 1 if no saliency): mask overlay
        mask_resized = masks[i]
        if mask_resized.shape[:2] != (TH, TW):
            mask_resized = np.array(
                Image.fromarray(mask_resized.astype(np.uint8)*255).resize(thumb_size, Image.NEAREST)
            ) > 127
        overlay = overlay_mask_on_image(thumb, mask_resized)
        grid.paste(overlay, (M + (cols-1)*(TW+M), y))

    return grid


def save_masks(masks: List[np.ndarray], output_dir: str, prefix: str = "mask"):
    """Save binary masks as PNG files (0 = background, 255 = foreground)."""
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        )
    print(f"[Viz] Saved {len(masks)} masks to {output_dir}")


def save_metrics_report(
    images:        List[Image.Image],
    masks:         List[np.ndarray],
    saliency_maps: List[np.ndarray],
    scores:        List[float],
    group_result:  Dict,
    output_dir:    str = "./report",
):
    """
    Save a self-contained HTML metrics report to output_dir/report.html.

    The report includes an overview grid, a per-image summary table,
    and all metric plots embedded inline as base64-encoded PNG images
    (no external files needed to open it).
    """
    import base64, io
    from utils.metrics import (plot_confidence_scores, plot_saliency_histograms,
                         plot_coverage_pie, plot_pairwise_similarity, mask_statistics)

    os.makedirs(output_dir, exist_ok=True)

    def fig_to_b64(fig) -> str:
        """Render a matplotlib figure to a base64-encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64

    stats = mask_statistics(masks)
    per   = group_result.get("per_image", [])

    # Generate all four plots and convert to base64
    b64_conf = fig_to_b64(plot_confidence_scores(scores))
    b64_sal  = fig_to_b64(plot_saliency_histograms(saliency_maps))
    b64_cov  = fig_to_b64(plot_coverage_pie(masks))
    b64_pair = fig_to_b64(plot_pairwise_similarity(stats))

    # Overview results grid
    grid = make_results_grid(images, masks, saliency_maps, title="SAMCo Results")
    buf  = io.BytesIO()
    grid.save(buf, format="PNG")
    buf.seek(0)
    b64_grid = base64.b64encode(buf.read()).decode()

    # Build per-image table rows
    table_rows = ""
    for i, m in enumerate(per):
        iou_str  = f"{m.get('iou',  0):.4f}" if isinstance(m.get('iou'),  float) else "-"
        dice_str = f"{m.get('dice', 0):.4f}" if isinstance(m.get('dice'), float) else "-"
        table_rows += f"""
        <tr>
            <td><b>Image {i+1}</b></td>
            <td>{scores[i]:.3f}</td>
            <td>{m.get('coverage', 0)*100:.1f}%</td>
            <td>{iou_str}</td>
            <td>{dice_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SAMCo Metrics Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; }}
  h1   {{ color: #e94560; }}  h2 {{ color: #0f3460; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
  th {{ background: #0f3460; color: white; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 8px 0; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style></head><body>
<h1>SAMCo — Metrics Report</h1>
<h2>Results Overview</h2>
<img src="data:image/png;base64,{b64_grid}" alt="Results Grid">

<h2>Per-Image Summary</h2>
<table><tr><th>Image</th><th>SAM Confidence</th><th>FG Coverage</th>
            <th>IoU (if GT)</th><th>Dice (if GT)</th></tr>
{table_rows}
</table>

<h2>Metric Plots</h2>
<div class="grid">
  <div><img src="data:image/png;base64,{b64_conf}" alt="Confidence"></div>
  <div><img src="data:image/png;base64,{b64_cov}"  alt="Coverage"></div>
  <div><img src="data:image/png;base64,{b64_sal}"  alt="Saliency"></div>
  <div><img src="data:image/png;base64,{b64_pair}" alt="Pairwise"></div>
</div>
</body></html>"""

    path = os.path.join(output_dir, "report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"[Viz] Metrics report saved to {path}")
    return path