
"""
app.py
------
SAMCo Gradio Demo — Automatic Co-segmentation via Semantic Consensus Prompting.

Key UI improvements over v1:
  - Dynamic image list: starts with 2 slots, "+" adds more (no hard limit)
  - Full metrics panel: confidence bars, saliency histograms, coverage pies,
    pairwise similarity, IoU/Dice bars, scatter plots
  - Download results button
  - Clean tabbed layout

Usage:
    python app.py --sam_checkpoint /path/to/sam_vit_h_4b8939.pth
    # Then open http://localhost:7860
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import tempfile
import numpy as np
from PIL import Image
import gradio as gr

from models.cosegmentation        import SAMCo
from utils.visualization         import overlay_mask_on_image, saliency_to_heatmap, draw_points_on_image
from utils.metrics               import (mask_statistics, evaluate_group,
                                   plot_confidence_scores, plot_saliency_histograms,
                                   plot_coverage_pie, plot_pairwise_similarity,
                                   plot_metrics_bar, plot_iou_dice_scatter)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

# ─────────────────────────────────────────────────────────────────────────────
#  Global model (loaded once)
# ─────────────────────────────────────────────────────────────────────────────
_model: SAMCo = None


def get_model(sam_checkpoint: str, sam_model_type: str, n_refine_iter: int) -> SAMCo:
    global _model
    if _model is None:
        _model = SAMCo(
            sam_model_type=sam_model_type,
            sam_checkpoint=sam_checkpoint,
            n_fg_points=5, n_bg_points=3,
            top_k_ratio=0.30, n_refine_iter=n_refine_iter,
        )
    return _model


def _fig_to_np(fig) -> np.ndarray:
    """Convert a matplotlib figure to a numpy RGBA array for Gradio."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return np.array(img)

#  Core inference

def run_cosegmentation(
    image_list,          # list[np.ndarray]  — from gr.State
    n_fg, n_bg, top_k, n_refine,
    sam_checkpoint, sam_model_type,
):
    """
    Runs the full SAMCo pipeline and returns:
      [overlay images] + [saliency images] + [prompt images]
      + metric plot images + status string
    """
    imgs_np = [x for x in (image_list or []) if x is not None]
    if len(imgs_np) < 2:
        return ([], [], [], None, None, None, None, None,
                "⚠️  Please upload at least 2 images.")

    images = [Image.fromarray(img).convert("RGB") for img in imgs_np]
    N = len(images)

    try:
        model = get_model(sam_checkpoint, sam_model_type, int(n_refine))
        model.prompter.n_fg_points   = int(n_fg)
        model.prompter.n_bg_points   = int(n_bg)
        model.prompter.top_k_ratio   = float(top_k)
        model.n_refine_iter          = int(n_refine)
        model.prompter.n_refine_iter = int(n_refine)

        orig_sizes    = [(img.height, img.width) for img in images]
        all_feats     = model.feat_extractor.extract_batch(images)
        saliency_maps = model.prompter.compute_consensus_saliency(all_feats)
        prompts       = model.prompter.generate_prompts(saliency_maps, model.feat_extractor, orig_sizes)
        results       = model.sam.predict_batch(images, prompts)
        masks         = [m for m, _ in results]
        scores        = [s for _, s in results]

        for _ in range(int(n_refine)):
            prompts = model.prompter.refine_prompts_with_masks(
                prompts, masks, saliency_maps, model.feat_extractor, orig_sizes
            )
            results = model.sam.predict_batch(images, prompts)
            masks   = [m for m, _ in results]
            scores  = [s for _, s in results]

    except Exception as e:
        return ([], [], [], None, None, None, None, None, f" Error: {str(e)}")

    # ── Build visualisation outputs ───────────────────────────────────
    overlays, saliencies, prompt_imgs = [], [], []
    for img, mask, smap, prompt in zip(images, masks, saliency_maps, prompts):
        overlays.append(np.array(overlay_mask_on_image(img, mask, color=(50, 220, 80), alpha=0.45)))
        blended = Image.blend(img, saliency_to_heatmap(smap, size=img.size), alpha=0.6)
        saliencies.append(np.array(blended))
        prompt_imgs.append(np.array(draw_points_on_image(img, prompt["fg_points"], prompt.get("bg_points"))))

    # Metric plots
    stats = mask_statistics(masks)
    group = evaluate_group(masks)   # no GT in demo

    fig_conf  = plot_confidence_scores(scores, title="SAM Confidence Scores per Image")
    fig_sal   = plot_saliency_histograms(saliency_maps, title="Consensus Saliency Distributions")
    fig_cov   = plot_coverage_pie(masks, title="Foreground Coverage per Image")
    fig_pair  = plot_pairwise_similarity(stats, title="Pairwise Mask Similarity (Dice)")

    metric_conf = _fig_to_np(fig_conf)
    metric_sal  = _fig_to_np(fig_sal)
    metric_cov  = _fig_to_np(fig_cov)
    metric_pair = _fig_to_np(fig_pair)

    status = (f" Done!  {N} images co-segmented. "
              f"Mean coverage: {stats['mean_coverage']*100:.1f}%  |  "
              f"Mean pairwise Dice: {stats['mean_pairwise_dice']:.3f}  |  "
              f"Mean SAM confidence: {np.mean(scores):.3f}")

    return (overlays, saliencies, prompt_imgs,
            metric_conf, metric_sal, metric_cov, metric_pair,
            status)

#  Dynamic image state helpers

def add_image(image_list, new_img):
    """Append a newly uploaded image to the state list."""
    if new_img is None:
        return image_list, image_list
    lst = list(image_list or [])
    lst.append(new_img)
    return lst, lst


def remove_image(image_list, index):
    """Remove image at given index from the state list."""
    lst = list(image_list or [])
    if 0 <= index < len(lst):
        lst.pop(index)
    return lst, lst


def clear_images():
    """Clear all uploaded images."""
    return [], []

#  Build Gradio UI

def build_ui(sam_checkpoint: str, sam_model_type: str) -> gr.Blocks:

    css = """
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 24px; border-radius: 12px; margin-bottom: 16px;
    }
    .header h1 { color: #e94560; font-size: 2em; margin: 0; }
    .header p  { color: #a8b2d8; margin: 6px 0 0 0; }
    .badge {
        display: inline-block; background: #e94560; color: white;
        padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 8px;
    }
    .image-card { border: 1px solid #333; border-radius: 8px; padding: 6px; }
    .add-btn { font-size: 2em !important; height: 80px !important; }
    """

    with gr.Blocks(title="SAMCo — Automatic Co-segmentation", css=css) as demo:

        # ── Banner ──────────────────────────────────────────────────
        gr.HTML("""
        <div class="header">
          <h1>SAMCo <span class="badge">EE655 · IIT Kanpur</span></h1>
          <p><b>Automatic Co-segmentation via Semantic Consensus Prompting</b></p>
          <p style="font-size:0.85em; color:#64748b;">
            Upload 2 or more images sharing a common object.
            SAMCo finds and segments it in all images — no manual prompts needed.
          </p>
        </div>
        """)

        # ── State: list of uploaded images ──────────────────────────
        image_state = gr.State([])   # List[np.ndarray]

        with gr.Tabs():

            # ════════════════════════════════════════════════════════
            #  TAB 1 — Input & Results
            # ════════════════════════════════════════════════════════
            with gr.Tab("Co-segmentation"):
                with gr.Row():

                    # ── Left column: image management ───────────────
                    with gr.Column(scale=1, min_width=320):

                        gr.Markdown(" Upload Images")
                        gr.Markdown(
                            "_Add at least 2 images. Click **＋ Add Image** to upload more._"
                        )

                        # Gallery preview of current images
                        gallery = gr.Gallery(
                            label="Uploaded Images",
                            columns=2, rows=3,
                            height=340,
                            object_fit="contain",
                            show_label=True,
                        )

                        with gr.Row():
                            upload_btn = gr.UploadButton(
                                "➕  Add Image",
                                file_types=["image"],
                                variant="secondary",
                                size="sm",
                            )
                            clear_btn  = gr.Button("🗑️  Clear All", variant="stop", size="sm")

                        gr.Markdown("---")
                        gr.Markdown(" ⚙️ Pipeline Parameters")
                        n_fg   = gr.Slider(1, 10, value=5, step=1,    label="Foreground Prompt Points")
                        n_bg   = gr.Slider(1, 6,  value=3, step=1,    label="Background Prompt Points")
                        top_k  = gr.Slider(0.1, 0.7, value=0.3, step=0.05, label="Top-K Consensus Ratio")
                        refine = gr.Slider(0, 4, value=2, step=1,     label="Refinement Iterations")

                        sam_ckpt_st  = gr.State(sam_checkpoint)
                        sam_type_st  = gr.State(sam_model_type)

                        run_btn = gr.Button("  Run SAMCo", variant="primary", size="lg")

                    # ── Right column: results ────────────────────────
                    with gr.Column(scale=2):
                        status_box = gr.Textbox(label="Status", interactive=False)

                        with gr.Tabs():
                            with gr.Tab(" Segmentation Masks"):
                                mask_gallery = gr.Gallery(
                                    label="Mask Overlays (green = common object)",
                                    columns=3, rows=2, height=420,
                                    object_fit="contain",
                                )
                            with gr.Tab("Saliency Maps"):
                                sal_gallery = gr.Gallery(
                                    label="Consensus Saliency (red = high consensus)",
                                    columns=3, rows=2, height=420,
                                    object_fit="contain",
                                )
                            with gr.Tab(" Prompt Points"):
                                prompt_gallery = gr.Gallery(
                                    label="Auto-generated Prompts (green=FG, red=BG)",
                                    columns=3, rows=2, height=420,
                                    object_fit="contain",
                                )

            #  TAB 2 — Metrics & Analytics
            with gr.Tab(" Metrics & Analytics"):
                gr.Markdown(
                    "### Analysis plots generated after running SAMCo\n"
                    "_Run the pipeline first, then switch to this tab._"
                )
                with gr.Row():
                    metric_conf_img  = gr.Image(label="SAM Confidence Scores",      height=300)
                    metric_cov_img   = gr.Image(label="Foreground Coverage (Pie)",   height=300)
                with gr.Row():
                    metric_sal_img   = gr.Image(label="Saliency Distributions",      height=300)
                    metric_pair_img  = gr.Image(label="Pairwise Mask Similarity",    height=300)
                gr.Markdown("""
> **Metric Glossary**
> - **SAM Confidence**: the IoU-prediction score from SAM for the selected mask (higher = more confident)
> - **Foreground Coverage**: fraction of image pixels classified as the common object
> - **Consensus Saliency Distribution**: histogram of cross-image cosine-similarity scores per patch
> - **Pairwise Mask Similarity (Dice)**: how consistent the co-segmentation masks are with each other
                """)

            # ════════════════════════════════════════════════════════
            #  TAB 3 — How It Works
            # ════════════════════════════════════════════════════════
            with gr.Tab("ℹ️ How It Works"):
                gr.Markdown("""
## SAMCo Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | **DINOv2 Feature Extraction** | ViT-S/14 → 37×37 patch tokens (384-dim) per image |
| 2 | **Cross-Image Consensus Saliency** ★ | For each patch, max cosine-sim to patches in all other images → saliency score |
| 3 | **Semantic Consensus Prompting** ★ | KMeans on high-saliency patches → auto SAM point prompts |
| 4 | **SAM Segmentation** | SAM uses prompts to produce precise binary masks |
| 5 | **Iterative Consistency Refinement** ★ | Masks re-weight saliency → better prompts → better masks |

★ = novel contributions

---

### Why DINOv2?
DINOv2 (DINO v2) is a self-supervised Vision Transformer trained on 142M images.
Its patch tokens exhibit remarkable *semantic consistency* — patches of the same
object class produce similar embeddings **across different images and viewpoints**.
This makes it ideal as the backbone for cross-image co-segmentation.

### Why SAM?
SAM (Segment Anything Model) produces state-of-the-art segmentation *given prompts*.
The challenge for co-segmentation is: **where do the prompts come from?**
SAMCo answers this by deriving prompts automatically from cross-image consensus.

### Iterative Refinement
After the first SAM forward pass, the masks reveal which pixels are foreground.
We use this information to *re-weight* the saliency map (boost high-saliency fg pixels,
suppress false positives) and re-generate prompts. This loop typically runs 2 times and
measurably improves mask quality.
                """)

        # ──────────────────────────────────────────────────────────
        #  Event wiring
        # ──────────────────────────────────────────────────────────

        def upload_and_update(file, current_list):
            if file is None:
                return current_list, current_list
            img_np = np.array(Image.open(file.name).convert("RGB"))
            lst = list(current_list or [])
            lst.append(img_np)
            return lst, lst

        upload_btn.upload(
            fn=upload_and_update,
            inputs=[upload_btn, image_state],
            outputs=[image_state, gallery],
        )

        clear_btn.click(
            fn=lambda: ([], []),
            outputs=[image_state, gallery],
        )

        run_btn.click(
            fn=run_cosegmentation,
            inputs=[image_state, n_fg, n_bg, top_k, refine, sam_ckpt_st, sam_type_st],
            outputs=[
                mask_gallery, sal_gallery, prompt_gallery,
                metric_conf_img, metric_sal_img, metric_cov_img, metric_pair_img,
                status_box,
            ],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth")
    p.add_argument("--sam_model_type", type=str, default="vit_h")
    p.add_argument("--port",           type=int, default=7860)
    p.add_argument("--share",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_ui(args.sam_checkpoint, args.sam_model_type)
    demo.launch(server_port=args.port, share=args.share)
