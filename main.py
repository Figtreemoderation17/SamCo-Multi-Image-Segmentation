import argparse
import os
import sys
import numpy as np
from PIL import Image

from models.cosegmentation  import SAMCo
from utils.visualization   import make_results_grid, save_masks, overlay_mask_on_image
from utils.metrics         import (mask_statistics, evaluate_group,
                                      generate_all_plots, plot_confidence_scores)

def parse_args():
    p = argparse.ArgumentParser(
        description="SAMCo: Automatic Co-segmentation via Semantic Consensus Prompting"
    )
    p.add_argument("--images",         nargs="+", required=False,
                   help="Paths to input images (omit for interactive input)")
    p.add_argument("--sam_checkpoint", type=str,  default="sam_vit_h_4b8939.pth",
                   help="Path to SAM checkpoint (.pth file)")
    p.add_argument("--sam_model_type", type=str,  default="vit_h",
                   help="SAM variant: vit_h | vit_l | vit_b")
    p.add_argument("--n_fg_points",    type=int,  default=5,
                   help="Number of foreground prompt points per image")
    p.add_argument("--n_bg_points",    type=int,  default=3,
                   help="Number of background prompt points per image")
    p.add_argument("--top_k_ratio",    type=float, default=0.30,
                   help="Fraction of cross-image patches used for consensus scoring")
    p.add_argument("--n_refine_iter",  type=int,  default=2,
                   help="Number of mask-guided refinement iterations (0 = disabled)")
    p.add_argument("--output_dir",     type=str,  default="./output")
    p.add_argument("--visualize",      action="store_true",
                   help="Save a side-by-side results grid to output_dir")
    p.add_argument("--metrics",        action="store_true",
                   help="Generate and save metric plots to output_dir/plots/")
    p.add_argument("--device",         type=str,  default="auto",
                   help="'cuda', 'cpu', or 'auto' (auto-detects CUDA)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = args.images

    # Interactive fallback: if no images were passed on the CLI, ask for them now
    if not image_paths:
        try:
            num_images = int(input("How many images do you want to input? "))
            image_paths = []
            for i in range(num_images):
                path = input(f"Enter path for image {i+1} (e.g., demo/image1.png): ")
                image_paths.append(path)
        except ValueError:
            print("[Error] Invalid input for number of images. Exiting.")
            sys.exit(1)

    if not image_paths:
        print("[Error] No images provided. Exiting.")
        sys.exit(1)

    # Load all images, checking that each file actually exists
    print(f"[SAMCo] Loading {len(image_paths)} images ...")
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"[Error] File not found: {path}")
            sys.exit(1)
        img = Image.open(path).convert("RGB")
        images.append(img)
        print(f"  {os.path.basename(path):30s}  {img.width}×{img.height}")

    # Build the SAMCo model (loads DINOv2 + SAM; takes a few seconds on first run)
    model = SAMCo(
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        n_fg_points=args.n_fg_points,
        n_bg_points=args.n_bg_points,
        top_k_ratio=args.top_k_ratio,
        n_refine_iter=args.n_refine_iter,
        device=args.device,
    )

    # Run the pipeline step by step so we can capture intermediate outputs
    # (saliency maps and prompts are needed for visualisation and metric plots)
    orig_sizes    = [(img.height, img.width) for img in images]
    all_feats     = model.feat_extractor.extract_batch(images)
    saliency_maps = model.prompter.compute_consensus_saliency(all_feats)
    prompts       = model.prompter.generate_prompts(saliency_maps, model.feat_extractor, orig_sizes)
    results       = model.sam.predict_batch(images, prompts)
    masks         = [m for m, _ in results]
    scores        = [s for _, s in results]

    # Iterative mask-guided refinement: each iteration reweights saliency
    # using the current masks, then re-prompts SAM for better boundaries
    for _ in range(args.n_refine_iter):
        prompts = model.prompter.refine_prompts_with_masks(
            prompts, masks, saliency_maps, model.feat_extractor, orig_sizes
        )
        results = model.sam.predict_batch(images, prompts)
        masks   = [m for m, _ in results]
        scores  = [s for _, s in results]

    print(f"[SAMCo] Done. Mean SAM confidence: {np.mean(scores):.3f}")

    # Save binary masks and semi-transparent overlays
    save_masks(masks, os.path.join(args.output_dir, "masks"))

    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    for i, (img, mask) in enumerate(zip(images, masks)):
        fname = os.path.splitext(os.path.basename(image_paths[i]))[0]
        overlay_mask_on_image(img, mask).save(
            os.path.join(overlay_dir, f"{fname}_overlay.png"))
    print(f"[SAMCo] Overlays saved to {overlay_dir}")

    # Optional: N-row results grid (original | prompts | saliency | mask)
    if args.visualize:
        grid = make_results_grid(images, masks, saliency_maps, prompts,
                                  title="SAMCo Co-segmentation")
        grid.save(os.path.join(args.output_dir, "results_grid.png"))
        print(f"[SAMCo] Grid saved to {args.output_dir}/results_grid.png")

    # Optional: full suite of metric and saliency plots
    if args.metrics:
        plot_dir = os.path.join(args.output_dir, "plots")
        saved = generate_all_plots(masks, saliency_maps, scores, output_dir=plot_dir)
        print(f"[SAMCo] {len(saved)} metric plots saved to {plot_dir}/")

    # Print a quick summary to stdout (no GT available here, so no IoU)
    stats = mask_statistics(masks)
    print(f"\n{'='*50}")
    print(f"  N images          : {stats['n_masks']}")
    print(f"  Mean FG coverage  : {stats['mean_coverage']*100:.1f}%")
    print(f"  Std FG coverage   : {stats['std_coverage']*100:.1f}%")
    print(f"  Mean pairwise Dice: {stats['mean_pairwise_dice']:.4f}")
    print(f"  Mean SAM conf.    : {np.mean(scores):.4f}")
    print(f"{'='*50}\n")

    print("[SAMCo] Pipeline complete!")


if __name__ == "__main__":
    main()