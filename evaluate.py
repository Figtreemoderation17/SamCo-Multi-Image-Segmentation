"""
evaluate.py

Evaluation script for SAMCo on standard co-segmentation benchmarks.

Supported datasets: iCoseg, MSRC v2, Cosal2015

Usage:
  python evaluate.py \
    --dataset icoseg \
    --data_root /path/to/icoseg \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --n_refine_iter 2 \
    --output_dir ./results

Dataset folder structure:
  <data_root>/
    <group_name>/
      images/   *.jpg
      gt/       *.png  (255=fg, 0=bg)
"""

import os, argparse, json
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.cosegmentation import SAMCo
from utils.metrics         import (evaluate_group, evaluate_dataset,
                                   generate_all_plots, mask_statistics)
from utils.visualization   import make_results_grid, save_masks


def load_group(group_dir: str):
    """
    Load all images and ground truth masks from a single dataset group folder.

    GT masks are matched by filename stem (e.g., bear_001.jpg → bear_001.png).
    If a GT file is missing for an image, None is stored in its place.
    """
    img_dir = os.path.join(group_dir, "images")
    gt_dir  = os.path.join(group_dir, "gt")
    files   = sorted(f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png")))

    images, gt_masks, names = [], [], []
    for fname in files:
        images.append(Image.open(os.path.join(img_dir, fname)).convert("RGB"))

        gt_name = os.path.splitext(fname)[0] + ".png"
        gt_path = os.path.join(gt_dir, gt_name)
        if os.path.exists(gt_path):
            # Threshold at 127 to convert uint8 GT into a boolean mask
            gt_masks.append(np.array(Image.open(gt_path).convert("L")) > 127)
        else:
            gt_masks.append(None)

        names.append(fname)

    return images, gt_masks, names


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",        type=str,   default="icoseg",
                   help="Dataset name used for the output filename")
    p.add_argument("--data_root",      type=str,   required=True,
                   help="Root folder of the dataset (contains one subfolder per group)")
    p.add_argument("--sam_checkpoint", type=str,   required=True)
    p.add_argument("--sam_model_type", type=str,   default="vit_h")
    p.add_argument("--n_fg_points",    type=int,   default=5)
    p.add_argument("--n_bg_points",    type=int,   default=3)
    p.add_argument("--top_k_ratio",    type=float, default=0.30)
    p.add_argument("--n_refine_iter",  type=int,   default=2)
    p.add_argument("--output_dir",     type=str,   default="./results")
    p.add_argument("--save_masks",     action="store_true",
                   help="Save predicted masks for every group")
    p.add_argument("--save_grids",     action="store_true",
                   help="Save visual results grids for every group")
    p.add_argument("--save_plots",     action="store_true",
                   help="Save metric plots per group")
    p.add_argument("--max_groups",     type=int,   default=-1,
                   help="Limit evaluation to the first N groups (useful for quick checks)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = SAMCo(
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        n_fg_points=args.n_fg_points,
        n_bg_points=args.n_bg_points,
        top_k_ratio=args.top_k_ratio,
        n_refine_iter=args.n_refine_iter,
    )

    # Collect all group subdirectories and optionally limit to max_groups
    group_dirs = sorted(
        os.path.join(args.data_root, d)
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    )
    if args.max_groups > 0:
        group_dirs = group_dirs[:args.max_groups]
    print(f"[Eval] Found {len(group_dirs)} groups in {args.data_root}")

    all_pred, all_gt, group_results = [], [], {}

    for group_dir in tqdm(group_dirs, desc="Groups"):
        name = os.path.basename(group_dir)
        try:
            images, gt_masks, _ = load_group(group_dir)
        except Exception as e:
            print(f"[Eval] Skip {name}: {e}")
            continue

        # Step-by-step run so we can access saliency maps and SAM scores
        orig_sizes    = [(img.height, img.width) for img in images]
        all_feats     = model.feat_extractor.extract_batch(images)
        saliency_maps = model.prompter.compute_consensus_saliency(all_feats)
        prompts       = model.prompter.generate_prompts(saliency_maps, model.feat_extractor, orig_sizes)
        results       = model.sam.predict_batch(images, prompts)
        masks         = [m for m, _ in results]
        scores        = [s for _, s in results]

        for _ in range(args.n_refine_iter):
            prompts = model.prompter.refine_prompts_with_masks(
                prompts, masks, saliency_maps, model.feat_extractor, orig_sizes)
            results = model.sam.predict_batch(images, prompts)
            masks   = [m for m, _ in results]
            scores  = [s for _, s in results]

        # Filter out images with no GT (evaluate only where GT exists)
        valid_gt = [g for g in gt_masks if g is not None]
        metrics  = evaluate_group(masks, valid_gt if valid_gt else None)
        metrics["sam_scores"] = scores
        group_results[name] = metrics

        all_pred.append(masks)
        # Fall back to using pred masks as "GT" for groups with no annotations
        # (these groups still contribute to structural metrics like pairwise Dice)
        all_gt.append(valid_gt if valid_gt else masks)

        if args.save_masks:
            save_masks(masks, os.path.join(args.output_dir, "masks", name))

        if args.save_grids:
            grid_dir = os.path.join(args.output_dir, "grids")
            os.makedirs(grid_dir, exist_ok=True)
            make_results_grid(images, masks, saliency_maps, title=name).save(
                os.path.join(grid_dir, f"{name}.png"))

        if args.save_plots:
            plot_dir = os.path.join(args.output_dir, "plots", name)
            generate_all_plots(masks, saliency_maps, scores,
                               gt_masks=valid_gt if valid_gt else None,
                               output_dir=plot_dir)

    dataset_metrics = evaluate_dataset(all_pred, all_gt)

    print(f"\n{'='*55}")
    print(f"  Dataset           : {args.dataset}")
    print(f"  Groups evaluated  : {len(all_pred)}")
    print(f"  Dataset mean IoU  : {dataset_metrics.get('dataset_mean_iou', 0):.4f}")
    print(f"  Dataset mean Dice : {dataset_metrics.get('dataset_mean_dice', 0):.4f}")
    print(f"  Dataset mean BIoU : {dataset_metrics.get('dataset_mean_boundary_iou', 0):.4f}")
    print(f"  Dataset Pixel Acc : {dataset_metrics.get('dataset_mean_pixel_accuracy', 0):.4f}")
    print(f"  Dataset mean MAE  : {dataset_metrics.get('dataset_mean_mae', 0):.4f}")
    print(f"{'='*55}\n")

    # Save full results (dataset-level + per-group) as JSON
    # per_image metrics are excluded to keep the file compact
    out_path = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset_metrics": dataset_metrics,
            "group_metrics": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("per_image",)}
                for k, v in group_results.items()
            },
            "config": vars(args),
        }, f, indent=2)
    print(f"[Eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()