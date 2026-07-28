# SAMCo: Automatic Co-segmentation via Semantic Consensus Prompting

**Course Project — EE655 Computer Vision and Deep Learning, IIT Kanpur (2025-26 Semester II)**

---

## What This Project Does

Co-segmentation is the task of finding and precisely outlining a common object across multiple images, without being told what the object is. For example, given five photos each containing a dog, the system should automatically draw an accurate mask around the dog in every photo — even though the dogs are in different poses, backgrounds, and lighting conditions.

SAMCo solves this problem without any manual prompts, without any labelled training data, and without knowing the name of the object. You give it a set of images; it figures out what they have in common and segments it.

The system works by combining two state-of-the-art models from Meta AI — SAM (Segment Anything Model) and DINOv2 — with a novel prompting strategy called Semantic Consensus Prompting that was designed specifically for this project.

---

## How It Works 

The pipeline has four steps.

**Step 1 — Feature Extraction.** Each image is passed through DINOv2, a self-supervised Vision Transformer. The model divides each image into a 37x37 grid of small patches and produces a 384-dimensional feature vector for each patch. These vectors capture semantic meaning: patches showing "fur" or "snout" will have similar feature vectors regardless of the image they came from.

**Step 2 — Consensus Saliency.** For every patch in Image 1, the system measures how similar it is to the most similar patch in Image 2, then Image 3, and so on. The average of these maximum similarities is the "consensus score" for that patch. Patches that look like they could belong to an object also present in every other image score high. Patches that are unique to one image (background clutter, different walls, different grass) score low. The result is a saliency map for each image, highlighting where the common object likely is.

**Step 3 — Prompt Generation.** High-scoring patches are clustered using KMeans to find the spatial centres of the likely foreground object. These centres, along with the lowest-scoring border patches as background cues, are formatted as point prompts and passed to SAM.

**Step 4 — Iterative Refinement.** SAM produces an initial segmentation mask. That mask is then used to reweight the saliency map — regions where the mask and the saliency agree get boosted, sharpening the signal. New prompts are generated from this improved saliency, and SAM runs again. This loop repeats two times by default, progressively improving the masks.

---

## Results

| Method                    | iCoseg mIoU | MSRC mIoU | Boundary IoU |
|---------------------------|-------------|-----------|--------------|
| GrabCut                   | 0.621       | 0.574     | —            |
| COSNet (supervised)       | 0.712       | 0.663     | 0.691        |
| GICD (supervised)         | 0.735       | 0.681     | 0.714        |
| Grounded-SAM (needs label)| 0.758       | 0.706     | —            |
| SAMCo                     | 0.781       | 0.729     | 0.752        |

SAMCo outperforms all baselines including supervised methods, despite requiring no training and no class labels. Grounded-SAM, its closest competitor, requires you to provide the name of the object as a text prompt.

---

## Installation

You will need Python 3.9 or above and either a CUDA GPU (recommended) or a CPU. On CPU, each image group will take roughly 30-60 seconds.

**Step 1 — Clone the repository**

```bash
git clone https://github.com/dsjc101/SAMCo.git
cd SAMCo
```

**Step 2 — Install Python dependencies**

```bash
pip install -r requirements.txt
```

This installs PyTorch, the Segment Anything package, scikit-learn, OpenCV, Gradio, and other dependencies. DINOv2 is loaded automatically from Meta's servers via torch.hub the first time you run the code (approximately 80 MB download).

**Step 3 — Download a SAM checkpoint**

SAM requires a pre-trained weights file. The ViT-H variant gives the best results but is a large download.

```bash
# Best quality (2.4 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Faster, slightly lower quality (375 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Place the downloaded file in the root of the project folder.

---

## Quick Start

The fastest way to use SAMCo is through the command-line script.

```bash
python main.py \
  --images path/to/image1.jpg path/to/image2.jpg \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --output_dir ./output \
  --visualize
```

You can pass as many images as you like. The more images you provide that share the same object, the better the consensus signal.

```bash
python main.py \
  --images dog1.jpg dog2.jpg dog3.jpg dog4.jpg \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --n_refine_iter 1 \
  --output_dir ./output \
  --visualize \
  --metrics
```
If you run `python main.py` without `--images`, it will prompt you interactively to enter image paths one by one.

Output files are saved to the folder you specify:

- `output/masks/` — binary mask PNGs for each image
- `output/overlays/` — original images with the segmentation mask blended on top
- `output/results_grid.png` — a side-by-side comparison grid (if `--visualize` is passed)
- `output/plots/` — metric and saliency visualisation plots (if `--metrics` is passed)

---

## Interactive Demo

If you prefer a web interface, SAMCo includes a Gradio demo that runs in your browser.

```bash
cd demo
python app.py --sam_checkpoint ../sam_vit_h_4b8939.pth
```

Then open `http://localhost:7860` in your browser. The interface lets you upload any number of images using the "Add Image" button, runs SAMCo when you click segment, and displays the results across four tabs: masks, saliency maps, prompt point visualisations, and a full metrics panel.

---

## Evaluating on a Dataset

To reproduce the benchmark numbers from the results table, use the evaluation script.

```bash
python evaluate.py \
  --dataset icoseg \
  --data_root /path/to/icoseg \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --n_refine_iter 2 \
  --output_dir ./results \
  --save_masks \
  --save_grids
```

Your dataset folder must follow this structure:

```
icoseg/
    bear/
        images/
            bear_001.jpg
            bear_002.jpg
            ...
        gt/
            bear_001.png
            bear_002.png
            ...
    elephant/
        images/
            ...
        gt/
            ...
```

Ground truth masks should be single-channel PNG files where pixel value 255 means foreground and 0 means background.

Results are saved as a JSON file at `results/results_icoseg.json` containing per-group and dataset-level metrics.

Supported dataset flags: `--dataset icoseg`, `--dataset msrc`, `--dataset cosal2015`.

Download links for the datasets:
- iCoseg: https://ieeexplore.ieee.org/document/5540080
- MSRC v2: https://www.microsoft.com/en-us/research/project/image-understanding/
- Cosal2015: https://ieeexplore.ieee.org/document/6548102

---

## Project File Structure
```
SAMCo/
|
|-- main.py                     entry point for CLI usage
|-- evaluate.py                 benchmarking against ground truth datasets
|-- requirements.txt
|-- README.md
|
|-- models/
|   |-- cosegmentation.py       top-level SAMCo pipeline class
|   |-- consensus_prompting.py  core novel module: SCP + iterative refinement
|   |-- feature_extractor.py    DINOv2 ViT-S/14 wrapper
|   |-- sam_wrapper.py          SAM predictor wrapper
|
|-- utils/
|   |-- metrics.py              IoU, Dice, BIoU, and all plotting utilities
|   |-- visualization.py        overlays, grids, saliency heatmaps
|
|-- demo/
|   |-- app.py                  Gradio web interface
|
|-- output/
    |-- masks/                  binary masks per image
    |-- overlays/               mask-blended original images
    |-- plots/                  metric and saliency plots
    |-- results_grid.png        side-by-side comparison grid
```
---

## File-by-File Explanation

### main.py

The entry point for running SAMCo on your own images from the command line. It parses your arguments, loads the images, builds the SAMCo model, runs the full pipeline step by step, saves the masks and overlays to disk, and optionally generates a results grid and metric plots. If you just want to try the system on a few images, this is the file to run.

### evaluate.py

The benchmarking script used to generate the numbers in the results table. It loads a dataset from disk (iCoseg, MSRC, etc.), runs SAMCo on each group of images, compares the predicted masks against ground truth using all metrics, and saves a JSON summary. It also supports saving per-group mask files, visualisation grids, and metric plots. Use this if you want to reproduce or extend the experiments.

### models/cosegmentation.py

The top-level pipeline class `SAMCo` that ties all three sub-modules together. It initialises the feature extractor, the consensus prompter, and SAM, then provides a `segment()` method that runs the full four-step pipeline. If you want to use SAMCo as a library in your own code, import from this file. The logic for the iterative refinement loop also lives here.

### models/consensus_prompting.py

This is the core of the novel contribution. The `SemanticConsensusPrompter` class implements three methods:

`compute_consensus_saliency()` — takes DINOv2 features from all N images and computes cross-image cosine similarity to produce a saliency map for each image, highlighting where the common object likely is.

`generate_prompts()` — converts a saliency map into SAM-compatible point prompts by running KMeans on high-saliency patches for foreground points and sampling border patches for background points.

`refine_prompts_with_masks()` — implements Iterative Consistency Refinement. Takes the masks produced by SAM, blends them back into the saliency maps to boost agreement regions, and regenerates improved prompts.

### models/feature_extractor.py

A wrapper around Meta's DINOv2 ViT-S/14 model. It loads the model via `torch.hub`, handles the image preprocessing (resize to 518x518, normalise), runs the forward pass, and returns the patch tokens reshaped as a spatial feature grid of shape (37, 37, 384). It also provides a utility method `patch_to_pixel_coords()` that converts patch grid indices back to pixel coordinates in the original image, which is needed when generating SAM prompts.

### models/sam_wrapper.py

A thin wrapper around Meta's `SamPredictor` class. It loads the SAM model from a checkpoint file, provides a `predict_with_points()` method that accepts foreground and background point arrays, runs SAM's multi-mask prediction, and returns the mask with the highest confidence score. The `predict_batch()` method runs this over a list of images and their corresponding prompts.

### utils/metrics.py

All evaluation logic lives here. It computes IoU (Intersection over Union), Dice coefficient, Precision, Recall, Pixel Accuracy, Boundary IoU (evaluated only on the edge band of the mask, not the full region), and MAE. It also has functions for aggregating metrics across a full dataset and generating all visualisation plots: per-image bar charts, radar charts, coverage pie charts, saliency histograms, pairwise mask similarity heatmaps, and an IoU vs Dice scatter plot with the theoretical curve overlaid.

### utils/visualization.py

Handles everything visual that is not a metric plot. This includes overlaying a semi-transparent coloured mask on an original image, rendering the SAM prompt points as coloured dots on the image, creating the multi-column results grid that shows original, saliency map, prompts, and mask side by side for all images, and saving mask files as PNG.

### demo/app.py

A Gradio web application. It builds a dynamic UI where users can upload any number of images, click a segment button, and see results across four tabs (masks, saliency maps, prompt visualisations, metrics panel). The "Add Image" button appends a new upload slot dynamically, and "Clear All" resets the interface. This file is self-contained and can be run independently of the rest of the project as long as the models are available.

---

## Tunable Parameters

All parameters can be passed as command-line flags to `main.py` and `evaluate.py`.

| Parameter       | Default | What it controls |
|-----------------|---------|-----------------|
| `n_fg_points`   | 5       | Number of foreground point prompts sent to SAM per image. More points give broader coverage of the object. |
| `n_bg_points`   | 3       | Number of background point prompts. These help SAM exclude the borders. |
| `top_k_ratio`   | 0.30    | Fraction of cross-image patch comparisons kept per image. Lower values make the saliency sharper but risk missing parts of the object. |
| `n_refine_iter` | 2       | How many times the mask-guided refinement loop runs. 0 disables refinement entirely. 2 is the sweet spot between quality and speed. |
| `smooth_sigma`  | 1.0     | Gaussian smoothing applied to saliency maps. Higher values reduce noise but blur the boundaries. |
| `n_clusters`    | 3       | Number of KMeans clusters used to select foreground prompt centres. |
| `sam_model_type`| vit_h   | SAM variant. vit_h is most accurate, vit_b is fastest. |

---

## Dependencies

All required packages are listed in `requirements.txt`. The key ones are:

- **torch and torchvision** — deep learning backbone for both DINOv2 and SAM
- **segment-anything** — Meta's SAM package, installed directly from their GitHub
- **scikit-learn** — used for KMeans clustering in the prompt generation step
- **scipy** — used for Gaussian filtering of saliency maps
- **opencv-python** — used for boundary extraction in the BIoU metric calculation
- **Pillow** — image loading, resizing, and saving throughout the codebase
- **matplotlib** — all metric and saliency visualisation plots
- **gradio** — the interactive web demo UI
- **tqdm** — progress bars during dataset evaluation

DINOv2 is not a pip package. It is loaded automatically on first run via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`, which downloads approximately 80 MB of model weights and caches them locally.

---

