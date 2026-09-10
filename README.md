# OsteoGen 2.0

From a fossil/skeleton to **(1)** a percentage geometric-compatibility report
against a database of living animals, and **(2)** an optional generative
render of the reconstructed animal, blending the textures of the most
compatible donor species. Works even for extinct animals never photographed
alive (e.g. dinosaurs), because the comparison is based only on skeleton
shape, never on a photo of the target.

For the full write-up (method, related work, results, limitations) see
**[`report/main.pdf`](report/main.pdf)** - that is the primary document;
this README is only a practical guide to the code.

**Use of AI**: this project used an AI coding assistant for part of its
engineering (interactive CLI, bug fixes, a keypoint-detector fine-tuning
experiment) on top of independently-developed modeling work. Full,
itemized disclosure is in `report/main.tex`, Section "The Use of AI" -
read that before anything else if you're grading this.

## Repository layout

```
OsteoGen_V.2/   <- THE deliverable. Run this. Everything below is about it.
OsteoGen_V.1/   <- earlier iteration (prompt-conditioned ControlNet)
OsteoGen_V.0/   <- earliest iteration (U-Net / PatchGAN / unconditional
                   ControlNet baseline) - both kept as historical/reference
                   material, described and compared against V.2 in the
                   report; not meant to be re-run (see "About V.0 and V.1").
report/         <- the report itself: main.tex + main.pdf, self-contained
                   (figures, references, style files all included).
```

`OsteoGen_V.2` is a single, self-installing pipeline: preprocessing ->
14-keypoint detection (ResNet18) -> geometric retrieval against a 65-species
database -> a JSON/HTML report -> an optional generative render (Stable
Diffusion + ControlNet + IP-Adapter). See `OsteoGen_V.2/documentazione_finale.txt`
for a per-module architecture reference, and
`OsteoGen_V.2/dataset_tools/SOURCES.md` for how to add more species to the
dataset.

## Quick start (the only command you actually need)

```bash
cd OsteoGen_V.2
python pipeline.py
```

(Equivalently, `python run.py` from the repo root - it asks which version
to run, defaulting to 2, then launches the command above for you. Useful
if you also want to try V.0/V.1; see "About V.0 and V.1" below.)

That's it. No setup steps before this. On first run it will, in order and
telling you what it's doing at each step:

1. **Install whatever Python dependencies are missing** (needs `pip`
   available for whichever `python`/`python3` you ran it with - if it
   isn't, the script prints the exact commands to create a virtual
   environment and stops there instead of failing unhelpfully).
2. **Download the trained keypoint-detector weights automatically** from a
   public Hugging Face Hub repo (`markcst/osteogen-keypoint-detector`, ~54MB,
   no account/token needed). It always re-checks for a newer version on
   Hugging Face before trusting whatever is already local, so it can't get
   silently stuck on an outdated copy.
3. **Ask for the fossil/skeleton image** - type a path or drag the file
   into the terminal.
4. **Ask whether to also generate the final render** - this stage needs a
   GPU for a reasonable runtime (a couple of minutes on an NVIDIA GPU; on
   CPU it works too, just much slower - expect many minutes). Answering
   "no" (or passing `--no-render`) skips straight to the report, which
   needs no GPU at all.
5. Run every stage with a visible progress indicator, then open the report
   and the produced images automatically.

For scripted/non-interactive use, the underlying flags still work and skip
all the prompts:

```bash
# Report only, no GPU needed at all:
python pipeline.py --image input/t-rex.jpg --no-render

# Report + generative render:
python pipeline.py --image input/t-rex.jpg

# Use a specific weights file instead of the auto-downloaded default:
python pipeline.py --image input/t-rex.jpg --weights /path/to/other_weights.pth
```

### If something doesn't work

- **No GPU, or an AMD/Intel GPU instead of NVIDIA**: the report stage
  (`--no-render`) is unaffected - it never needs a GPU. The render stage
  will run on CPU (slow) rather than fail; if that's not acceptable, just
  skip it.
- **`pip` not found for your Python**: the script tells you the exact
  `python -m venv` commands to run first; do that once, then re-run
  `pipeline.py` inside that environment.
- **Weights fail to download (offline, firewall, etc.)**: the script falls
  back to asking for a local path once, then remembers it - point it at a
  `.pth` file if you have one another way.

## What each `OsteoGen_V.2` file does

```
input image
     |
     v
preprocess_target.py    letterbox onto 512x512 black (+ optional background
                         removal / Canny-edge canonicalization for real photos)
     |
     v
inference.py + resnet.py      ResNet18KeypointDetector -> 14 heatmaps
                               -> (x, y) + confidence per keypoint
     |
     v
geometry.py              translate + scale to make keypoints comparable
                          across animals of different size/framing
     |
     v
compatibility.py  <---  data/processed/geometric_database.json (built by build_DB.py)
     |    RMS distance -> softmax -> % ranking (whole species + per body segment)
     |
     +---> report.py + display_names.py   -> JSON + self-contained HTML report
     |
     +---> render/control_map.py + render/generate.py
               skeleton -> line-drawing control map (ControlNet)
               top donors' textures -> IP-Adapter reference
               report -> English text prompt
               -> Stable Diffusion -> final image

All of the above is orchestrated by pipeline.py, with ui.py (prompts,
progress, weights resolution) and bootstrap.py (dependency auto-install)
wrapping it into the single interactive command shown above.
```

| File | Role |
|---|---|
| `pipeline.py` | Orchestrates the whole pipeline end-to-end; the CLI entry point (interactive or scripted via flags). |
| `bootstrap.py` | Stdlib-only dependency checker: installs whatever's missing via `pip` before anything else runs. |
| `ui.py` | Interactive layer: image-path prompt, per-stage progress, final summary/comparison, and `resolve_weights_path` (auto-downloads/updates the trained weights from Hugging Face). |
| `preprocess_target.py` | Background removal (`rembg`, or a GrabCut fallback) and letterboxing to a 512x512 black canvas; `canonicalizza_bordi` blends in Canny edges to bring real photos closer to the training illustrations' style. |
| `resnet.py` | The `ResNet18KeypointDetector` architecture: an ImageNet-pretrained ResNet18 encoder + a `ConvTranspose2d` decoder outputting 14 raw heatmaps (no final activation - the loss is computed on the logits directly). |
| `inference.py` | Runs the trained detector on one image; turns each heatmap into an `(x, y)` coordinate (or `None` if confidence is too low) plus a confidence score. |
| `geometry.py` | Defines the 14-keypoint/5-segment schema and the scale/position normalization every other geometric computation builds on. |
| `compatibility.py` | The retrieval algorithm: normalized-keypoint RMS distance to every DB animal, converted to a softmax percentage ranking, both for the whole silhouette and per anatomical segment. |
| `build_DB.py` | One-off script that builds `geometric_database.json` from the 65 species' ground-truth annotations (`data/processed/Labels_X/`), not from network inference. |
| `report.py` | Turns a compatibility result into a JSON file and a self-contained HTML report (embedded thumbnails, no server needed). |
| `display_names.py` | Maps the dataset's Italian species/segment keys to English labels for anything a person reads (console, report, the render's text prompt) - the underlying data keys are untouched. |
| `dataset_paths.py` | Resolves the `path_scheletro`/`path_texture` fields stored in the database into an absolute path for the current machine, self-healing if the stored value is stale. |
| `render/control_map.py` | Draws the target skeleton as a white-line-on-black control image for ControlNet. |
| `render/generate.py` | The actual generative stage: Stable Diffusion 1.5 + ControlNet (on the control map) + IP-Adapter (on the best donors' textures, weighted by compatibility) + an auto-built English text prompt. |
| `train.py` | Trains `ResNet18KeypointDetector` from scratch on the 65-species dataset (this produced the shipped `best_keypoint_detector.pth`). |
| `finetune.py` | Continues training from the existing weights for a few more epochs on supplementary species (used to add the T-Rex/Brachiosaurus/Triceratops examples - see the report). |
| `heatmap.py` | `SkeletonKeypointDataset`: loads an image + its annotated keypoints, generates the Gaussian-heatmap training targets, and (when `augment=True`) applies flip/color/edge-blend augmentation. |
| `keypoints.py` | The interactive point-and-click tool used to hand-annotate the original 65-species dataset. |
| `check_heatmap.py`, `debug.py` | Diagnostics: save the generated training-target heatmaps, or a trained model's raw output heatmaps, as images to sanity-check visually. |
| `dataset_tools/fetch_wikimedia.py` | Searches Wikimedia Commons for openly-licensed candidate images and (only on request) downloads + logs one in `attributions.csv`. |

See `OsteoGen_V.2/documentazione_finale.txt` for more architectural detail
per module, and `OsteoGen_V.2/dataset_tools/SOURCES.md` for how to add
more species to the dataset.

## About V.0 and V.1

These are the two earlier architectures described in the report's ablation
study (Section 3) - a plain U-Net / PatchGAN / unconditional-ControlNet
baseline (V.0), then a prompt-conditioned ControlNet (V.1) - both
superseded by V.2's geometric-retrieval approach for the reasons discussed
in the report. Their code is kept, cleaned up, and portable (no
machine-specific paths). Like V.2, they resolve their own trained weights
automatically from the same Hugging Face repo (see `weights_hub.py`) - run
them the same way, either directly or via `python run.py --version 0` (or
`1`):

```bash
cd OsteoGen_V.0 && python main.py --image path/to/skeleton.jpg
cd OsteoGen_V.1 && python main.py --image path/to/skeleton.jpg
```

The actual result figures these produced are already included in
`report/` - regenerating them isn't necessary to read the report, only if
you want to reproduce them yourself.

## Training data and attribution

`OsteoGen_V.2/data/processed/` ships with the annotated 65-species dataset
(plus 3 supplementary extinct-species examples used only to fine-tune the
keypoint detector, see the report). Sources and licenses for anything added
beyond the original set are logged in
`OsteoGen_V.2/dataset_tools/attributions.csv`.
