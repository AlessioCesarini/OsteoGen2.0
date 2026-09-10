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
OsteoGen_V.1/   <- earlier iteration (prompt-conditioned ControlNet), kept as
OsteoGen_V.0/   <- historical/reference material - both are described and
                   compared against V.2 in the report, not meant to be re-run
                   (see "About V.0 and V.1" below).
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

## About V.0 and V.1

These are the two earlier architectures described in the report's ablation
study (Section 3) - a plain U-Net / PatchGAN / unconditional-ControlNet
baseline (V.0), then a prompt-conditioned ControlNet (V.1) - both
superseded by V.2's geometric-retrieval approach for the reasons discussed
in the report. Their code is kept, cleaned up, and portable (no
machine-specific paths), but **running them yourself requires their trained
weights, which are not included in this repository** (too large for git) -
see the report for why V.2 was adopted instead, and for the actual result
figures these produced (already included in `report/`, no need to
regenerate them). If those weights get hosted somewhere, this section will
be updated with the exact command.

## Training data and attribution

`OsteoGen_V.2/data/processed/` ships with the annotated 65-species dataset
(plus 3 supplementary extinct-species examples used only to fine-tune the
keypoint detector, see the report). Sources and licenses for anything added
beyond the original set are logged in
`OsteoGen_V.2/dataset_tools/attributions.csv`.

---

A Dinosaur Project
