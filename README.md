# OsteoGen 2.0

From a fossil/skeleton to (1) a percentage biological compatibility report
against the animals in a geometric database, and (2) a generative render of
the reconstructed animal, blending the donors' textures weighted by
compatibility. Works even for extinct animals never photographed alive
(e.g. dinosaurs), because the retrieval is based only on skeleton shape,
never on a photo of the target.

See `OsteoGen_V.2/documentazione_finale.txt` for the detailed architecture
of each module, and `OsteoGen_V.2/dataset_tools/SOURCES.md` to expand the
dataset.

## Quick start

```bash
cd OsteoGen_V.2
python pipeline.py
```

Running it with no arguments starts an interactive run: it checks for
missing dependencies and installs them automatically, resolves the trained
keypoint-detector weights (asking for their path once if they cannot be
found or downloaded automatically), then asks for the image and walks
through every stage with a visible progress indicator, ending with the
report and the produced images opened automatically.

For scripted/non-interactive use, the original flags still work:

```bash
# Compatibility report only (no GPU needed beyond the keypoint inference,
# which also runs on CPU, just slower):
python pipeline.py --fossile input/my_fossil.jpg --no-render

# Report + generative render (a GPU is recommended for a reasonable runtime):
python pipeline.py --fossile input/my_fossil.jpg
```

Note: the trained keypoint-detector weights (`best_keypoint_detector.pth`)
are not stored in this repository (too large for git) and must be provided
separately - see the setup prompt on first run, or `OsteoGen_V.2/pipeline.py`'s
`--pesi` flag.

---

A Dinosaur Project
