"""
ui.py

Interactive layer on top of pipeline.py: asks for the input image, shows
progress per pipeline stage, resolves the trained weights file, and
displays the results at the end. No pipeline logic lives here.

Degrades cleanly if something is missing:
- no 'rich' -> plain text status lines instead of a spinner.
- no display available -> the final comparison is saved to a file instead
  of being shown on screen.
"""
import os
import pathlib
import shutil
import sys
import time
import webbrowser
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
    _RICH = True
except ImportError:
    _console = None
    _RICH = False


BANNER = """\
OsteoGen 2.0 - from a fossil/skeleton to a reconstructed animal
-----------------------------------------------------------------
Pipeline stages:
  1) Detection of 14 anatomical keypoints (ResNet18)
  2) Biological retrieval: percentage compatibility against a
     geometric database of animals (a geometric similarity score,
     not a taxonomic classification)
  3) Readable report of the retrieval (JSON + HTML)
  4) Generative render (local diffusion, ControlNet + IP-Adapter)
     conditioned on the target skeleton and on the textures of the
     most compatible donors
-----------------------------------------------------------------\
"""


def show_banner():
    if _RICH:
        _console.print(BANNER, style="cyan")
    else:
        print(BANNER)


def ask_image_path():
    """Prompt for the input image path, with validation. Keeps asking
    until it gets a readable image file."""
    from PIL import Image
    while True:
        raw = input("\nPath to the fossil/skeleton image "
                    "(you can also drag the file into the terminal): ").strip()
        raw = raw.strip('"').strip("'")
        if not raw:
            continue
        path = os.path.expanduser(raw)
        if not os.path.isfile(path):
            print(f"  File not found: {path}")
            continue
        try:
            Image.open(path).verify()
        except Exception:
            print(f"  This does not look like a valid image: {path}")
            continue
        return path


def ask_yes_no(question, default=True):
    hint = "Y/n" if default else "y/N"
    answer = input(f"{question} [{hint}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


@contextmanager
def stage(description):
    """Context manager for one pipeline stage: a spinner (if 'rich' is
    available) or a plain status line, then the elapsed time. Exceptions
    are not swallowed - a failed stage still fails with a normal
    traceback, it does not just vanish inside a spinner."""
    start = time.time()
    if _RICH:
        with _console.status(f"[bold blue]{description}[/bold blue]", spinner="dots"):
            try:
                yield
            except Exception:
                _console.print(f"[bold red]x[/bold red] {description} -> FAILED")
                raise
        _console.print(f"[bold green]OK[/bold green] {description} "
                        f"[dim]({time.time() - start:.1f}s)[/dim]")
    else:
        print(f"-> {description}")
        try:
            yield
        except Exception:
            print(f"   FAILED after {time.time() - start:.1f}s")
            raise
        print(f"   done ({time.time() - start:.1f}s)")


def print_summary(report):
    from display_names import species_name
    ranking = report["ranking_specie"][:8]
    if _RICH:
        table = Table(title="Biological compatibility (retrieval)")
        table.add_column("Animal")
        table.add_column("Compatibility", justify="right")
        for r in ranking:
            table.add_row(species_name(r["animale"]), f"{r['compatibilita_pct']:.1f}%")
        _console.print(table)
    else:
        print("\nBiological compatibility (retrieval):")
        for r in ranking:
            print(f"  {species_name(r['animale']):20s} {r['compatibilita_pct']:5.1f}%")


def open_report_in_browser(html_path):
    if not html_path or not os.path.exists(html_path):
        return
    try:
        webbrowser.open(pathlib.Path(html_path).absolute().as_uri())
    except Exception as e:
        print(f"(could not open the report in a browser: {e})")


def show_final_comparison(panels, output_dir):
    """panels: list of (title, path). Missing paths are skipped. Shows a
    matplotlib figure with the images side by side; if no display is
    available it saves the figure to a file instead."""
    panels = [(title, path) for title, path in panels if path and os.path.exists(path)]
    if not panels:
        return None

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("(matplotlib not installed: skipping the visual comparison, "
              "the files are still in outputs/)")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, path) in zip(axes, panels):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("OsteoGen 2.0 - result")
    fig.tight_layout()

    if matplotlib.get_backend().lower() == "agg":
        # No graphical display available (e.g. an SSH session without X11).
        comparison_path = os.path.join(output_dir, "final_comparison.png")
        fig.savefig(comparison_path, dpi=150)
        print(f"No display available: comparison image saved to {comparison_path}")
        return comparison_path

    plt.show()
    return None


# Set this to a Hugging Face Hub model repo id (e.g. "username/osteogen-keypoint-detector")
# to have the trained weights download automatically on any machine with
# internet access. Left empty until the weights are hosted somewhere.
WEIGHTS_HF_REPO = os.environ.get("OSTEOGEN_WEIGHTS_REPO", "")


def resolve_weights_path(default_path):
    """Finds the trained keypoint-detector weights (.pth). Tries, in
    order: the default project path, a few common local folders, an
    optional Hugging Face Hub repo, and finally an interactive prompt -
    whatever is found gets copied into the default path so later runs
    never have to ask again."""
    if os.path.isfile(default_path):
        return default_path

    filename = os.path.basename(default_path)

    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.expanduser("~/Downloads"), filename),
        os.path.join(os.path.expanduser("~/Desktop"), filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            _save_as_default(candidate, default_path)
            return default_path

    if WEIGHTS_HF_REPO:
        try:
            from huggingface_hub import hf_hub_download
            print(f"[setup] Downloading trained weights from {WEIGHTS_HF_REPO} (one-time)...")
            downloaded = hf_hub_download(repo_id=WEIGHTS_HF_REPO, filename=filename)
            _save_as_default(downloaded, default_path)
            return default_path
        except Exception as e:
            print(f"[setup] Automatic download failed ({e}).")

    if not sys.stdin.isatty():
        raise FileNotFoundError(
            f"Trained weights not found at {default_path}, no local copy found, "
            f"and no download source configured (OSTEOGEN_WEIGHTS_REPO). "
            f"Pass --pesi <path> or run interactively to be prompted for it."
        )

    print(f"\nCould not find the trained keypoint-detector weights ({filename}).")
    while True:
        raw = input("Enter the path to the .pth file "
                    "(it will be copied into place, so this is only asked once): ").strip()
        path = os.path.expanduser(raw.strip('"').strip("'"))
        if os.path.isfile(path):
            _save_as_default(path, default_path)
            return default_path
        print(f"  File not found: {path}")


def _save_as_default(source, default_path):
    os.makedirs(os.path.dirname(default_path), exist_ok=True)
    if os.path.abspath(source) != os.path.abspath(default_path):
        shutil.copy2(source, default_path)
        print(f"[setup] Weights saved to {default_path} - future runs will find them automatically.")
