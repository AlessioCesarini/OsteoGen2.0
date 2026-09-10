"""
run.py

Single entry point to run any of the three OsteoGen pipeline versions,
each launched as its own process (so each version's own local imports and
working directory stay exactly as if you had `cd`-ed into it and run it
yourself - see each version's own main.py/pipeline.py for its own options).

Version 2 is the actual deliverable (see report/main.pdf for why); 0 and 1
are the earlier iterations described in the report's ablation study, kept
runnable for reference. Whichever version you pick, its own trained
weights are resolved and downloaded automatically (see weights_hub.py) -
nothing to set up by hand either way.

Usage:
    python run.py                      # asks interactively which version, then runs it
    python run.py --version 2          # skip the version prompt
    python run.py --version 0 --image path/to/skeleton.jpg   # extra args are forwarded as-is
"""
import argparse
import os
import subprocess
import sys

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VERSIONS = {
    "0": ("OsteoGen_V.0", "main.py",
          "Version 0 - U-Net / PatchGAN / unconditional ControlNet baseline (ablation study)"),
    "1": ("OsteoGen_V.1", "main.py",
          "Version 1 - prompt-conditioned ControlNet"),
    "2": ("OsteoGen_V.2", "pipeline.py",
          "Version 2 - geometric retrieval + generative render (the actual deliverable)"),
}
DEFAULT_VERSION = "2"


def ask_version():
    print("Which version do you want to run?")
    for key in ("2", "1", "0"):
        marker = " (default, recommended)" if key == DEFAULT_VERSION else ""
        print(f"  [{key}] {VERSIONS[key][2]}{marker}")
    choice = input(f"> [{DEFAULT_VERSION}] ").strip()
    return choice if choice in VERSIONS else DEFAULT_VERSION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", choices=list(VERSIONS), default=None,
                         help="Which pipeline version to run. If omitted, asked interactively "
                              "(or defaults to 2 when not running in a terminal).")
    args, rest = parser.parse_known_args()

    if args.version:
        version = args.version
    elif sys.stdin.isatty():
        version = ask_version()
    else:
        version = DEFAULT_VERSION

    version_dir, entry_file, label = VERSIONS[version]
    print(f"\nRunning {label}\n")

    entry_path = os.path.join(_BASE_DIR, version_dir, entry_file)
    result = subprocess.run([sys.executable, entry_path, *rest], cwd=os.path.join(_BASE_DIR, version_dir))
    sys.exit(result.returncode)
