"""
bootstrap.py

Dependency check shared by Version 0 and Version 1's main.py (Version 2
keeps its own separate copy in OsteoGen_V.2/, untouched by this file).
Uses only the standard library (no third-party imports here), so it works
even on a bare Python install: if a required package is missing, it
installs it with pip for the interpreter currently running the script,
instead of the run dying on an import error partway through.
"""
import importlib.util
import subprocess
import sys


def ensure_packages(packages, label):
    """packages: list of (import_name, pip_name) tuples. Installs whatever
    is missing via pip. Exits with a clear, actionable message if that
    fails (no internet, no pip, a locked-down "externally managed"
    environment, ...) instead of leaving a raw traceback on screen."""
    missing = [pip_name for import_name, pip_name in packages
               if importlib.util.find_spec(import_name) is None]
    if not missing:
        return

    print(f"[setup] Installing missing dependencies for {label}: {', '.join(missing)}")
    print("[setup] One-time step; the first run may take a few minutes.")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    except subprocess.CalledProcessError as exc:
        print(f"""
[setup] Automatic installation failed (exit code {exc.returncode}).
Install the missing packages manually and re-run:

    {sys.executable} -m pip install {' '.join(missing)}

If pip refuses with an "externally-managed-environment" error, create a
virtual environment first:

    {sys.executable} -m venv .venv
    source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)
    {sys.executable} -m pip install {' '.join(missing)}
""")
        sys.exit(1)

    importlib.invalidate_caches()
    still_missing = [pip_name for import_name, pip_name in packages
                      if importlib.util.find_spec(import_name) is None]
    if still_missing:
        print(f"[setup] Still missing after installation: {', '.join(still_missing)}. "
              f"Install manually and re-run.")
        sys.exit(1)
