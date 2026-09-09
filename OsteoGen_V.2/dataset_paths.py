"""
dataset_paths.py

geometric_database.json stores a path per image (path_scheletro,
path_texture). Those paths must be portable across machines, so
build_DB.py writes them relative to the project root, forward-slash
style, instead of whatever os.path.join happens to produce on the
machine that built the database.

resolve_dataset_path is the matching read-side half: it turns a stored
path back into an absolute one for the current machine, and falls back
to rebuilding the expected path from the known dataset layout if the
stored value does not resolve (e.g. an older database that still has an
absolute, machine-specific path baked in).
"""
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def to_relative(absolute_path):
    """Store as project-root-relative, forward-slash style, regardless
    of the OS that built the database."""
    return os.path.relpath(absolute_path, _BASE_DIR).replace(os.sep, "/")


def resolve_dataset_path(stored_path, categoria, filename):
    """categoria: "input_x" or "target_y". filename: the dataset key,
    e.g. "aquila.png"."""
    if stored_path:
        candidate = os.path.join(_BASE_DIR, *stored_path.replace("\\", "/").split("/"))
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(_BASE_DIR, "data", "processed", categoria, filename)
