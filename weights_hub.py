"""
weights_hub.py

Shared helper so each pipeline version (OsteoGen_V.0, V.1, V.2) can
auto-resolve its own trained weights from a single Hugging Face Hub repo:

    https://huggingface.co/markcst/osteogen-keypoint-detector

Layout in that repo:
    best_keypoint_detector.pth      <- Version 2 (repo root, unchanged;
                                        OsteoGen_V.2/ui.py resolves this
                                        one independently, see that file)
    v0/best_simple_UNET.pth
    v0/best_generator_PatchGAN.pth
    v0/best_controlnet.pth
    v1/controlnet_best_model/...    <- a whole diffusers-format folder

Same "always check Hugging Face first" rule as OsteoGen_V.2/ui.py's
resolve_weights_path (see that file for the full rationale): a plain "use
it if it already exists locally" check would keep using a stale local
copy forever once one exists, with no way to notice an updated version was
published. Falls back to whatever is already local if the Hub can't be
reached (offline, etc.); raises a clear error only if neither is available.
"""
import os
import shutil

HF_REPO = "markcst/osteogen-keypoint-detector"


def resolve_file(remote_path, local_path):
    """remote_path: path inside the HF repo, e.g. "v0/best_simple_UNET.pth".
    local_path: where the local copy should end up. Returns local_path."""
    import filecmp

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=HF_REPO, filename=remote_path)
        had_local_copy = os.path.isfile(local_path)
        if had_local_copy and filecmp.cmp(downloaded, local_path, shallow=False):
            return local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(downloaded, local_path)
        print(f"[setup] {'Updated' if had_local_copy else 'Downloaded'} "
              f"{os.path.basename(local_path)} from {HF_REPO}.")
        return local_path
    except Exception as e:
        if os.path.isfile(local_path):
            return local_path
        raise FileNotFoundError(
            f"Could not find or download {os.path.basename(local_path)} "
            f"(tried {HF_REPO}/{remote_path}): {e}"
        )


def resolve_folder(remote_subfolder, local_dir):
    """remote_subfolder: folder path inside the HF repo, e.g.
    "v1/controlnet_best_model". local_dir: where its contents should end
    up locally. Returns local_dir."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_dir = snapshot_download(repo_id=HF_REPO, allow_patterns=f"{remote_subfolder}/*")
        source = os.path.join(snapshot_dir, remote_subfolder)
        if os.path.abspath(source) != os.path.abspath(local_dir):
            parent = os.path.dirname(local_dir)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            shutil.copytree(source, local_dir)
            print(f"[setup] Weights folder ready at {local_dir} (from {HF_REPO}/{remote_subfolder}).")
        return local_dir
    except Exception as e:
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            return local_dir
        raise FileNotFoundError(
            f"Could not find or download the weights folder "
            f"(tried {HF_REPO}/{remote_subfolder}): {e}"
        )
