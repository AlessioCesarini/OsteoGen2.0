import io
import os
from pathlib import Path
from PIL import Image
from rembg import remove
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TARGET_SIZE = (512, 512)

def process_single_image(img_path: Path) -> Image.Image:
    # 1. Background removal.
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    output_bytes = remove(img_bytes)

    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # 2. Auto-crop to the alpha channel's bounding box.
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 3. Preserve aspect ratio with symmetric padding.
    w, h = img.size
    max_dim = max(w, h)

    squared_img = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    squared_img.paste(img, (paste_x, paste_y))

    # 4. Final resize to 512x512.
    resized_img = squared_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Convert to RGB on a black background for the network.
    final_img = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    final_img.paste(resized_img, (0, 0), mask=resized_img.split()[3])

    return final_img

def run_pipeline():
    for subfolder in ["input_x", "target_y"]:
        raw_folder = RAW_DIR / subfolder
        out_folder = PROCESSED_DIR / subfolder
        out_folder.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in raw_folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        print(f"Processing {len(files)} images in '{subfolder}'...")

        for file_path in tqdm(files):
            processed_img = process_single_image(file_path)
            out_path = out_folder / f"{file_path.stem}.png"
            processed_img.save(out_path, "PNG")

    # Validation check: X and Y must be in sync.
    x_files = {f.name for f in (PROCESSED_DIR / "input_x").glob("*.png")}
    y_files = {f.name for f in (PROCESSED_DIR / "target_y").glob("*.png")}

    print(f"\n--- Dataset validation ---")
    print(f"Files in input_x: {len(x_files)} | Files in target_y: {len(y_files)}")
    assert len(x_files) == len(y_files), "Error: the number of X and Y files does not match!"
    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
