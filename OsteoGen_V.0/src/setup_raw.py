import shutil
from pathlib import Path

SRC_DIR = Path("imgs_to_work")
RAW_X = Path("data/raw/input_x")
RAW_Y = Path("data/raw/target_y")

RAW_X.mkdir(parents=True, exist_ok=True)
RAW_Y.mkdir(parents=True, exist_ok=True)

for filepath in SRC_DIR.iterdir():
    if not filepath.is_file():
        continue

    # Clean trailing spaces in file stem
    stem = filepath.stem.strip()

    if "_ossa" in stem:
        animal_id = stem.replace("_ossa", "").strip()
        shutil.copy(filepath, RAW_X / f"{animal_id}.png")
    elif "_output" in stem:
        animal_id = stem.replace("_output", "").strip()
        shutil.copy(filepath, RAW_Y / f"{animal_id}.png")

x_count = len(list(RAW_X.glob("*.png")))
y_count = len(list(RAW_Y.glob("*.png")))

print(f"Dataset organizzato: {x_count} scheletri in X | {y_count} rendering in Y")