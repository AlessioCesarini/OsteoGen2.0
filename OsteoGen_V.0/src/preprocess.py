import os
from pathlib import Path
from PIL import Image
from rembg import remove
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TARGET_SIZE = (512, 512)

def process_single_image(img_path: Path) -> Image.Image:
    # 1. Caricamento e Scontorno (Y.2.1)
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    output_bytes = remove(img_bytes)
    
    # Conversione in PIL RGBA
    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    
    # 2. Auto-Cropping sulla Bounding Box del canale Alpha (Y.2.2)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    # 3. Preservazione Aspect Ratio con Padding Simmetrico (Y.2.2)
    w, h = img.size
    max_dim = max(w, h)
    
    # Sfondo neutro trasparente (o nero/bianco in RGB)
    squared_img = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    squared_img.paste(img, (paste_x, paste_y))
    
    # 4. Resize Finale a 512x512
    resized_img = squared_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    # Conversione in RGB con sfondo nero neutro per la rete
    final_img = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    final_img.paste(resized_img, (0, 0), mask=resized_img.split()[3])
    
    return final_img

def run_pipeline():
    for subfolder in ["input_x", "target_y"]:
        raw_folder = RAW_DIR / subfolder
        out_folder = PROCESSED_DIR / subfolder
        out_folder.mkdir(parents=True, exist_ok=True)
        
        files = sorted([f for f in raw_folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        print(f"Elaborazione {len(files)} immagini in '{subfolder}'...")
        
        for file_path in tqdm(files):
            processed_img = process_single_image(file_path)
            out_path = out_folder / f"{file_path.stem}.png"
            processed_img.save(out_path, "PNG")

    # Y.2.3 Validation Check
    x_files = {f.name for f in (PROCESSED_DIR / "input_x").glob("*.png")}
    y_files = {f.name for f in (PROCESSED_DIR / "target_y").glob("*.png")}
    
    print(f"\n--- Validazione Dataset (Y.2.3) ---")
    print(f"File in input_x: {len(x_files)} | File in target_y: {len(y_files)}")
    assert len(x_files) == len(y_files), "Errore: Il numero di file X e Y non coincide!"
    print("Pipeline di preprocessing completata con successo.")

if __name__ == "__main__":
    import io
    run_pipeline()