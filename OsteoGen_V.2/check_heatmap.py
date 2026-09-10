import os
import matplotlib.pyplot as plt
from heatmap import SkeletonKeypointDataset  # make sure the previous file is named heatmap.py

def salva_controlli_visivi():
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(_base_dir, "data", "processed", "input_x")
    json_dir = os.path.join(_base_dir, "data", "processed", "Labels_X")
    out_dir = os.path.join(_base_dir, "data", "processed", "Heatmap_Checks")

    os.makedirs(out_dir, exist_ok=True)

    dataset = SkeletonKeypointDataset(img_dir=img_dir, json_dir=json_dir)

    print(f"Saving {len(dataset)} check images...")

    for i in range(len(dataset)):
        img_tensor, heatmap_tensor = dataset[i]

        # Denormalize the image for display.
        img_vis = img_tensor.numpy().transpose(1, 2, 0)
        img_vis = (img_vis * 0.5) + 0.5

        # Sum the 14 heatmaps to see them all together.
        heatmap_vis = heatmap_tensor.numpy().sum(axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_vis)
        ax[0].set_title("Original Input")
        ax[0].axis("off")

        ax[1].imshow(img_vis)
        ax[1].imshow(heatmap_vis, cmap="jet", alpha=0.5)
        ax[1].set_title("Target: 14 Gaussian Heatmaps")
        ax[1].axis("off")

        nome_file = dataset.img_names[i]
        plt.savefig(os.path.join(out_dir, nome_file), bbox_inches='tight')

        plt.close(fig)

        print(f"Saved check for: {nome_file}")

    print(f"\nDone! Check the folder: {out_dir}")

if __name__ == "__main__":
    salva_controlli_visivi()
