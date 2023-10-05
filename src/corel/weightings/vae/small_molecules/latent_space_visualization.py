"""
Decodes a grid in latent space, and visualizes it
using the selfie_to_png function in the utilities.
"""
import matplotlib.pyplot as plt

from corel.util.small_molecules.visualization import plot_grid
from corel.weightings.vae.small_molecules.load_vae import load_vae

if __name__ == "__main__":
    vae = load_vae()

    img = plot_grid(vae, n_rows=10, n_cols=10, individual_img_size=200)
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
