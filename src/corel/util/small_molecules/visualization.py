"""
A set of utilities to draw molecules from SELFIE and SMILES strings,
using RDKit and cairosvg.
"""
from itertools import product
from pathlib import Path
from typing import Tuple

from PIL import Image

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import MolToImage

import cairosvg

import selfies as sf

from corel.weightings.vae.small_molecules.vae_selfies import VAESelfies
from corel.util.small_molecules.data import load_zinc_250k_alphabet


def selfie_to_png(
    selfie: str, save_path: Path, width: int = 200, height: int = 200, title: str = None
):
    """
    Save a molecule (specified as a selfie string) as png file.

    Taken and adapted from the following stack overflow answer:
    https://stackoverflow.com/a/73449342/3516175
    """
    if title is not None:
        # Expand the image a bit, to give room to the title
        # at the bottom
        height += int(height * 0.15)

    # Convert selfie to mol
    mol = Chem.MolFromSmiles(sf.decoder(selfie))
    assert mol is not None, f"Couldn't convert {selfie} to mol"

    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Add title to the image
    svg = drawer.GetDrawingText()
    if title is not None:
        svg = svg.replace(
            "</svg>",
            f'<text x="{width // 3}" y="{height - 20}" font-size="15" fill="black">{title}</text></svg>',
        )

    # Export to png
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(save_path))


def selfies_to_image(selfies: str, width: int = 200, height: int = 200) -> Image:
    """
    Convert a SELFIE string to a PIL image.
    """
    mol = Chem.MolFromSmiles(sf.decoder(selfies))
    assert mol is not None, f"Couldn't convert {selfies} to mol"

    return MolToImage(mol, size=(width, height))


def plot_grid(
    model: VAESelfies,
    x_lims: Tuple[float, float] = (-5, 5),
    y_lims: Tuple[float, float] = (-5, 5),
    n_rows: int = 10,
    n_cols: int = 10,
    individual_img_size: int = 200,
    sample: bool = False,
    ax: plt.Axes = None,
) -> np.ndarray:
    """
    A helper function which plots, as images, the levels in a
    fine grid in latent space, specified by the provided limits,
    number of rows and number of columns.

    The figure can be plotted in a given axis; if none is passed,
    a new figure is created.

    This function also returns the final image (which is the result
    of concatenating all the individual decoded images) as a numpy
    array.
    """
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    zs = np.array([[a, b] for a, b in product(z1, z2)])

    molecules_dist = model.decoder.layers(zs)
    if sample:
        molecules_as_ints = molecules_dist.sample()
    else:
        molecules_as_ints = tf.math.argmax(molecules_dist.logits, axis=-1)

    alphabet_string_to_index = load_zinc_250k_alphabet()
    alphabet_index_to_string = {v: k for k, v in alphabet_string_to_index.items()}

    molecules_as_selfies = []
    for molecule_ in molecules_as_ints.numpy():
        molecule = "".join([alphabet_index_to_string[i] for i in molecule_])
        molecules_as_selfies.append(molecule)

    images = [
        selfies_to_image(
            molecule, width=individual_img_size, height=individual_img_size
        )
        for molecule in molecules_as_selfies
    ]

    images = np.array([np.array(img) for img in images])
    img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

    positions = {
        (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
    }

    final_img = np.zeros(
        (n_cols * individual_img_size, n_rows * individual_img_size, 3)
    )
    for z, (i, j) in positions.items():
        final_img[
            i * individual_img_size : (i + 1) * individual_img_size,
            j * individual_img_size : (j + 1) * individual_img_size,
        ] = img_dict[z]

    final_img = final_img.astype(int)

    if ax is not None:
        ax.imshow(final_img, extent=[*x_lims, *y_lims])

    return final_img
