from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def latent_space_fig(x, y, labels, observations=None, title: str="", range: Tuple=None, ref_point: Tuple=None, target_point: Tuple=None):
    fig, ax = plt.subplots()
    # get contour of latent space: # TODO: obtain grid by invoking black box function?
    # xi = np.linspace(min(x), max(x), 1000)
    # yi = np.linspace(min(y), max(y), 1000)
    # Xi, Yi = np.meshgrid(xi, yi)
    # Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    # cs = ax.contourf(Xi, Yi, Zi, cmap="PRGn")
    # ax.clabel(cs, inline=True, fontsize=16)
    # show maximal median and minimal values
    sort_vals_idx = np.argsort(labels)
    max_labels = labels[sort_vals_idx][-50:]
    x_max = x[sort_vals_idx][-50:]
    y_max = y[sort_vals_idx][-50:]
    min_labels = labels[sort_vals_idx][:50]
    x_min = x[sort_vals_idx][:50]
    y_min = y[sort_vals_idx][:50]
    median_idx = len(labels) // 2
    median_labels = labels[sort_vals_idx][median_idx-25: median_idx+25]
    x_median = x[sort_vals_idx][median_idx-25: median_idx+25]
    y_median = y[sort_vals_idx][median_idx-25: median_idx+25]
    if observations is None:
        c_vals = labels
    else:
        c_vals = observations
    label_min = min(labels)
    label_max = max(labels)
    # TODO: mark min and max of labels and observations
    plt.scatter(x, y, c=c_vals, s=2., marker=".", alpha=0.3, vmin=label_min, vmax=label_max, cmap="PRGn")
    plt.scatter(x_max, y_max, c=max_labels, s=25., marker="^", vmin=label_min, vmax=label_max, alpha=0.8, cmap="PRGn", label="top")
    plt.scatter(x_median, y_median, c=median_labels, s=25., marker="o", vmin=label_min, vmax=label_max, alpha=0.8, cmap="PRGn", label="median")
    plt.scatter(x_min, y_min, c=min_labels, s=25., alpha=0.8, marker="v", vmin=label_min, vmax=label_max, cmap="PRGn", label="min")
    plt.colorbar(label="pred. observation")
    if ref_point is not None:
        plt.scatter(ref_point[0], ref_point[1], marker="X", c="black", s=55., label="start")
    if target_point is not None:
        plt.scatter(target_point[0], target_point[1], marker="X", c="darkred", s=55., label="target")
    # mark starting/reference value
    if range is not None:
        x_start = range[0]
        x_end = range[1]
        y = range[0]
        width = x_end - x_start
        rect = patches.Rectangle((x_start, y), width, width, linewidth=2.5, linestyle="dashed", edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    plt.xlabel(r"$z_1$", fontsize=18)
    plt.ylabel(r"$z_2$", fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend()
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z2_gfp_space_samples.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z2_gfp_space_samples.pdf")
    plt.tight_layout()
    plt.show()