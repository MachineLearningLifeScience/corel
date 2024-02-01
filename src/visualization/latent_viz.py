from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def latent_space_fig(x, y, labels, observations=None, title: str="", range: Tuple=None, ref_point: Tuple=None, target_point: Tuple=None, oracle_n=10):
    fig, ax = plt.subplots(figsize=(6,6))
    # get contour of latent space: # TODO: obtain grid by invoking black box function?
    # xi = np.linspace(min(x), max(x), 1000)
    # yi = np.linspace(min(y), max(y), 1000)
    # Xi, Yi = np.meshgrid(xi, yi)
    # Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    # cs = ax.contourf(Xi, Yi, Zi, cmap="PRGn")
    # ax.clabel(cs, inline=True, fontsize=16)
    # show maximal median and minimal values
    sort_vals_idx = np.argsort(labels)
    max_labels = labels[sort_vals_idx][-oracle_n:]
    x_max = x[sort_vals_idx][-oracle_n:]
    y_max = y[sort_vals_idx][-oracle_n:]
    min_labels = labels[sort_vals_idx][:oracle_n]
    x_min = x[sort_vals_idx][:oracle_n]
    y_min = y[sort_vals_idx][:oracle_n]
    median_idx = len(labels) // 2
    median_labels = labels[sort_vals_idx][median_idx-int(oracle_n/2): median_idx+int(oracle_n/2)]
    x_median = x[sort_vals_idx][median_idx-int(oracle_n/2): median_idx+int(oracle_n/2)]
    y_median = y[sort_vals_idx][median_idx-int(oracle_n/2): median_idx+int(oracle_n/2)]
    if observations is None:
        c_vals = labels
    else:
        c_vals = observations
    label_min = min(labels)
    label_max = max(labels)
    plt.scatter(x, y, c=c_vals, s=.25, marker=".", alpha=0.125, vmin=label_min, vmax=label_max, cmap="PRGn")
    plt.scatter(x_max, y_max, c=max_labels, s=30., marker="^", vmin=label_min, vmax=label_max, alpha=1., cmap="PRGn", label="top")
    plt.scatter(x_median, y_median, c=median_labels, s=30., marker="o", vmin=label_min, vmax=label_max, alpha=1., cmap="PRGn", label="median")
    plt.scatter(x_min, y_min, c=min_labels, s=30., alpha=1., marker="v", vmin=label_min, vmax=label_max, cmap="PRGn", label="min")
    plt.colorbar(label="pred. observation")
    if ref_point is not None:
        plt.scatter(ref_point[0], ref_point[1], marker="X", c="black", s=60., label="start")
    if target_point is not None:
        plt.scatter(target_point[0], target_point[1], marker="X", c="darkred", s=60., label="target")
    # mark starting/reference value
    if range is not None:
        x_start = range[0]
        x_end = range[1]
        y = range[0]
        width = x_end - x_start
        rect = patches.Rectangle((x_start, y), width, width, linewidth=2.5, linestyle="dashed", edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    plt.ylim((-1, 0.7))
    plt.xlim((-1, 0.7))
    plt.xlabel(r"$z_1$", fontsize=18)
    plt.ylabel(r"$z_2$", fontsize=18)
    ax.set_aspect("equal")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    # plt.title(title, fontsize=20)
    plt.legend()
    plt.subplots_adjust(top=0.99, right=0.989, left=0.15, bottom=0.02)
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z2_gfp_space_samples.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z2_gfp_space_samples.pdf")
    # plt.tight_layout()
    plt.show()