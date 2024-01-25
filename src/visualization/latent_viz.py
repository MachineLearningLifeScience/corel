
import matplotlib.pyplot as plt
import numpy as np


def latent_space_fig(x, y, labels, title: str=""):
    fig, ax = plt.subplots()
    # get contour of latent space: # TODO: obtain grid by invoking black box function?
    # xi = np.linspace(min(x), max(x), 1000)
    # yi = np.linspace(min(y), max(y), 1000)
    # Xi, Yi = np.meshgrid(xi, yi)
    # Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    # cs = ax.contourf(Xi, Yi, Zi, cmap="PRGn")
    # ax.clabel(cs, inline=True, fontsize=16)
    # show maximal median and minimal values
    min_vals_idx = np.argsort(labels)[-50:]
    top_vals_idx = np.argsort(labels)[:50]
    median_idx = len(labels) // 2
    median_vals_idx = np.argsort(labels)[median_idx-25: median_idx+25]
    plt.scatter(x, y, c=labels, s=2., marker=".", alpha=0.3)
    plt.scatter(x[top_vals_idx], y[top_vals_idx], c=labels[top_vals_idx], s=25., marker="x", alpha=0.8, cmap="PRGn", label="top")
    plt.scatter(x[median_vals_idx], y[median_vals_idx], c=labels[median_vals_idx], s=25., marker="o", alpha=0.8, cmap="PRGn", label="median")
    plt.scatter(x[min_vals_idx], y[min_vals_idx], c=labels[min_vals_idx], s=25., alpha=0.8, marker="d", cmap="PRGn", label="min")
    plt.xlabel("d=1")
    plt.ylabel("d=2")
    plt.colorbar(label="pred. observation")
    plt.title(title)
    plt.legend()
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z2_gfp_space_samples.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z2_gfp_space_samples.pdf")
    plt.show()