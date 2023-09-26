import argparse
import os
import numpy as np
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple

from gpflow import default_float, set_trainable
from gpflow.logdensities import multivariate_normal
from gpflow.optimizers import Scipy
from gpflow.kernels import Matern52

from corel.weightings.vae.base.models import VAE
from corel.weightings.vae.base.train_vae import _preprocess
from corel.weightings.vae.base.vae_factory import VAEFactory
from corel.weightings.hmm.hmm_factory import HMMFactory
from corel.kernel import Hellinger
from corel.kernel import WeightedHellinger
from visualization.kernel_viz import plotkernelsample_in_R
from visualization.kernel_viz import plotkernelsample_in_Ps
from visualization.kernel_viz import plotlatentspace_lvm
from visualization.kernel_viz import plotlatentspace_lvm_refpoint

# TODO: this is bad practice, ADD experiments module to setup cfg
import sys
if "/Users/rcml/corel/" not in sys.path:
    sys.path.append("/Users/rcml/corel/")
from experiments.assets.data.rfp_fam import rfp_train_dataset, rfp_test_dataset
from experiments.assets.data.blat_fam import blat_train_dataset, blat_test_dataset


def run_latent_space_visualization(
    seed: int, model_factory: str, data_key: str, train_dataset, test_dataset,
    val_range: tuple=(-1,-1), ref_point: tuple=(0,0)
    ) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    x1, x2 = val_range

    x0 = next(iter(train_dataset))[0][0]

    ## LOAD LVM MODEL
    p_model = model_factory.create(None)

    # VISUALIZE 2D Latent space all points against each other...
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True, squeeze=False)
    plotlatentspace_lvm(Matern52(), p_model, ax=ax[0,0], xmin=x1, xmax=x2)
    L, n_cat = p_model.encoder.input_dims
    plotlatentspace_lvm(Hellinger(L=L, AA=n_cat), p_model, ax=ax[0,1], xmin=x1, xmax=x2)
    # TODO: add weighted Hellinger Kernel
    # weighted hellinger kernel with p0
    p0 = p_model.p(tf.zeros((1,2)))
    # plotlatentspace_lvm(WeightedHellinger(p=p0, L=L, AA=n_cat), p_model, ax=ax[0,2], xmin=x1, xmax=x2)
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_all_Matern52_Hellinger_{data_key}.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_all_Matern52_Hellinger_{data_key}.pdf")
    plt.show()

    # VISUALIZE 2D Latent space against reference point
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True, squeeze=False)
    plotlatentspace_lvm_refpoint(Matern52(), p_model, ax=ax[0,0], xmin=x1, xmax=x2, ref_point=ref_point)
    L, n_cat = p_model.encoder.input_dims
    plotlatentspace_lvm_refpoint(Hellinger(L=L, AA=n_cat), p_model, ax=ax[0,1], xmin=x1, xmax=x2, ref_point=ref_point)
    # weighted hellinger kernel with p0
    # # TODO: implement weighted HK
    # p0 = p_model.p(tf.zeros((1,2)))
    # plotlatentspace_lvm_refpoint(WeightedHellinger(z=p0, L=L, AA=n_cat), p_model, ax=ax[0,2], xmin=x1, xmax=x2)
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_{'_'.join([str(x) for x in ref_point])}_Matern52_Hellinger_{data_key}.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_{'_'.join([str(x) for x in ref_point])}_Matern52_Hellinger_{data_key}.pdf")
    plt.show()

    # VISUALIZE 2D Latent space against sequences from the training set
    N = 5
    L, n_cat = p_model.encoder.input_dims
    fig, ax = plt.subplots(2, N, figsize=(N*6, N*2), sharex=True, sharey=True, squeeze=False)
    #first_seq_batch = next(iter(train_dataset))[0]
    first_seq_batch = next(iter(test_dataset))[0]
    for i, k in enumerate([Matern52(), Hellinger(L=L, AA=n_cat)]):
        points = []
        for j in range(N):
             # TODO: currently sample from training BUT we may want the observed RFP sequences here
            seq_i = first_seq_batch[j] # TODO: they are all very close to the center! take elements on the edge of the latent shapes?
            z_x = p_model.encoder.layers(seq_i[None,:]).mean() # NOTE: one could also take expected value of limited sample
            z_x = tf.cast(z_x, tf.float64)
            points.append(list(z_x.numpy()))
            plotlatentspace_lvm_refpoint(k, p_model, ax=ax[i,j], xmin=x1, xmax=x2, ref_point=z_x)
            plotlatentspace_lvm_refpoint(k, p_model, ax=ax[i,j], xmin=x1, xmax=x2, ref_point=z_x)
            # weighted hellinger kernel with p0
            # # TODO: implement weighted HK
            # p0 = p_model.p(tf.zeros((1,2)))
            # plotlatentspace_lvm_refpoint(WeightedHellinger(z=p0, L=L, AA=n_cat), p_model, ax=ax[0,2], xmin=x1, xmax=x2, ref_point=x)
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_samples_{N}_Matern52_Hellinger_{data_key}.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_samples_{N}_Matern52_Hellinger_{data_key}.pdf")
    plt.show()

    # VISUALIZE 2D Latent space against other points in latent space
    L, n_cat = p_model.encoder.input_dims
    kernel_list = [Matern52(), Hellinger(L=L, AA=n_cat)]
    points = np.array([[-2, -2], [0, -2], [-2, 0], [2, -2], [-2, 2], [2, 2]])
    fig, ax = plt.subplots(len(kernel_list), points.shape[0], figsize=(points.shape[0]*6, points.shape[0]*2), sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(kernel_list):
        for j, point in enumerate(points):
             # TODO: currently sample from training BUT we may want the observed RFP sequences here
            plotlatentspace_lvm_refpoint(k, p_model, ax=ax[i,j], xmin=x1, xmax=x2, ref_point=tf.convert_to_tensor(point[None,:], dtype=tf.float64))
            plotlatentspace_lvm_refpoint(k, p_model, ax=ax[i,j], xmin=x1, xmax=x2, ref_point=tf.convert_to_tensor(point[None,:], dtype=tf.float64))
            # weighted hellinger kernel with p0
            # # TODO: implement weighted HK
            # p0 = p_model.p(tf.zeros((1,2)))
            # plotlatentspace_lvm_refpoint(WeightedHellinger(z=p0, L=L, AA=n_cat), p_model, ax=ax[0,2], xmin=x1, xmax=x2, ref_point=x)
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_{str(points.shape[0])}_latent_points_Matern52_Hellinger.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_{str(points.shape[0])}_latent_points_Matern52_Hellinger.pdf")
    plt.show()


def run_latent_samples_visualization(model_factory: object, data_key: str, train_dataset, test_dataset):
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    alpha = 0.25 if "blat" not in data_key else 0.15
    ## LOAD LVM MODEL
    p_model = model_factory.create(None)
    z_coords = []
    for _batch, _ in train_dataset: # TODO: compute for all data
        z_dist_batch = p_model.encoder.layers(_batch)
        mean_coords = z_dist_batch.mean().numpy()
        z_coords.append(mean_coords)
    for _batch, _ in test_dataset:
        z_dist_batch = p_model.encoder.layers(_batch)
        mean_coords = z_dist_batch.mean().numpy()
        z_coords.append(mean_coords)
    coords = tf.concat(z_coords, 0).numpy()
    ax.scatter(coords[:, 0], coords[:, 1], alpha=alpha, s=2)
    # ax.set_xlim((-5,2))
    # ax.set_ylim((-6,3))
    plt.title(f"Latent Representation \n{data_key}")
    plt.savefig(f"/Users/rcml/corel/results/figures/lvm/latent_z_{data_key}_latent_mean.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/lvm/latent_z_{data_key}_latent_mean.pdf")
    plt.show()


def load_dataset(data_key: str) -> Tuple[tf.Tensor]:
    ## LOAD DATASET
    if data_key.lower() == "rfp_fam":
        train_dataset = rfp_train_dataset.map(_preprocess).batch(128).prefetch(tf.data.AUTOTUNE).shuffle(42)
        test_dataset = rfp_test_dataset.map(_preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    elif data_key.lower() == "blat_fam":
        train_dataset = blat_train_dataset.map(_preprocess).batch(128).prefetch(tf.data.AUTOTUNE).shuffle(42)
        test_dataset = blat_test_dataset.map(_preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    else:
        raise ValueError("Specify dataset from list of available datasets!")
    return train_dataset, test_dataset


if __name__ == '__main__':
    data_list = ["blat_fam", "rfp_fam",]
    model_list = ["vae", "hmm"]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--seed", help="Seed for distributions and random sampling of values.", type=int, default=0)
    #parser.add_argument("--samples", help="Number of samples to compute values.", type=int, default=2000)
    parser.add_argument("--ranges", help="Samples of latent space are in ranges (x1, x2)\in R", type=tuple, default=(-4, 4))
    parser.add_argument("-m", "--model", help="Latent model factory identifier", type=str, choices=model_list, default="vae")
    parser.add_argument("--reference", help="Reference point in LVM", type=tuple, default=(0,0))
    parser.add_argument("-d", "--dataset", help="Protein Family, for sample and LVM loading", type=str, choices=data_list, default=data_list[0])
    args = parser.parse_args()
    tf.config.run_functions_eagerly(run_eagerly=True)

    X_train, X_test = load_dataset(args.dataset)

    model_factories = {
        "vae": VAEFactory(f"/Users/rcml/corel/results/models/vae_z_2_{args.dataset}.ckpt", problem_name=args.dataset),
        "hmm": HMMFactory("./assets/hmms/rfp.hmm", None)
    }

    run_latent_samples_visualization(model_factory=model_factories.get(args.model), data_key=args.dataset,
                                    train_dataset=X_train, test_dataset=X_test)
    run_latent_space_visualization(seed=args.seed, data_key=args.dataset,
        train_dataset=X_train, test_dataset=X_test,
        val_range=args.ranges, model_factory=model_factories.get(args.model), 
        ref_point=args.reference)
