import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import Parameter, default_float, set_trainable
from gpflow.kernels import Matern52
from poli import objective_factory
from poli.objective_repository import gfp_cbas
from sklearn.decomposition import PCA

import corel
from corel.kernel import Hellinger, WeightedHellinger
from corel.util.constants import (ALGORITHM, BATCH_SIZE, MODEL,
                                  PADDING_SYMBOL_INDEX, SEED, STARTING_N)
from corel.util.util import (get_amino_acid_integer_mapping_from_info,
                             set_seeds,
                             transform_string_sequences_to_integer_arrays)
from corel.weightings.hmm.hmm_factory import HMMFactory
from corel.weightings.vae.base.models import VAE
from corel.weightings.vae.base.train_vae import _preprocess
from corel.weightings.vae.base.vae_factory import VAEFactory
from corel.weightings.vae.cbas import CBASVAEWeightingFactory
from visualization.kernel_viz import (plotkernelsample_in_Ps,
                                      plotkernelsample_in_R,
                                      plotlatentspace_lvm,
                                      plotlatentspace_lvm_refpoint,
                                      plotlatentspace_lvm_refpoint_contour)
from visualization.latent_viz import latent_space_fig

STEPSIZE=0.01


def get_maternK_values(weighting: object, ref_point: Tuple, xmin, xmax, stepsize=STEPSIZE) -> np.ndarray:
    xxyy = tf.convert_to_tensor(np.mgrid[xmin:xmax:stepsize, xmin:xmax:stepsize].reshape(2, -1).T)
    if not isinstance(ref_point, tf.Tensor): # convert tuple to tensor
        ref_point = tf.cast(tf.convert_to_tensor(ref_point)[None,:], tf.float64)
    k = Matern52()
    k_values = k(xxyy, ref_point)
    k_values = tf.squeeze(k_values).numpy()
    return k_values


def get_hellK_values(weighting: object, ref_point: Tuple, L, AA, xmin, xmax, stepsize=STEPSIZE) -> np.ndarray:
    xxyy = tf.convert_to_tensor(np.mgrid[xmin:xmax:stepsize, xmin:xmax:stepsize].reshape(2, -1).T)
    if not isinstance(ref_point, tf.Tensor): # convert tuple to tensor
        ref_point = tf.cast(tf.convert_to_tensor(ref_point)[None,:], tf.float64)
    decoded_latent = weighting.vae.decode(xxyy) # TODO: this is for the 2D case
    ps = tf.reshape(tf.squeeze(decoded_latent), shape=(len(decoded_latent), decoded_latent.shape[-1]*decoded_latent.shape[-2]))
    decoded_reference = weighting.vae.decode(ref_point.reshape(1, 2)) 
    ps_ref = tf.reshape(tf.squeeze(decoded_reference), shape=(1, decoded_reference.shape[-1]*decoded_reference.shape[-2]))
    k = Hellinger(L=L, AA=AA)
    k_values = k(ps, ps_ref)
    k_values = tf.squeeze(k_values).numpy()
    return k_values


def get_wHK_values(weighting: object, ref_point: Tuple, L, AA, xmin, xmax, stepsize=STEPSIZE, one_hot=False) -> np.ndarray:
    xxyy = tf.convert_to_tensor(np.mgrid[xmin:xmax:stepsize, xmin:xmax:stepsize].reshape(2, -1).T)
    if not isinstance(ref_point, tf.Tensor): # convert tuple to tensor
        ref_point = tf.cast(tf.convert_to_tensor(ref_point)[None,:], tf.float64)
    reference_point = tf.reshape(ref_point, shape=(1, 2))
    weighting_matrix = weighting.vae.decode(reference_point).reshape(L, AA) # transform to [L, AA] for kernel weighting
    decoded_latent = weighting.vae.decode(xxyy)
    decoded_seq = tf.one_hot(tf.argmax(decoded_latent, axis=-1), AA).reshape(1, decoded_latent.shape[0], L*AA)
    decoded_reference = weighting.vae.decode(ref_point.reshape(1, 2))
    decoded_ref_seq = tf.one_hot(tf.argmax(decoded_reference, axis=-1), AA).reshape(1, ref_point.shape[0], L*AA)
    k = WeightedHellinger(w=weighting_matrix, L=L, AA=AA)
    if one_hot:
        k_values = k(decoded_seq, decoded_ref_seq) # NOTE: weighting distribution in weighting vector, use OneHot for X
    else:
        k_values = k(decoded_latent.reshape(1, decoded_latent.shape[0], L*AA), decoded_reference.reshape(1, decoded_reference.shape[0], L*AA))
    k_values = tf.squeeze(k_values).numpy()
    return k_values


def run_gfp_latent_visualization(seed: int=0, val_range=(.2, .35), n_observation: int=54025, suffix: str="", reference_point=None):
    set_seeds(seed)
    caller_info = {
        BATCH_SIZE: None,
        SEED: seed,
        STARTING_N: n_observation,
        MODEL: None,
        ALGORITHM: "VISUALIZATION",
    }
    problem = "gfp_cbas_gp"
    problem_info, _f, _x0, _y0, _ = objective_factory.create(
        name=problem,
        seed=seed,
        caller_info=caller_info,
        batch_size=None, # determines return shape of y0
        observer=None,
        force_register=True,
        parallelize=False,
        problem_type=problem.split("_")[-1]
    )
    # explicitly load the 2D VAE for the problem
    vae_model_path = Path(corel.__file__).parent.parent.parent.resolve() / "results" / "models"
    weighting = CBASVAEWeightingFactory().create(problem_info=problem_info, model_path=vae_model_path, latent_dim=2)
    all_sequences = weighting.get_training_data()
    all_measurements = weighting.get_training_labels()

    L = problem_info.get_max_sequence_length()
    AA = len(problem_info.get_alphabet())

    aa_int_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
    int_aa_mapping = {aa_int_mapping[a]: a for a in aa_int_mapping.keys()}
    # TODO: subset for development FIXME
    all_sequences_int = transform_string_sequences_to_integer_arrays(all_sequences[:n_observation], L, aa_int_mapping)
    # NOTE: the persisted gfp_labels might be misaligned!
    gfp_labels_path = Path(gfp_cbas.__file__).parent.resolve() / "assets" / "gfp_gt_evals.npy"
    labels = np.load(gfp_labels_path) #
    # this is Nucleotide sequence for reference
    reference_sequence_fasta = Path(gfp_cbas.__file__).parent.resolve() / "assets" / "avGFP_reference_sequence.fa"
    reference_seq = all_sequences[2] # lookup reference_sequence_fasta in original csv
    embedding = weighting.vae.encode(tf.one_hot(all_sequences_int, AA))
    if reference_point is None: # default: use ref. sequence encoding
        reference_point = weighting.vae.encode(tf.one_hot(transform_string_sequences_to_integer_arrays(np.atleast_1d(reference_seq), L, aa_int_mapping), AA))
    title = "GFP VAE "
    title += " (D. H. Brookes et al. 2019)"
    ## 2D VAE space
    if embedding.shape[-1] > 2:
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(embedding)
        title += " PCA (n=2)"
    x = embedding[:n_observation,0]
    y = embedding[:n_observation,1]
    z = labels[:n_observation] # GP labels
    max_point = x[z.argmax(axis=0)], y[z.argmax(axis=0)]
    # latent_space_fig(x, y, z, observations=all_measurements[:n_observation], ref_point=reference_point[0], target_point=max_point, title=title, range=val_range)

    matern_vals = get_maternK_values(weighting, ref_point=reference_point, xmin=val_range[0], xmax=val_range[1])
    hk_vals = get_hellK_values(weighting, ref_point=reference_point, xmin=val_range[0], xmax=val_range[1], L=L, AA=AA)
    whk_vals = get_wHK_values(weighting, ref_point=reference_point, xmin=val_range[0], xmax=val_range[1], L=L, AA=AA)

    # # VISUALIZE 2D Latent space against reference point
    # fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True, squeeze=False, subplot_kw={'aspect': 'equal'})
    # plt.subplots_adjust(bottom=0.2)
    # img1 = plotlatentspace_lvm_refpoint(matern_vals, ax=ax[0,0], xmin=val_range[0], suffix="\nMatern")
    # _ = plotlatentspace_lvm_refpoint(hk_vals, ax=ax[0,1], xmin=val_range[0], xmax=val_range[1], suffix="\nHellinger")
    # _ = plotlatentspace_lvm_refpoint(whk_vals, ax=ax[0,2], xmin=val_range[0], xmax=val_range[1], suffix="\nHellinger weighted")
    # cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    # fig.colorbar(img1, cax=cbar_ax, orientation="horizontal")
    # plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_Matern52_Hellinger_{problem}{suffix}.png")
    # plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_Matern52_Hellinger_{problem}{suffix}.pdf")
    # plt.tight_layout()
    # plt.show()

    # VISUALIZE 2D Latent space against CONTOUR reference point
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True, squeeze=False, subplot_kw={'aspect': 'equal'})

    img1 = plotlatentspace_lvm_refpoint_contour(matern_vals, ax=ax[0,0], suffix="\nMatern")
    _ = plotlatentspace_lvm_refpoint_contour(hk_vals, ax=ax[0,1], xmin=val_range[0], xmax=val_range[1], suffix="\nHellinger")
    _ = plotlatentspace_lvm_refpoint_contour(whk_vals, ax=ax[0,2], xmin=val_range[0], xmax=val_range[1], ref_point=reference_point, suffix="\nHellinger weighted")
    plt.subplots_adjust(bottom=0.26, top=0.9, wspace=0., hspace=0., right=0.98, left=0.09)
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/CONTOUR_latent_z_Matern52_Hellinger_{problem}{suffix}.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/CONTOUR_latent_z_Matern52_Hellinger_{problem}{suffix}.pdf")
    plt.show()


if __name__ == '__main__':
    model_list = ["vae", "hmm"]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--seed", help="Seed for distributions and random sampling of values.", type=int, default=0)
    parser.add_argument("--ranges", help="Samples of latent space are in ranges (x1, x2)\in R", type=tuple, default=(-.3, .3)) # set 
    parser.add_argument("-m", "--model", help="Latent model factory identifier", type=str, choices=model_list, default="vae") # TODO: add HMM?
    parser.add_argument("--reference", help="Reference point in LVM", type=tuple, default=(0,0))
    args = parser.parse_args()
    tf.config.run_functions_eagerly(run_eagerly=True)

    # WT is reference point
    run_gfp_latent_visualization(seed=args.seed, val_range=args.ranges, suffix="_GFP_reference")
    # zero is reference point
    run_gfp_latent_visualization(seed=args.seed, val_range=args.ranges, suffix="_zeroprior_reference", reference_point=np.array([0.,0.]))
    # for ref_point in [(0., 0.), (.3, .2), (-.5, 0.), (.5, 0), (.3, .2), (0., -.5), (0, .5)]:
    #     run_gfp_latent_visualization(seed=args.seed, val_range=args.ranges, suffix=f"_GFP_{str(ref_point[0])}_{str(ref_point[1])}", reference_point=ref_point)