import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float, set_trainable
from gpflow.kernels import Matern52
from gpflow.logdensities import multivariate_normal
from gpflow.optimizers import Scipy
from poli import objective_factory
from poli.objective_repository import gfp_cbas
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

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
                                      plotlatentspace_lvm_refpoint)
from visualization.latent_viz import latent_space_fig


def run_gfp_latent_visualization(seed: int=0, val_range=(-1.5,1.5), n_observation: int=10000, suffix: str=""):
    set_seeds(seed)
    caller_info = {
        BATCH_SIZE: None,
        SEED: seed,
        STARTING_N: n_observation,
        MODEL: None,
        ALGORITHM: "VISUALIZATION",
    }
    problem = "gfp_cbas_vae"
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

    L = problem_info.get_max_sequence_length()
    AA = len(problem_info.get_alphabet())

    aa_int_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
    int_aa_mapping = {aa_int_mapping[a]: a for a in aa_int_mapping.keys()}
    # TODO: subset for development FIXME
    all_sequences_int = transform_string_sequences_to_integer_arrays(all_sequences[:10000], L, aa_int_mapping)
    gfp_labels_path = Path(gfp_cbas.__file__).parent.resolve() / "assets" / "gfp_gt_evals.npy"
    # this is Nucleotide sequence for reference
    reference_sequence_fasta = Path(gfp_cbas.__file__).parent.resolve() / "assets" / "avGFP_reference_sequence.fa"
    reference_seq = all_sequences[2] # lookup reference_sequence_fasta in original csv
    labels = np.load(gfp_labels_path)
    # embedd sequences
    embedding = weighting.vae.encode(tf.one_hot(all_sequences_int, AA))
    reference_point = weighting.vae.encode(tf.one_hot(transform_string_sequences_to_integer_arrays(np.atleast_1d(reference_seq), L, aa_int_mapping), AA))
    title = "GFP VAE"
    ## 2D VAE space
    if embedding.shape[-1] > 2:
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(embedding)
        title += " PCA (n=2)"
    x = embedding[:10000,0]
    y = embedding[:10000,1]
    z = labels[:10000]
    latent_space_fig(x, y, z, title=title)

    # VISUALIZE 2D Latent space against reference point
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(bottom=0.2)
    img1 = plotlatentspace_lvm_refpoint(Matern52(), weighting, ax=ax[0,0], xmin=val_range[0], xmax=val_range[1], ref_point=reference_point)#, vmin=0.5)
    _ = plotlatentspace_lvm_refpoint(Hellinger(L=L, AA=AA), weighting, ax=ax[0,1], xmin=val_range[0], xmax=val_range[1], ref_point=reference_point)#, vmin=0.5)
    # weighted hellinger kernel with
    weighting_matrix = weighting.vae.decode(reference_point).reshape(L, AA) # transform to [L, AA]
    _ = plotlatentspace_lvm_refpoint(WeightedHellinger(w=weighting_matrix, L=L, AA=AA), weighting, ax=ax[0,2], xmin=val_range[0], xmax=val_range[1], ref_point=reference_point)#, vmin=0.5)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    fig.colorbar(img1, cax=cbar_ax, orientation="horizontal")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_Matern52_Hellinger_{problem}{suffix}.png")
    plt.savefig(f"/Users/rcml/corel/results/figures/kernel/latent_z_Matern52_Hellinger_{problem}{suffix}.pdf")
    plt.show()


if __name__ == '__main__':
    data_list = ["blat_fam", "rfp_fam",]
    model_list = ["vae", "hmm"]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--seed", help="Seed for distributions and random sampling of values.", type=int, default=0)
    #parser.add_argument("--samples", help="Number of samples to compute values.", type=int, default=2000)
    parser.add_argument("--ranges", help="Samples of latent space are in ranges (x1, x2)\in R", type=tuple, default=(-1.5, 1))
    parser.add_argument("-m", "--model", help="Latent model factory identifier", type=str, choices=model_list, default="vae")
    parser.add_argument("--reference", help="Reference point in LVM", type=tuple, default=(0,0))
    parser.add_argument("-d", "--dataset", help="Protein Family, for sample and LVM loading", type=str, choices=data_list, default=data_list[0])
    args = parser.parse_args()
    tf.config.run_functions_eagerly(run_eagerly=True)

    run_gfp_latent_visualization(seed=args.seed, val_range=args.ranges, suffix="_GFP_reference")