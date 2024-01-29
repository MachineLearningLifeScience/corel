from distutils.command.install_egg_info import to_filename
from pathlib import Path
from tkinter import Variable

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.optimizers import Scipy
from gpflow.utilities import print_summary, to_default_float
from scipy.stats import multivariate_normal

from corel.kernel import Hellinger, WeightedHellinger
from corel.kernel.hellinger import get_mean_and_amplitude

gpflow.config.set_default_float(np.float64)
np.random.seed(123)

color_dict = {
    "unlabelled": "darkblue", 
    "labelled": "darkred", 
    "unseen": "grey"
    }

OUTPUT_PATH = Path(__file__).parent.parent.resolve() / "results" / "figures" / "one_dim"

# Setting up Toy data
n_unseen = 2
n_obs = 3
n_unlabelled = 4
n_total = n_unseen + n_obs + n_unlabelled
density_alpha = np.array([1e-7]*n_unseen+[1.]*n_obs+[0.2]*n_unlabelled)
p_x = np.random.dirichlet(density_alpha, 1).reshape(n_total, 1)
p_w = np.ones((n_total,1))/n_total # uniform weighting for toy case

y_values = np.random.normal(scale=.25, size=(n_total,1))
observations_annot = list(zip(y_values, ["unseen"]*n_unseen + ["labelled"]*n_obs + ["unlabelled"]*n_unlabelled))

# Visualize initial discrete observations
xx = np.arange(n_total)

plt.figure(figsize=(3.5, 3), dpi=300)
for i, (obs, annot) in  enumerate(observations_annot):
    plt.plot(i, obs, "X", ms=7., c=color_dict.get(annot), label=annot)
handles, labels = plt.gca().get_legend_handles_labels()
dict_label_handles = {label:handle for label, handle in zip(labels, handles)}
plt.xlabel("X input", fontsize=18)
plt.ylabel("observation", fontsize=18)
plt.ylim((-1.5, 1.5))
plt.title("Discrete", fontsize=21)
plt.legend(dict_label_handles.values(), dict_label_handles.keys(), loc="lower right")
plt.tight_layout()
plt.savefig(f"{str(OUTPUT_PATH)}/discrete_setup.png")
plt.savefig(f"{str(OUTPUT_PATH)}/discrete_setup.pdf")
plt.show()


# fitting base GP regression
k = Hellinger(L=1, AA=1)
gp_model = gpflow.models.GPR((p_x[n_unseen:n_obs+n_unseen], y_values[n_unseen:n_obs+n_unseen]), kernel=k)
gp_model.likelihood.variance = to_default_float(0.001)

xx = np.linspace(0+1e-7, 1-1e-7, 200).reshape(200, 1)
# xx_samples = np.linspace(0+1e-7, 1-1e7, 200).reshape(200, 1)
prior_mean, prior_var = gp_model.predict_f(xx, full_cov=False) # THIS HERE FAILS! NaN values introduced, why?
prior_samples = gp_model.predict_f_samples(xx, 100) # THIS HERE FAILS! NaN values introduced, why?

plt.figure(figsize=(3.5, 3), dpi=300)
plt.plot(xx, prior_mean, "C0", lw=2., label=r"prior $\mu$")
plt.plot(xx, prior_samples[:,:,0].numpy().T, "C0", lw=.25, alpha=.25)
plt.fill_between(xx[:,0], prior_mean[:,0]-1.96*np.sqrt(prior_var[:,0]), prior_mean[:,0]+1.96*np.sqrt(prior_var[:,0]), color="C0", alpha=0.25, label=r"2$\sigma$")
plt.fill_between(xx[:,0], prior_mean[:,0]-0.98*np.sqrt(prior_var[:,0]), prior_mean[:,0]+0.98*np.sqrt(prior_var[:,0]), color="C0", alpha=0.5, label=r"$\sigma$")
plt.plot(p_x[n_unseen:n_obs+n_unseen], y_values[n_unseen:n_obs+n_unseen], "X", ms=7., c=color_dict.get("labelled"), label="labelled")
plt.ylim((-1.5, 1.5))
plt.xlabel(r"$p_{\phi}(X)$", fontsize=18)
plt.ylabel("observations", fontsize=18)
plt.title("Continuous", fontsize=21)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{str(OUTPUT_PATH)}/discrete_to_continuous.png")
plt.savefig(f"{str(OUTPUT_PATH)}/discrete_to_continuous.pdf")
plt.show()


# fit GP posterior predictive
opt = Scipy()
opt.minimize(gp_model.training_loss, 
            gp_model.trainable_variables, 
            bounds=[(np.exp(-350), None), # constraint length_scale, prohibit values close to or equal zero
            (0, 0.0001)],)
post_mean, post_var = gp_model.predict_f(xx, full_cov=False) # THIS HERE FAILS! NaN values introduced, why?
post_samples = gp_model.predict_f_samples(xx, 100) # THIS HERE FAILS! NaN values introduced, why?


plt.figure(figsize=(4, 3), dpi=300)
plt.plot(xx, post_mean, "C0", lw=2.)
plt.plot(xx, prior_samples[:,:,0].numpy().T, "C0", lw=.25, alpha=.25)
plt.fill_between(xx[:,0], post_mean[:,0]-1.96*np.sqrt(post_var[:,0]), post_mean[:,0]+1.96*np.sqrt(post_var[:,0]), color="C0", alpha=0.25)
plt.plot(p_x[n_unseen:n_obs+n_unseen], y_values[n_unseen:n_obs+n_unseen], "x", ms=4., c=color_dict.get("labelled"), label="labelled")
plt.xlabel(r"$p(X)$ input", fontsize=18)
plt.ylabel(r"$\bar{f}(x)$", fontsize=18)
plt.title("optimized posterior fit", fontsize=21)
plt.legend()
plt.tight_layout()
plt.savefig(f"{str(OUTPUT_PATH)}/gp_posterior.png")
plt.savefig(f"{str(OUTPUT_PATH)}/gp_posterior.pdf")
plt.show()
