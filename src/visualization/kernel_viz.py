import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.kernels import Kernel
from matplotlib.ticker import FixedFormatter, FixedLocator


def plotkernelsample_in_R(k: Kernel, ax, xmin: float=-3., xmax: float=3.):
    """
    Reference: https://gpflow.readthedocs.io/en/v1.5.1-docs/notebooks/advanced/kernels.html [04/09/2023]
    """
    xx = np.linspace(xmin, xmax, 100)[:,None]
    K = k(xx)
    ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)
    ax.set_xlim((xmin, xmax))
    ax.set_title(f"Samples {k.__class__.__name__}\n"+ r"\in \mathbf{R}", fontsize=21)


def plotkernelsample_in_Ps(k: Kernel, lvm: object, ax, xmin: float=-3., xmax: float=3., z_dim=2, n_elements=10):
    xx = tf.convert_to_tensor(np.linspace(xmin, xmax, z_dim)[None,:])
    lpp = lvm.decoder.layers(xx) 
    pp = tf.cast(tf.math.exp(lpp), tf.float64) # convert logits into probits
    K = k(pp)
    K = tf.squeeze(K)
    if K.shape[0] > n_elements:
        K = K[0:n_elements, 0:n_elements]
    ax.plot(np.arange(K.shape[0]), np.random.multivariate_normal(np.zeros(K.shape[0]), K, 3).T)
    ax.set_xlim((0, n_elements-1))
    ax.set_xlabel("position idx")
    ax.set_title(f"Samples {k.__class__.__name__}\n" + r"\in mathbf{P}", fontsize=21)


def plotlatentspace_lvm(k: Kernel, lvm: object, ax, xmin: float=-2., xmax: float=2., stepsize=0.2, vmin=0., vmax=1., cmap="viridis") -> object:
    xxyy = tf.convert_to_tensor(np.mgrid[xmin:xmax:stepsize, xmin:xmax:stepsize].reshape(2, -1).T)
    if "hellinger" not in k.__class__.__name__.lower():
        values = k(xxyy) # NOTE: This kernel is on latent space coordinates. One could consider a kernel on probits
    else:
        ps = lvm.p(xxyy)
        _ps = tf.squeeze(ps) # NOTE: shape input to be gpflow compliant [batch..., N, D] and D=L*cat
        ps = tf.reshape(_ps, shape=(1, _ps.shape[0], _ps.shape[-1]*_ps.shape[-2])) # keep first batch dim
        values = k(ps)
    values = tf.squeeze(values).numpy()
    img = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(f"2D Latent Space\n {k.__class__.__name__}", fontsize=20)
    return img

def plotlatentspace_lvm_refpoint(k_values: np.ndarray, ax, xmin: float=-2., xmax: float=2., stepsize=0.01, ref_point=(0,0), vmin=0., vmax=1., suffix="", cmap="viridis"):
    im_size = int(np.sqrt(k_values.shape)) # quadratic image
    ax.imshow(k_values.reshape((im_size, im_size)), vmin=vmin, vmax=vmax, cmap=cmap)
    # ax.plot(ref_point, "x", color="darkred" markersize=25.)
    # TODO: set correct x,y labels
    # if isinstance(ref_point, tf.Tensor):
    #     ref_point = ref_point.numpy()
    ref_point = np.array(tf.squeeze(ref_point))
    ax.set_title(f"z=[{str(np.round(ref_point[0], 2))}, {str(np.round(ref_point[1], 2))}]{suffix}", fontsize=23)


def plotlatentspace_lvm_refpoint_contour(k_values: Kernel, ax, xmin: float=-2., xmax: float=2., stepsize=0.01, ref_point=(0,0), vmin=0., vmax=1., suffix="", cmap="viridis"):
    im_size = int(np.sqrt(k_values.shape)) # quadratic image
    cs = ax.contour(k_values.reshape((im_size, im_size)), cmap=cmap)
    ax.clabel(cs, inline=True, fontsize=12)
    x_range = np.arange(xmin, xmax, step=stepsize)
    tick_labels = [f'{tick:.2f}' for tick in x_range]
    # ax.set_xticks(x_range, tick_labels)
    # ax.set_yticks(x_range, tick_labels)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels) #TODO tick-labels are off fix this!
    ax.set_xlabel(r"$z_1$", fontsize=18)
    ax.set_ylabel(r"$z_2$", fontsize=18)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ref_point = np.array(tf.squeeze(ref_point))
    ax.set_title(f"z=[{str(np.round(ref_point[0], 2))}, {str(np.round(ref_point[1], 2))}]{suffix}", fontsize=20)