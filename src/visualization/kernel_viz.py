import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.kernels import Kernel


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

def plotlatentspace_lvm_refpoint(k: Kernel, lvm: object, ax, xmin: float=-2., xmax: float=2., stepsize=0.01, ref_point=(0,0), vmin=0., vmax=1., cmap="viridis"):
    xxyy = tf.convert_to_tensor(np.mgrid[xmin:xmax:stepsize, xmin:xmax:stepsize].reshape(2, -1).T)
    if not isinstance(ref_point, tf.Tensor): # convert tuple to tensor
        ref_point = tf.cast(tf.convert_to_tensor(ref_point)[None,:], tf.float64)
    if "hellinger" not in k.__class__.__name__.lower(): # compare euclidean kernels directly in latent space
        # mean reduction across positions # CASE: Matern
        values = k(xxyy, ref_point)
    elif "weighted" not in k.__class__.__name__.lower(): # CASE: base HK
        decoded_latent = lvm.vae.decode(xxyy) # TODO: this is for the 2D case
        ps = tf.reshape(tf.squeeze(decoded_latent), shape=(len(decoded_latent), decoded_latent.shape[-1]*decoded_latent.shape[-2]))
        decoded_reference = lvm.vae.decode(ref_point.reshape(1, 2)) 
        ps_ref = tf.reshape(tf.squeeze(decoded_reference), shape=(1, decoded_reference.shape[-1]*decoded_reference.shape[-2]))
        values = k(ps, ps_ref)
    else: # CASE: wHK
        decoded_latent = lvm.vae.decode(xxyy)
        decoded_seq = tf.one_hot(tf.argmax(decoded_latent, axis=-1), k.AA).reshape(1, decoded_latent.shape[0], k.L*k.AA)
        decoded_reference = lvm.vae.decode(ref_point.reshape(1, 2))
        decoded_ref_seq = tf.one_hot(tf.argmax(decoded_reference, axis=-1), k.AA).reshape(1, ref_point.shape[0], k.L*k.AA)
        values = k(decoded_seq, decoded_ref_seq) # NOTE: weighting distribution in weighting vector, use OneHot for X
    values = tf.squeeze(values).numpy()
    im_size = int(np.sqrt(values.shape)) # quadratic image
    ax.imshow(values.reshape((im_size, im_size)), vmin=vmin, vmax=vmax, cmap=cmap)
    # ax.plot(ref_point, "x", color="darkred" markersize=25.)
    # TODO: set correct x,y labels
    # if isinstance(ref_point, tf.Tensor):
    #     ref_point = ref_point.numpy()
    ref_point = np.array(tf.squeeze(ref_point))
    ax.set_title(f"z=[{str(np.round(ref_point[0], 2))}, {str(np.round(ref_point[1], 2))}]\n {k.__class__.__name__}", fontsize=23)