"""
Module for computing enlacement/interlacement histograms between binary objects.
"""

import math
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.io import imread, imread_collection, imsave
from skimage.color import rgb2grey

import bresenham
import data


def _quadrant(theta):
    """
    Return whether an angle is in the horizontal quadrants or the vertical
    quadrants.
    """
    return ((0 <= theta <= np.pi / 4)
        or (3 * np.pi / 4 <= theta <= 5 * np.pi / 4)
        or (7 * np.pi / 4 <= theta < 2 * np.pi))


def _cart(a, b):
    """
    Return the cartesian product of `a` and `b`.
    """
    a_cart = np.repeat(a, len(b), axis=0)
    b_cart = np.repeat([b], len(a), axis=0).reshape(-1, 2)
    return a_cart, b_cart


def E(line, a, b):
    """
    Compute the interlacement of an object with regard to another along given a
    directional line.

    Parameters
    ----------
    line : (ii, jj) ndarrays of int
        The pixel coordinates of the line.
    a, b : (w, h,) array_like
        The two binary images to consider. `a` and `b` must have the same size.

    Returns
    -------
    E : scalar
        The interlacement of `a` with regard to `b` along the line `line`.

    """
    a_cuts = bresenham.cuts(line, a)
    b_cuts = bresenham.cuts(line, b)
    if len(a_cuts) == 0 or len(b_cuts) == 0:
        return 0

    # TODO: optimize the lines below

    bab = np.hstack(_cart(np.hstack(_cart(b_cuts, a_cuts)), b_cuts))
    mask = np.where(np.logical_and(bab[:, 2] > bab[:, 1], bab[:, 4] > bab[:, 3]))
    bab = bab[mask]

    return ((bab[:, 1] - bab[:, 0] + 1) * (bab[:, 3] - bab[:, 2] + 1) * (bab[:, 5] - bab[:, 4] + 1)).sum()


def enlacement(a, b, n_dirs=180, inversed=False, normalized=True, n_jobs=-1):
    """
    Compute the directional enlacement of an object with regard to an other.

    Parameters
    ----------
    a, b : (w, h,) array_like
        The two binary images to consider. `a` and `b` must have the same size.
    n_dirs : int
        The number of discrete directions to consider in [O, pi]. Default is 180.
    inversed : bool
        Inverse the force applied to object `a`. (experimental, default = False)
    normalized : bool
        Normalize by the area of objects for scaling invariance.
    n_jobs : int
        Number of jobs used to parallelize. If -1 all CPUs are used. If None a
        normal loop over discrete directions is used.

    Returns
    -------
    enlacement : (n_dirs,) ndarray
        The enlacement of `a` with regards to `b` along `n_dirs` directions.

    Notes
    -----
    Parallization using the joblib library.

    """
    if (a.shape != b.shape):
        raise ValueError("a and b must have the same shape.")
    h, w = a.shape

    thetas = np.linspace(0, np.pi, num=n_dirs, endpoint=False)

    # add 1 pixel pad to detect cuts on borders
    a = np.pad(a, pad_width=1, mode='constant', constant_values=0)
    b = np.pad(b, pad_width=1, mode='constant', constant_values=0)

    if n_jobs is not None:
        enlacement = np.array(joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_enlacement_direction)(a, b, theta, inversed)
                for theta in thetas))

    else:
        enlacement = np.zeros(n_dirs)
        for i, theta in enumerate(thetas):
            enlacement[i] = _enlacement_direction(a, b, theta, inversed)

    if normalized:
        a_area, b_area = len(np.where(a != 0)[0]), len(np.where(b != 0)[0])
        enlacement /= (a_area * b_area)

    return enlacement


def _enlacement_direction(a, b, theta, inversed):
    """Called by `enlacement` for the parallelization of each direction."""
    e = 0
    lines = bresenham.parallel_lines(theta, *a.shape)
    for line in lines:
        e += E(line, a, b)
    # Isotropy normalization
    if _quadrant(theta):
        e /= abs(math.cos(theta)) ** 2
    else:
        e /= abs(math.sin(theta)) ** 2
    return e


def interlacement(a, b, n_dirs=180, inversed=False, normalized=True, n_jobs=-1):
    """
    Compute the directional interlacement of two binary objects (harmonic mean
    of their respective enlacements).

    Parameters
    ----------
    a, b : (w, h,) array_like
        The two binary images to consider. `a` and `b` must have the same size.
    n_dirs : int
        The number of directions to consider. Default is 180.
    inversed : bool
        Inverse the force applied to respective objects enlacement.
        (experimental, default = False)
    normalized : bool
        Normalize by the area of objects for scaling invariance.
    n_jobs : int
        Number of jobs used to parallelize. If -1 all CPUs are used. If None a
        classical loop is used.

    Returns
    -------
    i_ab, e_ab, e_ba,  : (n_dirs,) ndarrays
        i_ab = interlacement between objects `a` and `b`.
        e_ab = enlacement of `a` with regards to `b`.
        e_ba = enlacement of `b` with regards to `a`.

    Notes
    -----
    Parallization using the joblib library.

    """
    e_ab = enlacement(a, b, n_dirs, inversed, normalized, n_jobs)
    e_ba = enlacement(b, a, n_dirs, inversed, normalized, n_jobs)
    # "0/0 = 0" convention is used
    olderr = np.seterr(invalid='ignore')
    i_ab = np.nan_to_num((2 * e_ab * e_ba) / (e_ab + e_ba))
    np.seterr(**olderr)
    return (i_ab, e_ab, e_ba)


def dump(path, a=None, b=None, e_ab=None, e_ba=None, i_ab=None,
    hm_ab=None, hm_ba=None):
    """
    Dump directional enlacement/interlacement descriptors to files.

    Parameters
    ----------
    path : str
        Path of the directory to dump files into.
    a, b : (w, h,) array_like, optional
        The binary images representing the objects used for the descriptors.
    e_ab : (n_dirs,) array_like, optional
        The enlacement descriptor E_AB.
    e_ba : (n_dirs,) array_like, optional
        The enlacement descriptor E_BA.
    i_ab : (n_dirs,) array_like, optional
        The interlacement descriptor I_AB.

    Notes
    -----
    At least `e_ab`, `e_ba` or `i_ab` arguments must be given.

    """
    if ((e_ab is None) and (e_ba is None) and (i_ab is None) and (hm_ab is None)
            and (hm_ba is None)):
        raise ValueError('require at least some data to save.')

    # Create target directory if needed
    if not os.path.exists(path):
        os.makedirs(path)

    # Save files
    if (a is not None):
        imsave(os.path.join(path, 'a.png'), a)
    if (b is not None):
        imsave(os.path.join(path, 'b.png'), b)
    if (e_ab is not None):
        data.dump(e_ab, os.path.join(path, 'e_ab.pickle'))
    if (e_ba is not None):
        data.dump(e_ba, os.path.join(path, 'e_ba.pickle'))
    if (i_ab is not None):
        data.dump(i_ab, os.path.join(path, 'i_ab.pickle'))

    # Save heatmaps
    if (hm_ab is not None):
        hm_path = os.path.join(path, 'hm_ab')
        if not os.path.exists(hm_path):
            os.makedirs(hm_path)
        n_dirs = hm_ab.shape[0]
        for i, hm in enumerate(hm_ab):
            degree = str(i * (360 // n_dirs)).zfill(3)
            plt.imsave(os.path.join(hm_path, '{}.png'.format(degree)), hm, cmap='gray')
        plt.imsave(os.path.join(path, 'hm_ab.png'), np.sum(hm_ab, axis=0), cmap='gray')

    if (hm_ba is not None):
        hm_path = os.path.join(path, 'hm_ba')
        if not os.path.exists(hm_path):
            os.makedirs(hm_path)
        n_dirs = hm_ba.shape[0]
        for i, hm in enumerate(hm_ba):
            degree = str(i * (360 // n_dirs)).zfill(3)
            plt.imsave(os.path.join(hm_path, '{}.png'.format(degree)), hm, cmap='gray')
        plt.imsave(os.path.join(path, 'hm_ba.png'), np.sum(hm_ba, axis=0), cmap='gray')

    if (hm_ab is not None) and (hm_ba is not None):
        plt.imsave(os.path.join(path, 'hm.png'), np.sum(hm_ab + hm_ba, axis=0), cmap='gray')


def load(path):
    """
    Return a dictionary containing enlacement/interlacement descriptors loaded
    from a results directory.

    Parameters
    ----------
    path : str
        Path of the results directory.

    Returns
    -------
    d : dict
        A dictionary with the following keys:
        - 'a': image of the object A
        - 'b': image of the object B
        - 'e_ab': enlacement descriptor E_AB
        - 'e_ba': enlacement descriptor E_BA
        - 'i_ab': enlacement descriptor I_AB.
        - 'hm_ab': heatmaps of the E_AB enlacement descriptor
        - 'hm_ba': heatmaps of the E_BA enlacement descriptor

    Notes
    -----
    If a file (image or descriptor) is not found, the corresponding key in the
    dictionary is set to `None`.

    """
    d = {}

    # Load object images if they exist
    for im in ('a', 'b'):
        try:
            d[im] = imread(os.path.join(path, '{}.png'.format(im)),
                           as_grey=True)
        except:
            d[im] = None

    # Load descriptors if they exist
    for desc in ('e_ab', 'e_ba', 'i_ab'):
        try:
            # Pickle file is the default
            d[desc] = data.load(os.path.join(path, '{}.pickle'.format(desc)))
        except:
            try:
                # Fallback method for compatibility
                d[desc] = np.loadtxt(os.path.join(path, '{}.txt'.format(desc)))
            except:
                d[desc] = None

    # Load heatmaps if they exist
    for hm in ('hm_a', 'hm_b'):
        try:
            hm_path = os.path.join(path, hm)
            coll = imread_collection(os.path.join(hm_path, '*.png'))
            d[hm] = []
            for im in coll:
                d[hm].append(rgb2grey(im))
            d[hm] = np.array(d[hm])
        except:
            d[hm] = None

    return d


def surrounding(e_ab, alpha=0.2):
    """
    Evaluate the surrounding spatial relation using the enlacement descriptor.

    Parameters
    ----------
    e_ab : (n_dirs,) array_like
        The enlacement descriptor of object A by object B.
    alpha : float between 0 and 1
        Tolerance threshold.

    Returns
    -------
    s_ab : float between 0 and 1
        Evaluation of the proposition``A is surrounded by B''.

    """
    if (not 0.0 <= alpha <= 1.0):
        raise ValueError('alpha should be a real number between 0 and 1.')

    if e_ab.max() > 0:
        e_ab /= e_ab.max()

    if alpha != 0:
        alpha_cut = np.minimum(e_ab, alpha)
        return (alpha_cut.sum() / (alpha * alpha_cut.size))

    else:
        return 1 - (e_ab[e_ab == 0].size / e_ab.size)


def enlacement_heatmaps(a, b, thetas=None, normalized=True, n_jobs=-1):
    """
    Compute individual heatmaps of the enlacement of an object by another for a
    set of given directions.

    Parameters
    ----------
    a, b : (w, h,) array_like
        The two binary images to consider. `a` and `b` must have the same size.
    thetas : float
        The direction angles to compute the individual heatmaps. Defaults to 180
        directions in [0, pi].

    Returns
    -------
    heatmaps : (n, w, h,) array_like
        The individual heatmaps of the enlacement of `a` by `b` in directions
        `thetas`.

    """
    if (a.shape != b.shape):
        raise ValueError("a and b must have the same shape.")

    if thetas is None:
        thetas = np.linspace(0, np.pi, num=180, endpoint=False)

    # add 1 pixel pad to detect cuts on borders
    a = np.pad(a, pad_width=1, mode='constant', constant_values=0)
    b = np.pad(b, pad_width=1, mode='constant', constant_values=0)

    a_area, b_area = len(np.where(a != 0)[0]), len(np.where(b != 0)[0])

    if n_jobs is not None:
        heatmaps = np.array(joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_enlacement_heatmaps_direction)(a, b, theta)
                for theta in thetas))

    else:
        heatmaps = []
        for theta in thetas:
            heatmaps.append(_enlacement_heatmaps_direction(a, b, theta))
        heatmaps = np.array(heatmaps)

    if normalized:
        heatmaps /= b_area

    return heatmaps


def _enlacement_heatmaps_direction(a, b, theta):

    heatmap = np.zeros((a.shape[0], a.shape[1]))
    lines = bresenham.parallel_lines(theta, *a.shape)

    for line in lines:

        a_cuts = bresenham.cuts(line, a)
        b_cuts = bresenham.cuts(line, b)

        if len(a_cuts) == 0 or len(b_cuts) == 0:
            continue

        ba = np.hstack(_cart(b_cuts, a_cuts))
        bab = np.hstack(_cart(ba, b_cuts))
        mask = np.where(np.logical_and(bab[:, 2] > bab[:, 1], bab[:, 4] > bab[:, 3]))
        bab = bab[mask]

        E = (bab[:, 1] - bab[:, 0] + 1) * (bab[:, 5] - bab[:, 4] + 1)

        for cut, e in zip(bab, E):
            a1, a2 = cut[2], cut[3] + 1
            heatmap[line[0][a1:a2], line[1][a1:a2]] += e

    # Isotropy normalization
    if _quadrant(theta):
        heatmap /= abs(math.cos(theta)) ** 2
    else:
        heatmap /= abs(math.sin(theta)) ** 2

    heatmap = heatmap[1:-1, 1:-1] # remove 1 pixel pad added before

    return heatmap


# TODO
# def evaluation(e_ab, alpha, width):
#     if width == 0:
#         raise ValueError('width must be > 0.')

#     # get closest alpha and width to discrete directions of e_ab
#     thetas = np.linspace(0, np.pi, num=len(e_ab))
#     arg_alpha = np.argmin(np.abs(thetas - alpha))
#     arg_width = np.argmin(np.abs(thetas - width))

#     # roll to center on first value on theta
#     roll = np.roll(e_ab, -arg_alpha)

#     # closest number of discrete directions to width
#     arg_width
#     # return evaluation
#     return (roll[:arg_width].sum() + roll[-arg_width:].sum()) / (arg_width * 2)
