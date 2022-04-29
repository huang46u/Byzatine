"""
Module for generating pixel coordinates of oriented lines in images.
"""

import math
import numpy as np
from skimage.draw import line as skimage_line


def line(theta, rho):
    """
    Return the pixel coordinates of a `theta` oriented line of length `rho`.

    Parameters
    ----------
    theta : float
        The orientation angle in radians.
    rho : float
        The length of the line from the origin.

    Returns
    -------
    ii, jj : (N,) ndarray of int
        Pixel coordinates of the line (row-column format).

    Notes
    -----
    Merely a wrapper for `skimage.draw.line` with polar coordinates.

    """
    x1, y1 = (0, 0)
    x2 = int(round(rho * math.cos(theta)))
    y2 = int(round(rho * math.sin(theta)))
    ii, jj = skimage_line(y1, x1, y2, x2)
    return ii, jj


def restrict(ii, jj, height, width):
    """
    Restrict the pixel coordinates of a line to the size of an image.

    Parameters
    ----------
    ii, jj : (N,) ndarray of int
        Pixel coordinates of the line.
    height, width : (int, int,)
        The size of the image.

    Returns
    -------
    ii, jj : (N,) ndarray of int
        Pixel coordinates of the line, restricted to the image size.

    Notes
    -----
    This function returns a copy of the initial pixel coordinates (i.e., the
    initial parameters `ii` and `jj` are untouched).

    """
    mask = (ii >= 0) & (ii < height) & (jj >= 0) & (jj < width)
    return ii[mask], jj[mask]


def parallel_lines(theta, height, width):
    """
    Return the pixel coordinates of all `theta` oriented parallel lines going
    through an image of given size.

    Parameters
    ----------
    theta : float
        The orientation angle in radians.
    height, width : (int, int,)
        The size of the image.

    Returns
    -------
    lines : list
        A list containing pixel coordinates of all parallel lines going through
        the image.

    """
    # Draw an initial line of sufficient length
    theta = math.fmod(theta, 2 * np.pi)
    rho = 2 * max(height, width)
    ii, jj = line(theta, rho)
    # Consider the bottom-left corner to be the origin
    ii = height - ii - 1

    # Translate the initial line according to the orientation angle
    # 1st quadrant from the bottom-left corner
    if 0 <= theta <= np.pi / 2:
        tx, ty = 0, 0
    # 2nd quadrant from the bottom-right corner
    elif np.pi / 2 < theta < np.pi:
        tx, ty = 0, width - 1
    # 3rd quadrant from the top-right corner
    elif np.pi <= theta <= 3 * np.pi / 2:
        tx, ty = 1 - height, width - 1
    # 4th quadrant from the top-left corner
    else:
        tx, ty = 1 - height, 0
    ii += tx
    jj += ty

    # Build the set of parallel lines by shifting the initial line
    lines = []
    ii_u, jj_u = ii_d, jj_d = restrict(ii, jj, height, width)
    lines.append((ii_u, jj_u)) # initial line

    # Shift the initial line downwards
    shift = 1
    while ii_u.size > 0 and jj_u.size > 0:
        if abs(ii[-1] - ii[0]) <= abs(jj[-1] - jj[0]):
            ii_u, jj_u = restrict(ii - shift, jj, height, width)
        else:
            ii_u, jj_u = restrict(ii, jj - shift, height, width)
        if ii_u.size > 0 and jj_u.size > 0:
            lines.append((ii_u, jj_u))
        shift += 1

    lines.reverse() # to keep a logical order

    # Shift the initial line upwards
    shift = 1
    while ii_d.size > 0 and jj_d.size > 0:
        if abs(ii[-1] - ii[0]) <= abs(jj[-1] - jj[0]):
            ii_d, jj_d = restrict(ii + shift, jj, height, width)
        else:
            ii_d, jj_d = restrict(ii, jj + shift, height, width)
        if ii_d.size > 0 and jj_d.size > 0:
            lines.append((ii_d, jj_d))
        shift += 1

    return lines


def cuts(line, im):
    """
    Return the longitudinal cut (list of segments) of an object along given line
    coordinates.

    Parameters
    ----------
    line : (ii, jj) ndarray of int
        Pixel coordinates of the line.
    im : (w, h) array_like
        The binary image representing the object.

    Returns
    -------
    cut_ends : (n_cuts, 2) ndarray
        List of longitudinal cut ends.

    Notes
    -----
    A pixel belongs to the object if its value is not 0.

    """
    cut = im[line] != 0
    cut_ends, = np.where(np.diff(cut) == True)
    if cut_ends.size <= 0:
        return []
    cut_ends = cut_ends.reshape(-1, 2)
    cut_ends[:, 0] += 1
    return cut_ends
