#!/usr/bin/env python
# -*- coding: utf-8 -*-
# rolling_ball.py
"""
Implements a few algos for a Rolling ball filter.

There are two separate implenetations in this file.

One is _exact_ and uses the concept of alpha shapes to estimate the background, but it is slow
and is only implemented in 2D so far.

The other is an approximation based on top hat transforms https://en.wikipedia.org/wiki/Top-hat_transform.https
It is fast and relatively accurate so long as the slope is not too steep in the image.

References
----------
- https://media.nature.com/original/nature-assets/srep/2016/160725/srep30179/extref/srep30179-s1.pdf
- https://github.com/imagej/imagej1/blob/master/ij/plugin/filter/BackgroundSubtracter.java
- http://ieeexplore.ieee.org/document/1654163/?reload=true

https://plot.ly/python/alpha-shapes/
In a family of alpha shapes, the parameter α controls the level of detail of the associated alpha shape.
If α decreases to zero, the corresponding alpha shape degenerates to the point set, S, while if it tends to
infinity the alpha shape tends to the convex hull of the set S.


Copyright (c) 2018, David Hoffman
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import interpolate, spatial
from scipy.ndimage._ni_support import _normalize_sequence


def sq_norm(v):
    """Squared norm."""
    return (v ** 2).sum(0)


def circumcircle(points, simplex):
    """Get the circumcenter and circum radius of all the simplices, works for 2D only.

    Compute the circumcenter and circumradius of a triangle (see their definitions
    https://en.wikipedia.org/wiki/Circumscribed_circle#Circumcircle_equations)

    http://mathworld.wolfram.com/Circumcircle.html
    """
    d = len(simplex)
    A = [points[k] for k in simplex]
    M = [[1.0] * (d + 1)]
    M += [[sq_norm(A[k]), A[k][0], A[k][1], 1.0] for k in range(d)]
    M = np.asarray(M, dtype=float)
    # M is equation (2) at mathworld
    # eq. (4)
    a = np.linalg.det(M[1:, 1:])
    # bx is (6)
    bx = -np.linalg.det(M[1:, [0, 2, 3]])
    # by is (7)
    by = np.linalg.det(M[1:, [0, 1, 3]])
    # eq. (8)
    c = -np.linalg.det(M[1:, :-1])
    # eqns. (11) and (12)
    center = np.array([-bx, -by]) / a / 2
    # eqn. (13)
    radius = np.sqrt(sq_norm(center) - c / a)
    return center, radius


def get_alpha_complex(alpha, points, simplices):
    """Get alpha complex."""
    # find the centers and radii of all circumcircles
    centers, radii = np.array([circumcircle(points, s) for s in simplices]).T
    # convert centers to array
    centers = np.vstack(centers)
    # find which points you want to keep
    to_keep = radii > alpha
    alpha_complex = points[simplices[to_keep]]
    # calculate where the center of the circle lies with respect to the vertices
    above_right = alpha_complex > centers[to_keep][:, None]
    return alpha_complex, above_right


def rolling_ball_filter_accurate(
    data, ball_radius, roll_along=-1, top=True, interpolator=interpolate.interp1d, **kwargs
):
    """Filter data via a rolling ball algorithm.
    
    Rolling ball filter implemented with alpha shapes
    
    Parameters
    ----------
    data : ndarray (n, d)
        Array of data points, assumed xyz ordering
    ball_radius : float
        The size of the ball to roll
    roll_along : int
        The axis perpendicular to the roll direction
    top : bool
        Top or bottom
    interpolator : callable
        needs to take two arrays and return a callable
    kwargs : for interpolator
    
    Returns
    -------
    data : ndarray (n, d)
        Smoothed data
    """
    n, d = data.shape
    tri = spatial.Delaunay(data)
    alpha_complex, above_right = get_alpha_complex(ball_radius, tri.points, tri.simplices)
    # calculate the ball rolling from the right
    # sum along vertices and direction, determine if more than one vertex is
    kept_points = above_right.sum(1)[:, roll_along] > 1
    if top:
        kept_points = ~kept_points
    d_kept = np.unique(alpha_complex[kept_points].reshape(-1, d), axis=0)
    x, y = data.T
    X, Y = d_kept.T
    interp = interpolator(X, Y, **kwargs)
    return x, interp(x)


def rolling_ball_filter(data, ball_radius, spacing=None, top=False, **kwargs):
    """Filter data via a rolling ball algorithm.

    Implemented with morphological operations

    This implenetation is very similar to that in ImageJ and uses a top hat transform
    with a ball shaped structuring element
    https://en.wikipedia.org/wiki/Top-hat_transform

    Parameters
    ----------
    data : ndarray
        image data (assumed to be on a regular grid)
    ball_radius : float
        the radius of the ball to roll
    spacing : int or sequence
        the spacing of the image data
    top : bool
        whether to roll the ball on the top or bottom of the data
    kwargs : key word arguments
        these are passed to the ndimage morphological operations

    Returns
    -------
    data_nb : ndarray
        data with background subtracted
    bg : ndarray
        background that was subtracted from the data
    """
    # get dimension of data
    ndim = data.ndim
    # set spacing to 1 if not specified
    if spacing is None:
        spacing = np.ones(ndim)
    else:
        spacing = _normalize_sequence(spacing, ndim)
    # arrayify radius
    radius = np.asarray(_normalize_sequence(ball_radius, ndim))
    # generate the mesh for the sphere
    mesh = np.array(
        np.meshgrid(*[np.arange(-r, r + s, s) for r, s in zip(radius, spacing)], indexing="ij")
    )
    # make the sphere and replace nan with 0
    structure = 2 * np.sqrt(1 - ((mesh / radius.reshape(-1, *((1,) * ndim))) ** 2).sum(0))
    structure[~np.isfinite(structure)] = 0
    # roll ball on top or bottom dpending on request
    if not top:
        # ndi.white_tophat(y, structure=structure, output=background)
        background = ndi.grey_erosion(data, structure=structure, **kwargs)
        background = ndi.grey_dilation(background, structure=structure, **kwargs)
    else:
        # ndi.black_tophat(y, structure=structure, output=background)
        background = ndi.grey_dilation(data, structure=structure, **kwargs)
        background = ndi.grey_erosion(background, structure=structure, **kwargs)

    return data - background, background


if __name__ == "__main__":
    # import plotting
    import matplotlib.pyplot as plt

    # remove randomness
    np.random.seed(42)
    # generate toy data
    x = np.linspace(-2 * np.pi, 2 * np.pi, 256)
    y = np.sin(10 * x)
    y *= np.exp(-2 * x ** 2)
    y += np.poly1d(np.random.randn(3))(x) * 0.01

    ball_r = 0.5
    X_top, Y_top = rolling_ball_filter_accurate(np.array((x, y)).T, ball_r)
    X_bottom, Y_bottom = rolling_ball_filter_accurate(np.array((x, y)).T, ball_r, top=False)
    spacing = x[1] - x[0]
    _, Yt_fast = rolling_ball_filter(y, ball_r, spacing, top=True)
    _, Yb_fast = rolling_ball_filter(y, ball_r, spacing, top=False)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Data")
    ax.plot(X_top, Y_top, label="Rolled on top")
    ax.plot(X_bottom, Y_bottom, label="Rolled on bottom")
    ax.plot(x, Yt_fast, label="Rolled on top fast")
    ax.plot(x, Yb_fast, label="Rolled on bottom fast")
    ax.legend()
    ax.set_title("Rolling ball filter with radius = {}".format(ball_r))

    plt.show()
