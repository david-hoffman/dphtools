#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
Various utility functions to be organized better

Copyright (c) 2021, David Hoffman
"""

import io
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy as sp
import tifffile as tif
import tqdm
from scipy.fftpack.helper import next_fast_len
from scipy.ndimage._ni_support import _normalize_sequence
from scipy.ndimage.fourier import fourier_gaussian
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import signaltools as sig
from scipy.special import zeta
from scipy.stats import nbinom

from .lm import curve_fit
from .rolling_ball import rolling_ball_filter

# from .llc import jit_filter_function, jit_filter1d_function

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fftn, fftshift, ifftn, ifftshift, irfftn, rfftn

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
    FFTW = True
except ImportError:
    from numpy.fft import fftn, fftshift, ifftn, ifftshift, irfftn, rfftn

    FFTW = False


import logging

logger = logging.getLogger(__name__)

eps = np.finfo(float).eps


def get_git(path="."):
    try:
        # we slice to remove trailing new line.
        cmd = ["git", "--git-dir=" + os.path.join(path, ".git"), "describe", "--long", "--always"]
        return subprocess.check_output(cmd).decode()[:-1]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(e)
        logger.error(" ".join(cmd))
        return "Unknown"


def generate_meta_data():
    pass


def bin_ndarray(ndarray, new_shape=None, bin_size=None, operation="sum"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Parameters
    ----------
    ndarray : array like object (can be dask array)
    new_shape : iterable (optional)
        The new size to bin the data to
    bin_size : scalar or iterable (optional)
        The size of the new bins

    Returns
    -------
    binned array.
    """
    if new_shape is None:
        # if new shape isn't passed then calculate it
        if bin_size is None:
            # if bin_size isn't passed then raise error
            raise ValueError("Either new shape or bin_size must be passed")
        # pull old shape
        old_shape = np.array(ndarray.shape)
        # calculate new shape, integer division!
        new_shape = old_shape // bin_size
        # calculate the crop window
        crop = tuple(slice(None, -r) if r else slice(None) for r in old_shape % bin_size)
        # crop the input array
        ndarray = ndarray[crop]
    # proceed as before
    operation = operation.lower()
    if operation not in {"sum", "mean"}:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError(f"Shape mismatch: {ndarray.shape} -> {new_shape}")
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def scale(data, dtype=None):
    """
    Scales data to [0.0, 1.0] range, unless an integer dtype is specified
    in which case the data is scaled to fill the bit depth of the dtype.

    Parameters
    ----------
    data : numeric type
        Data to be scaled, can contain nan
    dtype : integer dtype
        Specify the bit depth to fill

    Returns
    -------
    scaled_data : numeric type
        Scaled data

    Examples
    --------
    >>> from numpy.random import randn
    >>> a = randn(10)
    >>> b = scale(a)
    >>> b.max()
    1.0
    >>> b.min()
    0.0
    >>> b = scale(a, dtype = np.uint16)
    >>> b.max()
    65535
    >>> b.min()
    0
    """
    if np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError("`scale` is not defined for complex values")
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if np.issubdtype(dtype, np.integer):
        tmin = np.iinfo(dtype).min
        tmax = np.iinfo(dtype).max
    else:
        tmin = 0.0
        tmax = 1.0
    return ((data - dmin) / (dmax - dmin) * (tmax - tmin) + tmin).astype(dtype)


def scale_uint16(data):
    """Convenience function to scale data to the uint16 range."""
    return scale(data, np.uint16)


def radial_profile(data, center=None, binsize=1.0):
    """Take the radial average of a 2D data array

    Adapted from http://stackoverflow.com/a/21242776/5030014

    See https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py
    for an alternative

    Parameters
    ----------
    data : ndarray (2D)
        the 2D array for which you want to calculate the radial average
    center : sequence
        the center about which you want to calculate the radial average
    binsize : sequence
        Size of radial bins, numbers less than one have questionable utility

    Returns
    -------
    radial_mean : ndarray
        a 1D radial average of data
    radial_std : ndarray
        a 1D radial standard deviation of data

    Examples
    --------
    >>> radial_profile(np.ones((11, 11)))
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    """
    # test if the data is complex
    if np.iscomplexobj(data):
        # if it is complex, call this function on the real and
        # imaginary parts and return the complex sum.
        real_prof, real_std = radial_profile(np.real(data), center, binsize)
        imag_prof, imag_std = radial_profile(np.imag(data), center, binsize)
        return real_prof + imag_prof * 1j, np.sqrt(real_std ** 2 + imag_std ** 2)
        # or do mag and phase
        # mag_prof, mag_std = radial_profile(np.abs(data), center, binsize)
        # phase_prof, phase_std = radial_profile(np.angle(data), center, binsize)
        # return mag_prof * np.exp(phase_prof * 1j), mag_std * np.exp(phase_std * 1j)
    # pull the data shape
    idx = np.indices((data.shape))
    if center is None:
        # find the center
        center = np.array(data.shape) // 2
    else:
        # make sure center is an array.
        center = np.asarray(center)
    # calculate the radius from center
    idx2 = idx - center[(Ellipsis,) + (np.newaxis,) * (data.ndim)]
    r = np.sqrt(np.sum([i ** 2 for i in idx2], 0))
    # convert to int
    r = np.round(r / binsize).astype(np.int)
    # sum the values at equal r
    tbin = np.bincount(r.ravel(), data.ravel())
    # sum the squares at equal r
    tbin2 = np.bincount(r.ravel(), (data ** 2).ravel())
    # find how many equal r's there are
    nr = np.bincount(r.ravel())
    # calculate the radial mean
    # NOTE: because nr could be zero (for missing bins) the results will
    # have NaN for binsize != 1
    radial_mean = tbin / nr
    # calculate the radial std
    radial_std = np.sqrt(tbin2 / nr - radial_mean ** 2)
    # return them
    return radial_mean, radial_std


def mode(data):
    """Quickly find the mode of data

    up to 1000 times faster than scipy mode
    but not nearly as feature rich

    Note: we can vectorize this to work on different
    axes with numba"""
    # will not work with negative numbers (for now)
    return np.bincount(data.ravel()).argmax()


def slice_maker(xs, ws):
    """
    A utility function to generate slices for later use.

    Parameters
    ----------
    y0 : int
        center y position of the slice
    x0 : int
        center x position of the slice
    width : int
        Width of the slice

    Returns
    -------
    slices : list
        A list of slice objects, the first one is for the y dimension and
        and the second is for the x dimension.

    Notes
    -----
    The method will automatically coerce slices into acceptable bounds.

    Examples
    --------
    >>> slice_maker((30,20),10)
    [slice(25, 35, None), slice(15, 25, None)]
    >>> slice_maker((30,20),25)
    [slice(18, 43, None), slice(8, 33, None)]
    """
    # normalize inputs
    xs = np.asarray(xs)
    ws = np.asarray(_normalize_sequence(ws, len(xs)))
    if not np.isrealobj((xs, ws)):
        raise TypeError("`slice_maker` only accepts real input")
    if np.any(ws < 0):
        raise ValueError(f"width cannot be negative, width = {ws}")
    # ensure integers
    xs = np.rint(xs).astype(int)
    ws = np.rint(ws).astype(int)
    # use _calc_pad
    toreturn = []
    for x, w in zip(xs, ws):
        half2, half1 = _calc_pad(0, w)
        xstart = x - half1
        xend = x + half2
        assert xstart <= xend, "xstart > xend"
        if xend <= 0:
            xstart, xend = 0, 0
        # the max calls are to make slice_maker play nice with edges.
        toreturn.append(slice(max(0, xstart), xend))
    # return a list of slices
    return tuple(toreturn)


def fft_pad(array, newshape=None, mode="median", **kwargs):
    """Pad an array to prep it for fft"""
    # pull the old shape
    oldshape = array.shape
    if newshape is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(next_fast_len(n) for n in oldshape)
    else:
        if isinstance(newshape, int):
            newshape = tuple(newshape for n in oldshape)
        else:
            newshape = tuple(newshape)
    # generate padding and slices
    padding, slices = padding_slices(oldshape, newshape)
    return np.pad(array[slices], padding, mode=mode, **kwargs)


def padding_slices(oldshape, newshape):
    """This function takes the old shape and the new shape and calculates
    the required padding or cropping.newshape

    Can be used to generate the slices needed to undo fft_pad above"""
    # generate pad widths from new shape
    padding = tuple(
        _calc_pad(o, n) if n is not None else _calc_pad(o, o) for o, n in zip(oldshape, newshape)
    )
    # Make a crop list, if any of the padding is negative
    slices = tuple(_calc_crop(s1, s2) for s1, s2 in padding)
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return padding, slices


# def easy_rfft(data, axes=None):
#     """utility method that includes fft shifting"""
#     return fftshift(
#         rfftn(
#             ifftshift(
#                 data, axes=axes
#             ), axes=axes
#         ), axes=axes)


# def easy_irfft(data, axes=None):
#     """utility method that includes fft shifting"""
#     return ifftshift(
#         irfftn(
#             fftshift(
#                 data, axes=axes
#             ), axes=axes
#         ), axes=axes)

# add np.pad docstring
fft_pad.__doc__ += np.pad.__doc__


def _calc_crop(s1, s2):
    """Calc the cropping from the padding"""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def _calc_pad(oldnum, newnum):
    """ Calculate the proper padding for fft_pad

    We have three cases:
    old number even new number even
    >>> _calc_pad(10, 16)
    (3, 3)

    old number odd new number even
    >>> _calc_pad(11, 16)
    (2, 3)

    old number odd new number odd
    >>> _calc_pad(11, 17)
    (3, 3)

    old number even new number odd
    >>> _calc_pad(10, 17)
    (4, 3)

    same numbers
    >>> _calc_pad(17, 17)
    (0, 0)

    from larger to smaller.
    >>> _calc_pad(17, 10)
    (-4, -3)
    """
    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side, smaller
    pad_s = width // 2
    # calculate the other, bigger
    pad_b = width - pad_s
    # if oldnum is odd and newnum is even
    # we want to pull things backward
    if oldnum % 2:
        pad1, pad2 = pad_s, pad_b
    else:
        pad1, pad2 = pad_b, pad_s
    return pad1, pad2


# If we have fftw installed than make a better fftconvolve
if FFTW:

    def fftconvolve(in1, in2, mode="same", threads=1):
        """Same as above but with pyfftw added in"""
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)

        if in1.ndim == in2.ndim == 0:  # scalar inputs
            return in1 * in2
        elif not in1.ndim == in2.ndim:
            raise ValueError("in1 and in2 should have the same dimensionality")
        elif in1.size == 0 or in2.size == 0:  # empty arrays
            return np.array([])

        s1 = np.array(in1.shape)
        s2 = np.array(in2.shape)
        complex_result = np.issubdtype(in1.dtype, complex) or np.issubdtype(in2.dtype, complex)
        shape = s1 + s2 - 1

        # Check that input sizes are compatible with 'valid' mode
        if sig._inputs_swap_needed(mode, s1, s2):
            # Convolution is commutative; order doesn't have any effect on output
            in1, s1, in2, s2 = in2, s2, in1, s1

        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [next_fast_len(int(d)) for d in shape]
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.
        if not complex_result and (sig._rfft_mt_safe or sig._rfft_lock.acquire(False)):
            try:
                sp1 = rfftn(in1, fshape, threads=threads)
                sp2 = rfftn(in2, fshape, threads=threads)
                ret = irfftn(sp1 * sp2, fshape, threads=threads)[fslice].copy()
            finally:
                if not sig._rfft_mt_safe:
                    sig._rfft_lock.release()
        else:
            # If we're here, it's either because we need a complex result, or we
            # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
            # is already in use by another thread).  In either case, use the
            # (threadsafe but slower) SciPy complex-FFT routines instead.
            sp1 = fftn(in1, fshape, threads=threads)
            sp2 = fftn(in2, fshape, threads=threads)
            ret = ifftn(sp1 * sp2, threads=threads)[fslice].copy()
            if not complex_result:
                ret = ret.real

        if mode == "full":
            return ret
        elif mode == "same":
            return sig._centered(ret, s1)
        elif mode == "valid":
            return sig._centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'.")

    # fftconvolve.__doc__ = "DPH Utils: " + sig.fftconvolve.__doc__
else:
    fftconvolve = sig.fftconvolve


def fftconvolve_fast(data, kernel, **kwargs):
    """A faster version of fft convolution

    In this case the kernel ifftshifted before FFT but the data is not.
    This can be done because the effect of fourier convolution is to 
    "wrap" around the data edges so whether we ifftshift before FFT
    and then fftshift after it makes no difference so we can skip the
    step entirely.
    """
    # TODO: add error checking like in the above and add functionality
    # for complex inputs. Also could add options for different types of
    # padding.
    dshape = np.array(data.shape)
    kshape = np.array(kernel.shape)
    # find maximum dimensions
    maxshape = np.max((dshape, kshape), 0)
    # calculate a nice shape
    fshape = [next_fast_len(int(d)) for d in maxshape]
    # pad out with reflection
    pad_data = fft_pad(data, fshape, "reflect")
    # calculate padding
    padding = tuple(_calc_pad(o, n) for o, n in zip(data.shape, pad_data.shape))
    # so that we can calculate the cropping, maybe this should be integrated
    # into `fft_pad` ...
    fslice = tuple(slice(s, -e) if e != 0 else slice(s, None) for s, e in padding)
    if kernel.shape != pad_data.shape:
        # its been assumed that the background of the kernel has already been
        # removed and that the kernel has already been centered
        kernel = fft_pad(kernel, pad_data.shape, mode="constant")
    k_kernel = rfftn(ifftshift(kernel), pad_data.shape, **kwargs)
    k_data = rfftn(pad_data, pad_data.shape, **kwargs)
    convolve_data = irfftn(k_kernel * k_data, pad_data.shape, **kwargs)
    # return data with same shape as original data
    return convolve_data[fslice]


def win_nd(size, win_func=sp.signal.hann, **kwargs):
    """
    A function to make a multidimensional version of a window function

    Parameters
    ----------
    size : tuple of ints
        size of the output window
    win_func : callable
        Default is the Hanning window
    **kwargs : key word arguments to be passed to win_func

    Returns
    -------
    w : ndarray
        window function
    """
    ndim = len(size)
    newshapes = tuple(
        [tuple([1 if i != j else k for i in range(ndim)]) for j, k in enumerate(size)]
    )

    # Initialize to return
    toreturn = 1.0

    # cross product the 1D windows together
    for newshape in newshapes:
        toreturn = toreturn * win_func(max(newshape), **kwargs).reshape(newshape)

    # return
    return toreturn


def anscombe(data):
    """Apply Anscombe transform to data

    https://en.wikipedia.org/wiki/Anscombe_transform
    """
    return 2 * np.sqrt(data + 3 / 8)


def anscombe_inv(data):
    """Apply inverse Anscombe transform to data

    https://en.wikipedia.org/wiki/Anscombe_transform
    """
    part0 = 1 / 4 * data ** 2
    part1 = 1 / 4 * np.sqrt(3 / 2) / data
    part2 = -11 / 8 / (data ** 2)
    part3 = 5 / 8 * np.sqrt(3 / 2) / (data ** 3)
    return part0 + part1 + part2 + part3 - 1 / 8


def fft_gaussian_filter(img, sigma):
    """FFT gaussian convolution

    Parameters
    ----------
    img : ndarray
        Image to convolve with a gaussian kernel
    sigma : int or sequence
        The sigma(s) of the gaussian kernel in _real space_

    Returns
    -------
    filt_img : ndarray
        The filtered image
    """
    # This doesn't help agreement but it will make things faster
    # pull the shape
    s1 = np.array(img.shape)
    # s2 = np.array([int(s * 4) for s in _normalize_sequence(sigma, img.ndim)])
    shape = s1  # + s2 - 1
    # calculate a nice shape
    fshape = [next_fast_len(int(d)) for d in shape]
    # pad out with reflection
    pad_img = fft_pad(img, fshape, "reflect")
    # calculate the padding
    padding = tuple(_calc_pad(o, n) for o, n in zip(img.shape, pad_img.shape))
    # so that we can calculate the cropping, maybe this should be integrated
    # into `fft_pad` ...
    fslice = tuple(slice(s, -e) if e != 0 else slice(s, None) for s, e in padding)
    # fourier transfrom and apply the filter
    kimg = rfftn(pad_img, fshape)
    filt_kimg = fourier_gaussian(kimg, sigma, pad_img.shape[-1])
    # inverse FFT and return.
    return irfftn(filt_kimg, fshape)[fslice]


def multi_exp(xdata, *args):
    """Power and exponent"""
    odd = len(args) % 2
    if odd:
        offset = args[-1]
    else:
        offset = 0
    res = np.ones_like(xdata, dtype=float) * offset
    for i in range(0, len(args) - odd, 2):
        a, k = args[i : i + 2]
        res += a * np.exp(-k * xdata)
    return res


def multi_exp_jac(xdata, *args):
    """Power and exponent jacobian"""
    odd = len(args) % 2

    tostack = []

    for i in range(0, len(args) - odd, 2):
        a, k = args[i : i + 2]
        tostack.append(np.exp(-k * xdata))
        tostack.append(-a * xdata * tostack[-1])

    if odd:
        # there's an offset
        tostack.append(np.ones_like(xdata))

    return np.vstack(tostack).T


def exponent(xdata, amp, rate, offset):
    """Utility function to fit nonlinearly"""
    return multi_exp(xdata, amp, rate, offset)


def _estimate_exponent_params(data, xdata):
    """utility to estimate exponent params"""
    assert np.isfinite(data).all(), "data is not finite"
    assert np.isfinite(xdata).all(), "xdata is not finite"
    assert len(data) == len(xdata), "Lengths don't match"
    assert len(data), "there is no data"
    if data[0] >= data[-1]:
        # decay
        offset = np.nanmin(data)
        data_corr = data - offset
        log_data_corr = np.log(data_corr)
        valid_pnts = np.isfinite(log_data_corr)
        m, b = np.polyfit(xdata[valid_pnts], log_data_corr[valid_pnts], 1)
        return np.nan_to_num((np.exp(b), -m, offset))
    else:
        amp, rate, offset = _estimate_exponent_params(-data, xdata)
        return np.array((-amp, rate, -offset))


def _estimate_components(data, xdata):
    """"""
    raise NotImplementedError


def exponent_fit(data, xdata=None, offset=True):
    """Utility function that fits data to the exponent function"""
    return multi_exp_fit(data, xdata, components=1, offset=offset)


def multi_exp_fit(data, xdata=None, components=None, offset=True, **kwargs):
    """Utility function that fits data to the exponent function

    Assumes evenaly spaced data.

    Parameters
    ----------
    data : ndarray (1d)
        data that can be modeled as a single exponential decay
    xdata : numeric
        x axis for fitting

    Returns
    -------
    popt : ndarray
        optimized parameters for the exponent wave
        (a0, k0, a1, k1, ... , an, kn, offset)
    pcov : ndarray
        covariance of optimized paramters


    label_base = "$y(t) = " + "{:+.3f} e^{{-{:.3g}t}}" * (len(popt) // 2) + " {:+.0f}$" * (len(popt) % 2)
    """
    # only deal with finite data
    # NOTE: could use masked wave here.
    if xdata is None:
        xdata = np.arange(len(data))

    if components is None:
        components = _estimate_components(data, xdata)

    finite_pnts = np.isfinite(data)
    data_fixed = data[finite_pnts]
    xdata_fixed = xdata[finite_pnts]
    # we need at least 4 data points to fit
    if len(data_fixed) > 3:
        # we can't fit data with less than 4 points
        # make guesses
        if components > 1:
            split_points = np.logspace(
                np.log(xdata_fixed[xdata_fixed > 0].min()),
                np.log(xdata_fixed.max()),
                components + 1,
                base=np.e,
            )
            # convert to indices
            split_idxs = np.searchsorted(xdata_fixed, split_points)
            # add endpoints, make sure we don't have 0 twice
            split_idxs = [None] + list(split_idxs[1:-1]) + [None]
            ranges = [slice(start, stop) for start, stop in zip(split_idxs[:-1], split_idxs[1:])]
        else:
            ranges = [slice(None)]
        pguesses = [_estimate_exponent_params(data_fixed[s], xdata_fixed[s]) for s in ranges]
        # clear out the offsets
        pguesses = [pguess[:-1] for pguess in pguesses[:-1]] + pguesses[-1:]
        # add them together
        pguess = np.concatenate(pguesses)
        if not offset:
            # kill the offset component
            pguess = pguess[:-1]
        # The jacobian actually slows down the fitting my guess is there
        # aren't generally enough points to make it worthwhile
        return curve_fit(
            multi_exp, xdata_fixed, data_fixed, p0=pguess, jac=multi_exp_jac, **kwargs
        )
    else:
        raise RuntimeError("Not enough good points to fit.")


def estimate_power_law(x, y, diagnostics=False):
    """Estimate the best fit parameters for a power law by linearly fitting the loglog plot"""
    # can't take log of negative points
    valid_points = y > 0
    # pull valid points and take log
    xx = np.log(x[valid_points])
    yy = np.log(y[valid_points])
    # weight by sqrt of value, make sure we get the trend right
    w = np.sqrt(y[valid_points])
    # fit line to loglog
    neg_b, loga = np.polyfit(np.log(x[valid_points]), np.log(y[valid_points]), 1, w=w)
    if diagnostics:
        plt.loglog(x[valid_points], y[valid_points])
        plt.loglog(x, np.exp(loga) * x ** (neg_b))
    return np.exp(loga), -neg_b


def _test_pow_law(popt, xmin):
    """Utility function for testing power law params"""
    a, b = popt
    assert a > 0, "Scale invalid"
    assert b > 1, "Exponent invalid"
    assert xmin > 0, "xmin invalid"


def power_percentile(p, popt, xmin=1):
    """Percentile of a single power law function"""
    assert 0 <= p <= 1, "percentile invalid"
    _test_pow_law(popt, xmin)
    a, b = popt
    x0 = (1 - p) ** (1 / (1 - b)) * xmin
    return x0


def power_percentile_inv(x0, popt, xmin=1):
    """Given an x value what percentile of the power law function does it correspond to"""
    _test_pow_law(popt, xmin)
    a, b = popt
    p = 1 - (x0 / xmin) ** (1 - b)
    return p


def power_intercept(popt, value=1):
    """At what x value does the function reach value"""
    a, b = popt
    assert a > 0, f"a = {value}"
    assert value > 0, f"value = {value}"
    return (a / value) ** (1 / b)


def power_law(xdata, *args):
    """An multi-power law function"""
    odd = len(args) % 2
    if odd:
        offset = float(args[-1])
    else:
        offset = 0.0
    res = np.ones_like(xdata) * offset
    lx = np.log(xdata)
    for i in range(0, len(args) - odd, 2):
        res += args[i] * np.exp(-args[i + 1] * lx)
    return res


def power_law_jac(xdata, *args):
    """Jacobian for a multi-power law function"""
    odd = len(args) % 2
    tostack = []
    lx = np.log(xdata)
    for i in range(0, len(args) - odd, 2):
        a, b = args[i : i + 2]
        # dydai
        tostack.append(np.exp(-b * lx))
        # dydki
        tostack.append(-lx * a * tostack[-1])

    if odd:
        # there's an offset
        tostack.append(np.ones_like(xdata))

    return np.vstack(tostack).T


def get_tif_urls(baseurl):
    """Return all the links from a webpage with .tif"""
    r = requests.get(baseurl)
    links = re.findall('"([^"]*\.tif)"', r.content.decode())
    head = "/".join(r.url.split("/")[:3])
    return [head + l for l in links]


def url_tifread(url):
    """Read a tif into memory from a url"""
    r = requests.get(url)
    return tif.imread(io.BytesIO(r.content))


def imread_bioformats(path):
    """Thin wrapper around bioformats, slow and shitty,
    but works in a pinch, only reads z stack -- no time"""
    # import modules tp ise
    import gc
    import itertools

    import bioformats
    import javabridge

    # start the jave VM with the appropriate classes loaded
    javabridge.start_vm(class_path=bioformats.JARS)
    # init the reader
    with bioformats.ImageReader(path) as reader:
        # init container
        data = []
        for i in itertool.count():
            try:
                data.append(reader.read(z=i, rescale=False))
            except javabridge.JavaException:
                # reached end of file, break
                break
    # clean up a little
    javabridge.kill_vm()
    gc.collect()
    # return ndarray
    return np.asarray(data)


def powerlaw_prng(alpha, xmin=1, xmax=1e7):
    """Function to calculate a psuedo random variable drawn from a discrete power law distribution
    with scale parameter alpha and xmin"""

    # don't want to waste time recalculating this
    bottom = zeta(alpha, xmin)

    def P(x):
        """Cumulative distribution function"""
        try:
            return zeta(alpha, x) / bottom
        except TypeError as e:
            print(alpha, x, r)
            raise e

    # maximum r
    rmax = 1 - P(xmax)
    r = 1

    # keep trying until we get one in range
    while r > rmax:
        r = np.random.random()

    # find bracket in log 2 space
    x = xnew = xmin
    while P(x) >= 1 - r:
        x *= 2
    x1 = x / 2
    x2 = x

    # binary search
    while x2 - x1 > 1:
        xnew = (x2 + x1) / 2
        if P(xnew) >= 1 - r:
            x1 = xnew
        else:
            x2 = xnew

    # return bottom
    return int(x1)


class PowerLaw(object):
    """A class for fitting and testing power law distributions"""

    def __init__(self, data):
        """Pass in data, it will be automagically determined to be
        continuous (float/inexact datatype) or discrete (integer datatype)"""
        self.data = data
        # am I discrete data
        self._discrete = np.issubdtype(data.dtype, np.integer)

    def fit(self, xmin=None, xmin_max=200, opt_max=False):
        """Fit the data, if xmin is none then estimate it"""
        if self._discrete:
            # discrete fitting
            if opt_max:
                # we should optimize the maximum x
                def func(m):
                    """an optimization function, m is max value"""
                    test_data = self.data[self.data < m]
                    power_sub = PowerLaw(test_data)
                    power_sub.fit(xmin=None, xmin_max=xmin_max, opt_max=False)
                    # copy params to main object
                    self.C, self.alpha, self.xmin = power_sub.C, power_sub.alpha, power_sub.xmin
                    self.ks_statistics = power_sub.ks_statistics
                    return power_sub.ks_statistics.min()

                opt = minimize_scalar(
                    func, bounds=(2 * xmin_max, self.data.max()), method="bounded"
                )

                if not opt.success:
                    raise RuntimeError("Optimal xmax not found.")

                self.xmax = int(opt.x)

            elif xmin is None:
                # this is a hacky way of doing things, just
                # try fitting multiple xmin
                args = [
                    (self._fit_discrete(x), x) for x in range(1, min(xmin_max, self.data.max()))
                ]

                # utility function to test KS
                def KS_test(alpha, xmin):
                    """Update self then run KS_test"""
                    # set internal variables
                    self.alpha = alpha
                    self.xmin = xmin
                    # generate statistic
                    ks = self._KS_test_discrete()
                    return ks

                # generate list of statistics
                self.ks_statistics = np.array([KS_test(arg[0][1], arg[1]) for arg in args])
                # we want to minimize the distance in KS space
                best_arg = args[self.ks_statistics.argmin()]
                # set internals
                (self.C, self.alpha), self.xmin = best_arg
                # estimate error (only valid for large n)
                self.alpha_error = (self.alpha - 1) / np.sqrt(
                    len(self.data[self.data >= self.xmin])
                )
            else:
                self._fit_discrete(xmin)
                self.ks_statistics = np.array([self._KS_test_discrete()])
        else:
            if xmin is None:
                xmin = 1
            return self._fit_continuous(xmin)

        return self.C, self.alpha

    @property
    def clipped_data(self):
        """Return data clipped to xmin"""
        return self.data[self.data >= self.xmin]

    def intercept(self, value=1):
        """Return the intercept calculated from power law values"""
        return power_intercept((self.C * len(self.data), self.alpha), value)

    def percentile(self, value):
        """Return the intercept calculated from power law values"""
        return power_percentile(value, (self.C * len(self.data), self.alpha), self.xmin)

    def gen_power_law(self):
        """x.append(xmin*pow(1.-random(),-1./(alpha-1.)))"""
        clipped_data = self.clipped_data

        # approximate
        # fake_data = (self.xmin - 0.5) * (1. - np.random.random(len(clipped_data) * 2)) ** (-1. / (self.alpha - 1.)) + 0.5
        # fake_data = fake_data[fake_data <= clipped_data.max()]

        # fake_data = np.rint(fake_data[:len(clipped_data)]).astype(clipped_data.dtype)

        # poisson
        N = len(clipped_data)
        xmax = clipped_data.max()

        x = np.arange(self.xmin, xmax + 1)

        fake_counts = np.random.poisson(self._power_law_fit_discrete(x) * N)
        fake_data = np.repeat(x, fake_counts)
        fake_data = np.random.choice(fake_data, N)

        # dmax = clipped_data.max()
        # fake_data = np.array([powerlaw_prng(self.alpha, self.xmin, dmax) for i in range(len(clipped_data))])

        return np.asarray(fake_data)

    def calculate_p(self, num=1000):
        """Make a bunch of fake data and run the KS_test on it"""
        ks_data = self._KS_test_discrete()
        # normalizing constant
        ks_tests = []
        for i in tqdm.trange(num):
            fake_data = self.gen_power_law()
            power = PowerLaw(fake_data)
            power.fit(self.xmin)
            ks_tests.append(power._KS_test_discrete())

        ks_tests = np.asarray(ks_tests)
        self.ks_tests = ks_tests
        self.ks_data = ks_data
        return (ks_tests > ks_data).sum() / len(ks_tests)

    def _convert_to_probability_discrete(self):
        """"""
        y = np.bincount(self.data)
        x = np.arange(len(y))

        # calculate the normalization constant from xmin onwards
        N = y[self.xmin :].sum()
        y = y / N
        return x, y, N

    def _power_law_fit_discrete(self, x):
        """Compute the power_law fit"""
        return x ** (-self.alpha) / zeta(self.alpha, self.xmin)

    def _KS_test_discrete(self):
        """Kolmogorov–Smirnov or KS statistic"""
        x, y, N = self._convert_to_probability_discrete()
        # clip at xmin
        x, y = x[self.xmin :], y[self.xmin :]
        assert np.allclose(y.sum(), 1), f"y not normalized {y}, xmin = {self.xmin}"
        # caculate the cumulative distribution functions
        expt_cdf = y.cumsum()
        power_law = self._power_law_fit_discrete(x)
        power_law_cdf = power_law.cumsum()

        return (abs(expt_cdf - power_law_cdf) / np.sqrt(power_law_cdf * (1 - power_law_cdf))).max()

    def _fit_discrete(self, xmin=1):
        """Fit a discrete power-law to data"""
        # update internal xmin
        self.xmin = xmin

        # clip data to be greater than xmin
        data = self.clipped_data

        # calculate the log sum of the data
        lsum = np.log(data).sum()
        # and length, don't want to recalculate these during optimization
        n = len(data)

        def nll(alpha, xmin):
            """negative log-likelihood of discrete power law"""
            return n * np.log(zeta(alpha, xmin)) + alpha * lsum

        # find best result
        opt = minimize_scalar(nll, args=(xmin,))
        if not opt.success:
            raise RuntimeWarning("Optimization failed to converge")

        # calculate normalization constant
        alpha = opt.x
        C = 1 / zeta(alpha, xmin)

        # save in object
        self.C = C
        self.alpha = alpha

        return C, alpha

    def _fit_continuous(self, xmin=1):
        """Fit a continuous power-law to data"""
        data = self.data

        data = data[data >= xmin]

        alpha = 1 + len(data) / np.log(data / xmin).sum()
        C = (alpha - 1) * xmin ** (alpha - 1)

        self.C = C
        self.alpha = alpha
        self.alpha_std = (alpha - 1) / np.sqrt(len(data))

        return C, alpha

    def plot(self, ax=None, density=True, norm=False):
        """plot data"""
        x, y, N = self._convert_to_probability_discrete()

        x, y = x[1:], y[1:]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        # plot
        # calculate fit
        power_law = self._power_law_fit_discrete(x)

        ax.set_xlabel("Number of frames")
        if not density:
            y *= N
            power_law *= N
            ymin = 0.5
            ax.set_ylabel("Occurences (#)")
        elif norm:
            ymax = y[0]
            y /= ymax
            power_law /= ymax
            ymin = 0.5 * y.min()
            ax.set_ylabel("Fraction of Maximum")
        else:
            ymin = 0.5 / N
            ax.set_ylabel("Frequency")

        ax.loglog(x, y, ".", label="Data")
        ax.loglog(x, power_law, label=r"$\alpha = {:.2f}$".format(self.alpha))
        ax.set_ylim(bottom=ymin)

        ax.axvline(
            self.xmin,
            color="y",
            linewidth=4,
            alpha=0.5,
            label="$x_{{min}} = {}$".format(self.xmin),
        )

        try:
            ax.axvline(
                self.xmax,
                color="y",
                linewidth=4,
                alpha=0.5,
                label="$x_{{max}} = {}$".format(self.xmax),
            )
        except AttributeError:
            pass

        return fig, ax


def fit_ztp(data):
    """Fit the data assuming it follows a zero-truncated Poisson model"""
    n = len(data)
    sum_x = data.sum()
    # ignore the constant offset
    # sum_log_x_fac = np.log(gamma(data)).sum()

    def negloglikelihood(lam):
        """Negative log-likelihood of ZTP"""
        # ignore the constant offset
        return n * np.log(np.exp(lam) - 1) - np.log(lam) * sum_x  # + sum_log_x_fac

    with np.errstate(divide="ignore", invalid="ignore"):
        opt = minimize_scalar(negloglikelihood)

    if not opt.success:
        raise RuntimeError("Fitting zero-truncated poisson failed")

    return opt.x


def NegBinom(a, m):
    """convert scipy's definition to mean and shape"""
    r = a
    p = m / (m + r)
    return nbinom(r, 1 - p)


def negloglikelihoodNB(args, x):
    """Negative log likelihood for negative binomial"""
    a, m = args
    numerator = NegBinom(a, m).pmf(x)
    return -np.log(numerator).sum()


def negloglikelihoodZTNB(args, x):
    """Negative log likelihood for zero truncated negative binomial"""
    a, m = args
    denom = 1 - NegBinom(a, m).pmf(0)

    return len(x) * np.log(denom) + negloglikelihoodNB(args, x)


def fit_ztnb(data, x0=(0.5, 0.5)):
    """Fit the data assuming it follows a zero-truncated Negative Binomial model"""

    opt = minimize(negloglikelihoodZTNB, x0, (data,), bounds=((0, np.inf), (0, np.inf)))

    if not opt.success:
        raise RuntimeError("Fitting zero-truncated negative binomial", opt)

    return opt.x


def find_prime_facs(n):
    """Find the prime factors of n"""
    list_of_factors = []
    i = 2
    while n > 1:
        if n % i == 0:
            list_of_factors.append(i)
            n = n / i
            i = i - 1
        i += 1
    return np.array(list_of_factors)


def montage(stack):
    """Take a stack and a new shape and cread a montage"""
    # assume data is ordered as color, tiles, ny, nx
    ntiles, ny, nx = stack.shape[:3]
    # Find the prime factor that makes the montage most square
    primes = find_prime_facs(ntiles)
    dx = primes[::2].prod()
    dy = ntiles // dx
    new_shape = (dy, dx, ny, nx) + stack.shape[3:]
    # sanity check
    assert (
        dy * dx == ntiles
    ), f"Number of tiles, {ntiles}, doesn't match montage dimensions ({dy}, {dx})"
    # reshape the stack
    reshaped_stack = stack.reshape(new_shape)
    # align the tiles
    reshaped_stack = np.moveaxis(reshaped_stack, 1, 2)
    # merge and return.
    return reshaped_stack.reshape((dy * ny, dx * nx) + stack.shape[3:])


def square_montage(stack):
    """Turn a 3D stack into a square montage"""
    # calculate nearest square
    new_num = int(np.ceil(np.sqrt(len(stack))) ** 2)
    # if square return montage
    if new_num == len(stack):
        return montage(stack)
    # add enough zeros to make square
    return montage(
        np.concatenate(
            (stack, [np.zeros(stack.shape[1:], dtype=stack.dtype)] * (new_num - len(stack)))
        )
    )


def latex_format_e(num, pre=2):
    """Format a number for nice latex presentation, the number will *not* be enclosed in $"""
    s = ("{:." + "{:d}".format(pre) + "e}").format(num)
    fp, xp = s.split("e+")
    return "{} \\times 10^{{{}}}".format(fp, int(xp))


def amira_bbox_str(*, bbox=None, resolution=None, shape=None, offset=None):
    """Correctly calculates the bounding box and makes a string to be stuffed into
    the image description tag of a tiff"""
    if bbox is not None:
        z0, z1, y0, y1, x0, x1 = bbox
    else:
        z0, y0, x0 = offset
        z1, y1, x1 = np.asarray(resolution) * (np.asarray(shape) - 1) + offset

    fmt_str = "BoundingBox {x0:} {x1:} {y0:} {y1:} {z0:} {z1:}"
    fmt_dict = dict(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)
    return fmt_str.format(**fmt_dict)


def localize_peak(data):
    """Small utility function to localize a peak center. Assumes passed data has
    peak at center and that data.shape is odd and symmetric. Then fits a
    parabola through each line passing through the center. This is optimized
    for FFT data which has a non-circularly symmetric shaped peaks.
    """
    # make sure passed data is symmetric along all dimensions
    assert len(set(data.shape)) == 1, f"data.shape = {data.shape}"
    # pull center location
    center = data.shape[0] // 2
    # generate the fitting lines
    my_pat_fft_suby = data[:, center]
    my_pat_fft_subx = data[center, :]
    # fit along lines, consider the center to be 0
    x = np.arange(data.shape[0]) - center
    xfit = np.polyfit(x, my_pat_fft_subx, 2)
    yfit = np.polyfit(x, my_pat_fft_suby, 2)
    # calculate center of each parabola
    x0 = -xfit[1] / (2 * xfit[0])
    y0 = -yfit[1] / (2 * yfit[0])
    # NOTE: comments below may be useful later.
    # save fits as poly functions
    # ypoly = np.poly1d(yfit)
    # xpoly = np.poly1d(xfit)
    # peak_value = ypoly(y0) / ypoly(0) * xpoly(x0)
    # #
    # assert np.isclose(peak_value,
    #                   xpoly(x0) / xpoly(0) * ypoly(y0))
    # return center
    return y0, x0


def get_max(xdata, ydata, axis=0):
    """Get the x value that corresponds to the max y value"""
    idx_max = ydata.argmax(axis)
    max_x = np.take_along_axis(xdata, np.expand_dims(idx_max, axis), axis).squeeze()
    return max_x


def edf(stack):
    """Simple extended depth of focus, take the value with the max gradient"""
    img = stack.astype("float32")
    gradient_x = scipy.ndimage.sobel(img, 2)
    gradient_y = scipy.ndimage.sobel(img, 1)
    grad_img = gradient_x * gradient_x + gradient_y * gradient_y
    return get_max(stack, grad_img)
