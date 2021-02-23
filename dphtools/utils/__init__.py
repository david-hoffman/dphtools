#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
Various utility functions to be organized better.

Copyright (c) 2021, David Hoffman
"""

import logging
import os
import subprocess

import numpy as np
import scipy
from numpy.fft import ifftshift, irfftn, rfftn
from scipy.fft import next_fast_len
from scipy.ndimage._ni_support import _normalize_sequence
from scipy.ndimage.fourier import fourier_gaussian
import scipy.signal

logger = logging.getLogger(__name__)

eps = np.finfo(float).eps


def get_git(path="."):
    """Get git description."""
    try:
        # we slice to remove trailing new line.
        cmd = ["git", "--git-dir=" + os.path.join(path, ".git"), "describe", "--long", "--always"]
        return subprocess.check_output(cmd).decode()[:-1]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(e)
        logger.error(" ".join(cmd))
        return "Unknown"


def bin_ndarray(ndarray, new_shape=None, bin_size=None, operation="sum"):
    """Bins an ndarray in all axes based on the target shape, by summing or averaging.

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
    """Scale data to [0.0, 1.0] range, unless an integer dtype is specified in which case the data is scaled to fill the bit depth of the dtype.

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
    """Scale data to the uint16 range."""
    return scale(data, np.uint16)


def radial_profile(data, center=None, binsize=1.0):
    """Take the radial average of a 2D data array.

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
    (array([1., 1., 1., 1., 1., 1., 1., 1.]), array([0., 0., 0., 0., 0., 0., 0., 0.]))
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
    r = np.round(r / binsize).astype(int)
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
    """Get mode of non-negative integer data.

    up to 1000 times faster than scipy mode
    but not nearly as feature rich

    Note: we can vectorize this to work on different
    axes with numba
    """
    # will not work with negative numbers (for now)
    return np.bincount(data.ravel()).argmax()


def slice_maker(xs, ws):
    """Generate a tuple of slices to cut out a sub-array centered on `xs` with widths `ws`.

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
    >>> slice_maker((30, 20), 10)
    (slice(25, 35, None), slice(15, 25, None))
    >>> slice_maker((30, 20), 25)
    (slice(18, 43, None), slice(8, 33, None))
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
    """Pad an array to prep it for FFT."""
    # pull the old shape
    oldshape = array.shape
    if newshape is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(next_fast_len(n) for n in oldshape)
    else:
        if hasattr(newshape, "__iter__"):
            # are we iterable?
            newshape = tuple(newshape)
        elif isinstance(newshape, int) or np.issubdtype(newshape, np.integer):
            # test for regular python int, then numpy ints
            newshape = tuple(newshape for n in oldshape)
        else:
            raise ValueError(f"{newshape} is not a recognized shape")
    # generate padding and slices
    padding, slices = padding_slices(oldshape, newshape)
    return np.pad(array[slices], padding, mode=mode, **kwargs)


def padding_slices(oldshape, newshape):
    """Calculate the required padding or cropping based on the old shape and the new shape.

    Can be used to generate the slices needed to undo fft_pad above
    """
    # generate pad widths from new shape
    padding = tuple(
        _calc_pad(o, n) if n is not None else _calc_pad(o, o) for o, n in zip(oldshape, newshape)
    )
    # Make a crop list, if any of the padding is negative
    slices = tuple(_calc_crop(s1, s2) for s1, s2 in padding)
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return padding, slices


# add np.pad docstring
fft_pad.__doc__ += np.pad.__doc__


def _calc_crop(s1, s2):
    """Calc the cropping from the padding."""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def _calc_pad(oldnum, newnum):
    """Calculate the proper padding for fft_pad.

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


def fftconvolve_fast(data, kernel, **kwargs):
    """FFT convolution, a faster version than scipy.

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


def win_nd(size, win_func=scipy.signal.hann, **kwargs):
    """Make a multidimensional version of a window function.

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
    """Apply Anscombe transform to data.

    https://en.wikipedia.org/wiki/Anscombe_transform
    """
    return 2 * np.sqrt(data + 3 / 8)


def anscombe_inv(data):
    """Apply inverse Anscombe transform to data.

    https://en.wikipedia.org/wiki/Anscombe_transform
    """
    part0 = 1 / 4 * data ** 2
    part1 = 1 / 4 * np.sqrt(3 / 2) / data
    part2 = -11 / 8 / (data ** 2)
    part3 = 5 / 8 * np.sqrt(3 / 2) / (data ** 3)
    return part0 + part1 + part2 + part3 - 1 / 8


def fft_gaussian_filter(img, sigma):
    """FFT gaussian convolution.

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


def find_prime_facs(n):
    """Find the prime factors of n."""
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
    """Take a stack and a new shape and cread a montage."""
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
    """Turn a 3D stack into a square montage."""
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
    """Format a number for nice latex presentation, the number will *not* be enclosed in "$"."""
    s = ("{:." + "{:d}".format(pre) + "e}").format(num)
    fp, xp = s.split("e+")
    return "{} \\times 10^{{{}}}".format(fp, int(xp))


def localize_peak(data):
    """Small utility function to localize a peak center.
    
    Assumes passed data has peak at center and that data.shape is odd and symmetric.
    Then fits a parabola through each line passing through the center. This is optimized
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
    """Get the x value that corresponds to the max y value."""
    idx_max = ydata.argmax(axis)
    max_x = np.take_along_axis(xdata, np.expand_dims(idx_max, axis), axis).squeeze()
    return max_x


def edf(stack):
    """Calculate extended depth of focus, simple algo, take the value with the max gradient."""
    img = stack.astype("float32")
    gradient_x = scipy.ndimage.sobel(img, 2)
    gradient_y = scipy.ndimage.sobel(img, 1)
    grad_img = gradient_x * gradient_x + gradient_y * gradient_y
    return get_max(stack, grad_img)
