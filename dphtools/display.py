#!/usr/bin/env python
# -*- coding: utf-8 -*-
# display.py
"""
Plotting utilities.

Copyright (c) 2021, David Hoffman
"""

import textwrap
from functools import partial

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.fft import rfftfreq, rfftn

from .utils import fft_gaussian_filter

# Topics: line, color, LineCollection, cmap, colorline, codex
"""
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
"""

# Data manipulation:


def make_segments(x, y):
    """Create list of line segments from x and y coordinates, in the correct format for LineCollection.
    
    Returns an array of the form numlines x (points per line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:


def colorline(
    x,
    y,
    z=None,
    cmap="inferno",
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
    ax=None,
    autoscale=True,
):
    """Plot a colored line with coordinates x and y.

    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    if not isinstance(cmap, Colormap):
        cmap = plt.get_cmap(cmap)

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    lc.set_capstyle("round")

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    if autoscale:
        ax.autoscale()

    return lc


def display_grid(
    data,
    showcontour=False,
    contourcolor="w",
    filter_size=None,
    figsize=3,
    auto=False,
    nrows=None,
    grid_aspect=None,
    sharex=False,
    sharey=False,
    **kwargs,
):
    """Display a dictionary of images in a nice grid.

    Parameters
    ----------
    data : dict
        a dictionary of images
    showcontour: bool (default, False)
        Whether to show contours or not
    """
    if not isinstance(data, dict):
        raise TypeError("Data is not a dictionary!")
    # figure out grid_aspect ratios of data (a = y / x)
    if grid_aspect is None:
        aspects = np.array([v.shape[0] / v.shape[1] for v in data.values() if v.ndim > 1])
        # if len is zero then everything was 1d
        if len(aspects):
            grid_aspect = aspects.mean()
            if not np.isfinite(grid_aspect):
                raise RuntimeError(f"grid_aspect isn't finite, grid_aspect = {grid_aspect}")
        else:
            grid_aspect = 1
    fig, axs = make_grid(
        len(data),
        nrows=nrows,
        figsize=figsize,
        grid_aspect=grid_aspect,
        sharex=sharex,
        sharey=sharey,
    )
    for (k, v), ax in zip(sorted(data.items()), axs.ravel()):
        if v.ndim == 1:
            ax.plot(v, **kwargs)
        else:
            if auto:
                # calculate vmin, vmax
                kwargs.update(auto_adjust(v))
            ax.matshow(v, **kwargs)
            if showcontour:
                if filter_size is None:
                    vv = v
                else:
                    vv = fft_gaussian_filter(v, filter_size)

                ax.contour(vv, colors=contourcolor)
        ax.set_title(k)
        ax.axis("off")

    clean_grid(fig, axs)

    return fig, axs


def wrap_name(dirname, figsize):
    """Wrap name to fit in subfig."""
    fontsize = plt.rcParams["font.size"]
    # 1/120 = inches/(fontsize*character)
    num_chars = int(figsize / fontsize * 72)
    return textwrap.fill(dirname, num_chars)


def make_grid(numitems, nrows=None, figsize=3, grid_aspect=1, **kwargs):
    """Make a grid of axes."""
    if numitems == 0:
        raise ValueError("numitems can't be zero.")
    if nrows is None:
        nrows = int(np.sqrt(numitems))
    if nrows == 0:
        nrows = ncols = 1
    else:
        ncols = int(np.ceil(numitems / nrows))

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize * ncols, figsize * nrows * grid_aspect),
        squeeze=False,
        **kwargs,
    )

    return fig, axs


def clean_grid(fig, axs):
    """Clean up a grid of axes by removing unused axes."""
    for ax in axs.ravel():
        if not (len(ax.images) or len(ax.lines) or len(ax.patches)):
            fig.delaxes(ax)
    return fig, axs


def take_slice(data, axis, midpoint=None):
    """Take slices."""
    if midpoint is None:
        midpoint = np.array(data.shape, dtype=np.int) // 2
    my_slice = [slice(None, None, None) for i in range(data.ndim)]
    my_slice[axis] = midpoint[axis]
    return data[tuple(my_slice)]


def slice_plot(data, center=None, allaxes=False, **kwargs):
    """Display slices through data at `center`."""
    take_slice2 = partial(take_slice, midpoint=center)
    return mip(data, func=take_slice2, allaxes=allaxes, **kwargs)


def recolor(cmap, ax=None, new_alpha=None, to_change="lines"):
    """Recolor the lines in ax with the cmap."""
    if isinstance(cmap, str):
        # user has passed a string
        # presumably the name of a registered color map
        cmap = plt.get_cmap(cmap)

    if ax is None:
        ax = plt.gca()
    # figure out how many lines are in ax
    objs = getattr(ax, to_change)
    num_objs = len(objs)
    # set the new alpha mapping, if wanted
    if new_alpha is not None:
        if "best" == new_alpha:
            r = 1 / num_objs
            try:
                expon = new_alpha["best"]
            except TypeError:
                expon = 2
            new_alpha = 1 - ((1 - np.sqrt(r)) / (1 + np.sqrt(r))) ** expon
    # cycle through colors and recolor lines
    for i, obj in enumerate(objs):
        # generate new color
        new_color = list(cmap(i / (num_objs - 1)))
        # replace alpha is wanted
        if new_alpha is not None:
            new_color[-1] = new_alpha
        # set the color
        obj.set_color(new_color)


def drift_plot(
    fit, title=None, dt=0.1, dx=130, lf=-np.inf, hf=np.inf, log=False, cmap="magma", xc="b", yc="r"
):
    """Show drift curves nicely.

    Parameters
    ----------
    fit : pandas DataFrame
        Assumes that it has attributes x0 and y0
    title : str (optional)
        Title of plot
    dt : float (optional)
        Sampling rate of data in seconds
    dx : pixel size (optional)
        Pixel size in nm
    lf : float (optional)
        Low frequency cutoff for fourier plot
    hf : float (optional)
        High frequency cutoff for fourier plot
    log : bool (optional)
        Take logarithm of FFT data before displaying
    cmap : string or matplotlib.colors.cmap instance
        Color map for scatter plot
    xc : string or `color` instance
        Color for x data
    yc : string or `color` instance
        Color for y data

    Returns
    -------
    fig : figure object
        The figure
    axs : tuple of axes objects
        In the following order, Real axis, FFT axis, Scatter axis
    """
    # set up plot
    fig = plt.figure()
    fig.set_size_inches(8, 4)
    axreal = plt.subplot(221)
    axfft = plt.subplot(223)
    axscatter = plt.subplot(122)
    # label it
    if title is not None:
        fig.suptitle(title, y=1.02, fontweight="bold")
    # detrend mean
    ybar = fit.y0 - fit.y0.mean()
    ybar *= dx
    xbar = fit.x0 - fit.x0.mean()
    xbar *= dx
    # Plot Real space
    t = np.arange(len(fit)) * dt
    axreal.plot(t, xbar, xc, label=r"$x_0$")
    axreal.plot(t, ybar, yc, label=r"$y_0$")
    axreal.set_xlabel("Time (s)")
    axreal.set_ylabel("Displacement (nm)")
    # add legend to real axis
    axreal.legend(loc="best")
    # calc FFTs
    Y = rfftn(ybar)
    X = rfftn(xbar)
    # calc FFT freq
    k = rfftfreq(len(fit), dt)
    # limit FFT display range
    kg = np.logical_and(k > lf, k < hf)
    # Plot FFT
    if log:
        axfft.semilogy(k[kg], abs(X[kg]), xc)
        axfft.semilogy(k[kg], abs(Y[kg]), yc)
    else:
        axfft.plot(k[kg], abs(X[kg]), xc)
        axfft.plot(k[kg], abs(Y[kg]), yc)
    axfft.set_xlabel("Frequency (Hz)")
    # Plot scatter
    axscatter.scatter(xbar, ybar, c=t, cmap=cmap, ec="k")
    axscatter.set_xlabel("x")
    axscatter.set_ylabel("y")
    # make sure the scatter plot is square
    lims = axreal.get_ylim()
    axscatter.set_ylim(lims)
    axscatter.set_xlim(lims)
    axscatter.set_aspect(1)
    # tight layout
    axs = (axreal, axfft, axscatter)
    fig.tight_layout()
    # return fig, axs to user for further manipulation and/or saving if wanted
    return fig, axs


def mip(data, zaspect=1, func=np.amax, allaxes=False, plt_kwds=None, **kwargs):
    """Plot max projection of data.

    Parameters
    ----------
    data : 2 or 3 dimensional ndarray
        the data to be plotted
    func :  callable
        a function to be called on the data, must accept and axes argument
    allaxes : bool
        whether to return all axes or not
    plt_kwds : dict
        A dictionary of keywords for the plots (2D case)
    kwargs : dict
        passed to matshow

    Returns
    -------
    fig : mpl figure instanse
        figure handle
    axs : ndarray of axes objects
        axes handles in a flat ndarray
    """
    # set default properly for dict
    if plt_kwds is None:
        plt_kwds = {}
    # pull shape and dimensions
    myshape = data.shape
    ndim = data.ndim
    # set up the grid for the subplots
    if ndim == 3:
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[1, myshape[0] * zaspect / myshape[2]],
            height_ratios=[1, myshape[0] * zaspect / myshape[1]],
        )
        # set up my canvas necessary to make the overall figure shape square,
        # without this the boxes aren't sized properly
        fig = plt.figure(figsize=(5 * myshape[2] / myshape[1], 5))
    elif ndim == 2:
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
        fig = plt.figure(figsize=(5, 5))
    else:
        raise TypeError(f"Data has too many dimensions, ndim = {ndim}")
    # set up each projection
    ax_xy = plt.subplot(gs[0])
    ax_xy.set_title("XY")
    # set up YZ
    ax_yz = plt.subplot(gs[1], sharey=ax_xy)
    ax_yz.set_title("YZ")
    # set up XZ
    ax_xz = plt.subplot(gs[2], sharex=ax_xy)
    ax_xz.set_title("XZ")
    # actually calc data and plot
    if ndim == 3:
        max_z = func(data, axis=0)
        ax_xy.matshow(max_z, **kwargs)
        max_y = func(data, axis=1)
        ax_xz.matshow(max_y, aspect=zaspect, **kwargs)
        max_x = func(data, axis=2)
        ax_yz.matshow(max_x.T, aspect=zaspect, **kwargs)
    else:
        max_z = data.copy()
        ax_xy.matshow(max_z, **kwargs)
        max_x = func(data, axis=1)
        ax_yz.plot(max_x, np.arange(myshape[0]), **plt_kwds)
        max_y = func(data, axis=0)
        ax_xz.plot(max_y, **plt_kwds)
    for ax in (ax_xy, ax_xz, ax_yz):
        ax.axis("off")
    # make all axis tight
    for ax in (ax_xy, ax_yz, ax_xz):
        ax.axis("tight")
    fig.tight_layout()
    # if the user requests all axes return them
    if allaxes:
        return fig, np.array([ax_xy, ax_yz, ax_xz, plt.subplot(gs[3])])
    else:
        return fig, np.array([ax_xy, ax_yz, ax_xz])


def auto_adjust(img):
    """Python translation of ImageJ autoadjust function.

    Parameters
    ----------
    img : ndarray

    Returns
    -------
    (vmin, vmax) : tuple of numbers
    """
    # calc statistics
    pixel_count = int(np.array((img.shape)).prod())
    # get image statistics
    # ImageStatistics stats = imp.getStatistics()
    # initialize limit
    limit = pixel_count / 10
    # histogram
    try:
        my_hist, bins = np.histogram(np.nan_to_num(img).ravel(), bins="auto")
        if len(bins) < 100:
            my_hist, bins = np.histogram(np.nan_to_num(img).ravel(), bins=128)
        # convert bin edges to bin centers
        bins = np.diff(bins) + bins[:-1]
        nbins = len(my_hist)
        # set up the threshold
        # Below is what ImageJ purportedly does.
        # auto_threshold = threshold_isodata(img, nbins=bins)
        # if auto_threshold < 10:
        #     auto_threshold = 5000
        # else:
        #     auto_threshold /= 2
        # this version, below, seems to converge as nbins increases.
        threshold = pixel_count / (nbins * 16)
        # find the minimum by iterating through the histogram
        # which has 256 bins
        valid_bins = bins[np.logical_and(my_hist < limit, my_hist > threshold)]
        # check if the found limits are valid.
        vmin = valid_bins[0]
        vmax = valid_bins[-1]
    except IndexError:
        vmin = 0
        vmax = 0

    if vmax <= vmin:
        vmin = img.min()
        vmax = img.max()

    return dict(vmin=vmin, vmax=vmax)


# @np.vectorize
def wavelength_to_rgb(wavelength, gamma=0.8):
    """Convert a given wavelength of light to an approximate RGB color value.
    
    The wavelength must be given in nanometers in the range from 380 nm through 750 nm (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    """
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B)


def add_scalebar(
    ax: mpl.axes.Axes,
    scalebar_size: float,
    pixel_size: float,
    unit: str = "µm",
    edgecolor: str = None,
    **kwargs,
) -> None:
    """Add a scalebar to the axis."""
    # NOTE: this is to be moved to dphtools when the package is ready
    scalebar_length = scalebar_size / pixel_size
    default_scale_bar_kwargs = dict(
        loc="lower right",
        pad=0.5,
        color="white",
        frameon=False,
        size_vertical=scalebar_length / 10,
        fontproperties=fm.FontProperties(weight="bold"),
    )
    default_scale_bar_kwargs.update(kwargs)
    if unit is not None:
        label = f"{scalebar_size} {unit}"
    else:
        label = ""
        if "lower" in default_scale_bar_kwargs["loc"]:
            default_scale_bar_kwargs["label_top"] = True
    scalebar = AnchoredSizeBar(ax.transData, scalebar_length, label, **default_scale_bar_kwargs)
    if edgecolor:
        scalebar.size_bar.get_children()[0].set_edgecolor(edgecolor)
        scalebar.txt_label.get_children()[0].set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground=edgecolor), path_effects.Normal()]
        )
    # add the scalebar
    ax.add_artist(scalebar)


class SymPowerNorm(Normalize):
    """Linearly map a given value to the 0-1 range and then apply a power-law normalization over that range."""

    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        """Initialize SymPowerNorm scaling."""
        Normalize.__init__(self, vmin, vmax, clip)
        self.gamma = gamma

    def _transform(self, value):
        return np.sign(value) * np.abs(value) ** self.gamma

    def _transform_inv(self, value):
        return np.sign(value) * np.abs(value) ** (1 / self.gamma)

    def __call__(self, value, clip=None):
        """Do scaling."""
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)
            resdat = result.data
            resdat = self._transform(resdat)
            vmin = self._transform(vmin)
            vmax = self._transform(vmax)
            resdat = (resdat - vmin) / (vmax - vmin)

            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        """Invert scale."""
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax

        vmin = self._transform(vmin)
        vmax = self._transform(vmax)

        if cbook.iterable(value):
            val = np.ma.asarray(value)
            return self._transform_inv(val * (vmax - vmin) + vmin)
        else:
            return self._transform_inv(value * (vmax - vmin) + vmin)

    def autoscale(self, A):
        """Set *vmin*, *vmax* to min, max of *A*."""
        self.vmin = np.ma.min(A)
        self.vmax = np.ma.max(A)

    def autoscale_None(self, A):
        """Autoscale only None-valued vmin or vmax."""
        A = np.asanyarray(A)
        if self.vmin is None and A.size:
            self.vmin = A.min()
        if self.vmax is None and A.size:
            self.vmax = A.max()


def hist_and_cumulative(data, ax=None, log=False):
    """Make a plot with both a histogram and cumulative distribution."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if log:
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 64)
    else:
        bins = "auto"
    ax.hist(data, bins=bins, density=True, log=False, histtype="step")
    ax.set_ylabel("Probability Density Function (PDF)")
    if log:
        ax.set_xscale("log")

    sorted_data = np.sort(data)
    N = len(sorted_data)
    b = np.arange(N) / N

    twin_ax = ax.twinx()
    color = ax._get_lines.get_next_color()
    twin_ax.plot(sorted_data, b, color=color, ls="steps-mid")
    twin_ax.tick_params(axis="y", labelcolor=color)
    twin_ax.set_ylabel("Cumulative Distribution Function (CDF)", color=color)
    twin_ax.set_ylim(bottom=0)

    return fig, (ax, twin_ax)


def make_rec(y, x, width, height, linewidth):
    """Make a rectangle of width and height _centered_ on (y, x)."""
    return plt.Rectangle(
        (x - width / 2, y - height / 2), width, height, color="w", linewidth=linewidth, fill=False
    )


def make_rec_from_slice(yxslice, **kwargs):
    """Make a rectangle of width and height _centered_ on (y, x)."""
    default_kwargs = dict(color="w", linewidth=1, fill=False)
    default_kwargs.update(kwargs)

    yslice, xslice = yxslice
    xy = (xslice.start, yslice.start)

    height = yslice.stop - yslice.start
    width = xslice.stop - xslice.start

    return plt.Rectangle(xy, width, height, **default_kwargs)
