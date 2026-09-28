# Public Python interface evidence

**Version 1.0** Extracted public signatures and docstrings for setup test intake. These document existing intent; they do not certify implementation. Source bodies, private helpers, and history are omitted. Original doctest repr strings may predate NumPy 2.


## dphtools/__init__.py

Tools for optics and image analysis.

Copyright (c) 2021, David Hoffman


## dphtools/display.py

Plotting utilities.

Copyright (c) 2021, David Hoffman


### make_segments
```python
make_segments(x, y)
```
Create list of line segments from x and y coordinates, in the correct format for LineCollection.

Returns an array of the form numlines x (points per line) x 2 (x and y) array


### colorline
```python
colorline(x, y, z=None, cmap='inferno', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, ax=None, autoscale=True)
```
Plot a colored line with coordinates x and y.

Optionally specify colors in the array z
Optionally specify a colormap, a norm function and a line width


### display_grid
```python
display_grid(data, showcontour=False, contourcolor='w', filter_size=None, figsize=3, auto=False, nrows=None, grid_aspect=None, sharex=False, sharey=False, **kwargs)
```
Display a dictionary of images in a nice grid.

Parameters
----------
data : dict
    a dictionary of images
showcontour: bool (default, False)
    Whether to show contours or not


### wrap_name
```python
wrap_name(dirname, figsize)
```
Wrap name to fit in subfig.


### make_grid
```python
make_grid(numitems, nrows=None, figsize=3, grid_aspect=1, **kwargs)
```
Make a grid of axes.


### clean_grid
```python
clean_grid(fig, axs)
```
Clean up a grid of axes by removing unused axes.


### take_slice
```python
take_slice(data, axis, midpoint=None)
```
Take slices.


### slice_plot
```python
slice_plot(data, center=None, allaxes=False, **kwargs)
```
Display slices through data at `center`.


### recolor
```python
recolor(cmap, ax=None, new_alpha=None, to_change='lines')
```
Recolor the lines in ax with the cmap.


### drift_plot
```python
drift_plot(fit, title=None, dt=0.1, dx=130, lf=-np.inf, hf=np.inf, log=False, cmap='magma', xc='b', yc='r')
```
Show drift curves nicely.

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


### mip
```python
mip(data, zaspect=1, func=np.amax, allaxes=False, plt_kwds=None, **kwargs)
```
Plot max projection of data.

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


### auto_adjust
```python
auto_adjust(img)
```
Python translation of ImageJ autoadjust function.

Parameters
----------
img : ndarray

Returns
-------
(vmin, vmax) : tuple of numbers


### wavelength_to_rgb
```python
wavelength_to_rgb(wavelength, gamma=0.8)
```
Convert a given wavelength of light to an approximate RGB color value.

The wavelength must be given in nanometers in the range from 380 nm through 750 nm (789 THz through 400 THz).

Based on code by Dan Bruton
http://www.physics.sfasu.edu/astro/color/spectra.html


### add_scalebar
```python
add_scalebar(ax: mpl.axes.Axes, scalebar_size: float, pixel_size: float, unit: str='µm', edgecolor: Optional[str]=None, **kwargs)
```
Add a scalebar to the axis.


### SymPowerNorm
Linearly map a given value to the 0-1 range and then apply a power-law normalization over that range.


#### SymPowerNorm.__init__
```python
__init__(self, gamma, vmin=None, vmax=None, clip=False)
```
Initialize SymPowerNorm scaling.


#### SymPowerNorm.__call__
```python
__call__(self, value, clip=None)
```
Do scaling.


#### SymPowerNorm.inverse
```python
inverse(self, value)
```
Invert scale.


#### SymPowerNorm.autoscale
```python
autoscale(self, A)
```
Set *vmin*, *vmax* to min, max of *A*.


#### SymPowerNorm.autoscale_None
```python
autoscale_None(self, A)
```
Autoscale only None-valued vmin or vmax.


### hist_and_cumulative
```python
hist_and_cumulative(data, ax=None, log=False)
```
Make a plot with both a histogram and cumulative distribution.


### make_rec
```python
make_rec(y, x, width, height, linewidth)
```
Make a rectangle of width and height _centered_ on (y, x).


### make_rec_from_slice
```python
make_rec_from_slice(yxslice, **kwargs)
```
Make a rectangle of width and height _centered_ on (y, x).


## dphtools/utils/__init__.py

Various utility functions to be organized better.

Copyright (c) 2021, David Hoffman


### get_git
```python
get_git(path='.')
```
Get git description.


### bin_ndarray
```python
bin_ndarray(ndarray, new_shape=None, bin_size=None, operation='sum')
```
Bins an ndarray in all axes based on the target shape, by summing or averaging.

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

Example
-------
>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> bin_ndarray(a, bin_size=2)
array([[10, 18],
       [42, 50]])


### scale
```python
scale(data, dtype=None)
```
Scale data to [0.0, 1.0] range, unless an integer dtype is specified in which case the data is scaled to fill the bit depth of the dtype.

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
>>> bool(b.max() == 1.0)
True
>>> bool(b.min() == 0.0)
True
>>> b = scale(a, dtype = np.uint16)
>>> bool(b.max() == 65535)
True
>>> bool(b.min() == 0)
True


### radial_profile
```python
radial_profile(data, center=None, binsize=1.0)
```
Take the radial average of a 2D data array.

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


### mode
```python
mode(data: np.ndarray)
```
Get mode of non-negative integer data.

up to 1000 times faster than scipy mode
but not nearly as feature rich

Note: we can vectorize this to work on different
axes with numba

Parameters
----------
data : np.ndarray
    Data to get mode of

Returns
-------
mode : int
    Modal value

Example
-------
>>> a = np.array([0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 10])
>>> bool(mode(a) == 4)
True


### slice_maker
```python
slice_maker(xs, ws)
```
Generate a tuple of slices to cut out a sub-array centered on `xs` with widths `ws`.

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
>>> slice_maker((30, 20), 10) == (slice(25, 35, None), slice(15, 25, None))
True
>>> slice_maker((30, 20), 25) == (slice(18, 43, None), slice(8, 33, None))
True


### fft_pad
```python
fft_pad(array, newshape=None, mode='median', **kwargs)
```
Pad an array to prep it for FFT.


### fftconvolve_fast
```python
fftconvolve_fast(data, kernel, **kwargs)
```
FFT convolution, a faster version than scipy.

In this case the kernel ifftshifted before FFT but the data is not.
This can be done because the effect of fourier convolution is to
"wrap" around the data edges so whether we ifftshift before FFT
and then fftshift after it makes no difference so we can skip the
step entirely.


### win_nd
```python
win_nd(size, win_func=scipy.signal.windows.hann, **kwargs)
```
Make a multidimensional version of a window function.

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


### anscombe
```python
anscombe(data)
```
Apply Anscombe transform to data.

https://en.wikipedia.org/wiki/Anscombe_transform


### anscombe_inv
```python
anscombe_inv(data)
```
Apply inverse Anscombe transform to data.

https://en.wikipedia.org/wiki/Anscombe_transform


### fft_gaussian_filter
```python
fft_gaussian_filter(img, sigma)
```
FFT gaussian convolution.

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


### find_prime_facs
```python
find_prime_facs(n)
```
Find the prime factors of n.

Example
-------
>>> find_prime_facs(10)
array([2, 5])


### montage
```python
montage(stack)
```
Take a stack and a new shape and cread a montage.


### square_montage
```python
square_montage(stack)
```
Turn a 3D stack into a square montage.


### latex_format_e
```python
latex_format_e(num, pre=2)
```
Format a number for nice latex presentation, the number will *not* be enclosed in "$".


### localize_peak_1d
```python
localize_peak_1d(data)
```
Small utility function to localize a peak center.


### localize_peak
```python
localize_peak(data)
```
Small utility function to localize a peak center.

Assumes passed data has peak at center and that data.shape is odd and symmetric.
Then fits a parabola through each line passing through the center. This is optimized
for FFT data which has a non-circularly symmetric shaped peaks.


### get_max
```python
get_max(xdata, ydata, axis=0)
```
Get the x value that corresponds to the max y value.


### edf
```python
edf(stack)
```
Calculate extended depth of focus, simple algo, take the value with the max gradient.


### plane_fit
```python
plane_fit(X, Y, Z)
```
Fit a plane to data.

Parameters
----------
X : np.ndarray
Y : np.ndarray
Z : np.ndarray

Returns
-------
C : np.ndarray
    Coefficients for plane fit


### remove_tilt
```python
remove_tilt(X, Y, Z)
```
Fit a plane to data and remove tilt (no rotation).


### find_normal
```python
find_normal(X, Y, Z)
```
Find the normal vector for a plane.


### rot_matrix
```python
rot_matrix(source, target)
```
Calculate the rotation matrix to rotate source to target.


### calc_angles
```python
calc_angles(mat_b)
```
Calculate angles based on rotation matrix.


### fit_quadratic
```python
fit_quadratic(x: np.ndarray, y: np.ndarray, z: np.ndarray)
```
Fit quadratic to point data.

Parameters are:
C[0] * X ** 2 + C[1] * Y ** 2 + C[2] * X + C[3] * Y + C[4] * X * Y + C[5]


### find_center
```python
find_center(x: np.ndarray, y: np.ndarray, z: np.ndarray)
```
Find center of parabola.

https://en.wikipedia.org/wiki/Quadratic_function#Minimum/maximum

Parameters
----------
x, y, z : np.ndarrays
    The x, y, and z coordinates of the data (assumes that x and y
    are independent variables and z is dependent)

Returns
-------
(x_m, y_m) : floats
    The center as determined by a parabolic fit


### find_center_quad_coefs
```python
find_center_quad_coefs(quad_coefs, stderr)
```
Find the center of a quadratic fit given it's coefficients.

Parameters
----------
quad_coefs : np.ndarray
    Coefficients of the quadratic fit: see `fit_quadratic` for functional form.
stderr : np.ndarray
    Standard error on those coefficients

Returns
-------
x0, y0 : tuple
    Found center
x0_e, y0_e : tuple
    standard error on the found center

NOTE: the error of the center is found using the normal simplification which may not
be applicable or desireable here (https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification)


### EasyTimer
Makes timing things easy with a `with` statement.


#### EasyTimer.__init__
```python
__init__(self, msg='')
```
Create an EasyTimer, choose the emit message.


### split_img
```python
split_img(img, sides)
```
Split an image (or volume) into tiles.

taken from https://github.com/david-hoffman/scripts/blob/master/simrecon_utils.py


### crop_image_for_split
```python
crop_image_for_split(img, sides)
```
Take an image and a side and crop it apropriately to ensure that split_img will work.


### combine_img
```python
combine_img(stack)
```
Reassemble a square grid of tiles returned by ``split_img``.


## dphtools/utils/beads.py

Bead specific functons.

Copyright (c) 2021, David Hoffman


### remove_coord_mean
```python
remove_coord_mean(df, *, coords=['x0', 'y0'])
```
Remove the mean value of the coordinates.


### calc_drift
```python
calc_drift(fiducials_df, *, coords=['x0', 'y0'], frame_name='slice', weighted='amp', diagnostics=False, frames_index=None)
```
Calculate image drift from multiple emitters in a FOV.

Given a list of DataFrames with each DF containing the coordinates
of a single fiducial calculate the mean or weighted mean of the coordinates
in each frame. ``weighted=""`` selects the unweighted mean; ``"coords"``
selects inverse coordinate-variance weights, and other nonempty strings
name the weight column (``"amp"`` by default).


## dphtools/utils/fitfuncs.py

Various functions for fitting things.

Copyright (c) 2021, David Hoffman


### multi_exp
```python
multi_exp(xdata, *args)
```
Sum of exponentials.

.. math:: y = bias + \sum_n A_i e^{-k_i x}


### multi_exp_jac
```python
multi_exp_jac(xdata, *args)
```
Jacopian for multi_exp.


### exponent
```python
exponent(xdata, amp, rate, offset)
```
Single exponential function.

.. math:: y = amp e^{-rate xdata} + offset


### exponent_fit
```python
exponent_fit(data, xdata=None, offset=True)
```
Fit data to a single exponential function.


### multi_exp_fit
```python
multi_exp_fit(data, xdata=None, components=None, offset=True, **kwargs)
```
Fit data to a multi-exponential function.

Assumes evenly spaced data.

Parameters
----------
data : ndarray (1d)
    data that can be modeled as a sum of exponential decays
xdata : numeric
    x axis for fitting
components : int
    Number of exponential components. Automatic selection with None is
    unsupported and raises NotImplementedError.

Returns
-------
popt : ndarray
    optimized parameters for the exponent wave
    (a0, k0, a1, k1, ... , an, kn, offset)
pcov : ndarray
    covariance of optimized paramters


label_base = "$y(t) = " + "{:+.3f} e^{{-{:.3g}t}}" * (len(popt) // 2) + " {:+.0f}$" * (len(popt) % 2)


### estimate_power_law
```python
estimate_power_law(x, y, diagnostics=False)
```
Estimate the best fit parameters for a power law by linearly fitting the loglog plot.


### power_percentile
```python
power_percentile(p, popt, xmin=1)
```
Percentile of a single power law function.


### power_percentile_inv
```python
power_percentile_inv(x0, popt, xmin=1)
```
Given an x value what percentile of the power law function does it correspond to.


### power_intercept
```python
power_intercept(popt, value=1)
```
At what x value does the function reach value.


### power_law
```python
power_law(xdata, *args)
```
Multi-power law function.


### power_law_jac
```python
power_law_jac(xdata, *args)
```
Jacobian for a multi-power law function.


### powerlaw_prng
```python
powerlaw_prng(alpha, xmin=1, xmax=10000000.0)
```
Calculate a psuedo random variable drawn from a discrete power law distribution with scale parameter alpha and xmin.


### PowerLaw
Class for fitting and testing power law distributions.


#### PowerLaw.__init__
```python
__init__(self, data)
```
Object representing power law data.

Pass in data, it will be automagically determined to be
continuous (float/inexact datatype) or discrete (integer datatype)


#### PowerLaw.fit
```python
fit(self, xmin=None, xmin_max=200, opt_max=False)
```
Fit the data, if xmin is none then estimate it.


#### PowerLaw.clipped_data
```python
clipped_data(self)
```
Return data clipped to xmin.


#### PowerLaw.intercept
```python
intercept(self, value=1)
```
Return the intercept calculated from power law values.


#### PowerLaw.percentile
```python
percentile(self, value)
```
Return the intercept calculated from power law values.


#### PowerLaw.gen_power_law
```python
gen_power_law(self)
```
x.append(xmin*pow(1.-random(),-1./(alpha-1.))).


#### PowerLaw.calculate_p
```python
calculate_p(self, num=1000)
```
Make a bunch of fake data and run the KS_test on it.


#### PowerLaw.plot
```python
plot(self, ax=None, density=True, norm=False)
```
Plot data.


### fit_ztp
```python
fit_ztp(data)
```
Fit the data assuming it follows a zero-truncated Poisson model.


### NegBinom
```python
NegBinom(a, m)
```
Convert scipy's definition to mean and shape.


### negloglikelihoodNB
```python
negloglikelihoodNB(args, x)
```
Negative log likelihood for negative binomial.


### negloglikelihoodZTNB
```python
negloglikelihoodZTNB(args, x)
```
Negative log likelihood for zero truncated negative binomial.


### fit_ztnb
```python
fit_ztnb(data, x0=(0.5, 0.5))
```
Fit the data assuming it follows a zero-truncated Negative Binomial model.


## dphtools/utils/histstats.py

Small utility functions for calculating histogram stats.

https://en.wikipedia.org/wiki/Standardized_moment

Copyright (c) 2016, David Hoffman


### hist_mean
```python
hist_mean(weights, bins=None)
```
Histogram mean.


### hist_var
```python
hist_var(weights, bins=None)
```
Histogram variance.


### hist_moment
```python
hist_moment(weights, bins=None, k=3)
```
Generalized histogram moment.

Defaults to the third one


## dphtools/utils/lm.py

Levenberg–Marquardt fitting with analytic derivatives.

The custom ``mle`` method minimizes the Laurence–Chromy Poisson deviance;
``ls`` provides an unweighted least-squares comparison. These custom methods
support fewer options than SciPy. ``curve_fit`` delegates its other supported
methods to SciPy; it is not a drop-in replacement for every SciPy option.

### References
1. Methods for Non-Linear Least Squares Problems (2nd ed.) http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3215 (accessed Aug 18, 2017).
1. [Laurence, T. A.; Chromy, B. A. Efficient Maximum Likelihood Estimator Fitting of Histograms. Nat Meth 2010, 7 (5), 338–339.](http://www.nature.com/nmeth/journal/v7/n5/full/nmeth0510-338.html)
1. Numerical Recipes in C: The Art of Scientific Computing, 2nd ed.; Press, W. H., Ed.; Cambridge University Press: Cambridge ; New York, 1992.
1. https://www.osti.gov/scitech/servlets/purl/7256021/
Copyright (c) 2017, David Hoffman


### make_lambda
```python
make_lambda(j, d0)
```
Make the diagonal matrix which takes care of scaling.

according to J. J. Moré's paper


### lm
```python
lm(func, x0, args=(), Dfun=None, full_output=False, col_deriv=True, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=None, epsfcn=None, factor=100, diag=None, method='ls')
```
Fit unweighted least squares or Poisson counts with analytic derivatives.

Parameters
----------
func : callable
    Called as ``func(params)``. For ``method="ls"``, return an M-vector
    of residuals. For ``method="mle"``, return ``(predictions, counts)``:
    strictly positive model predictions and nonnegative observed counts.
    The objective is half the Poisson deviance,
    ``sum(mu - y + y * log(y / mu))``, with the log term zero for y=0.
    Invalid trial predictions are rejected, not clipped.
x0 : array_like
    Initial N-vector of parameters; the initial objective must be finite.
args : tuple, optional
    Extra-argument forwarding is unimplemented; only the empty tuple is
    supported. Bind additional data in ``func`` and ``Dfun`` instead.
Dfun : callable
    Analytic Jacobian, called as ``Dfun(params)``, of residuals (ls) or
    predictions (mle). Return shape (M, N): one row per observation and
    one column per parameter. Numerical derivatives are unimplemented.
full_output : bool, optional
    If True, return the five-item result described below. Default False.
col_deriv : bool, optional
    Must be True (the default). This legacy flag does not use SciPy's
    orientation convention; the Jacobian still has shape (M, N).
    False is unimplemented.
ftol : float, optional
    Stop after an accepted step reduces the objective by at most this
    fraction of its previous value. Default 1.49012e-8.
xtol : float, optional
    Stop when the proposed step norm is at most
    ``xtol * (norm(params) + xtol)``. Default 1.49012e-8.
gtol : float, optional
    Stop when the largest absolute gradient component is at most gtol.
    The default 0 disables this check.
maxfev : int, optional
    Legacy name for the trial-iteration limit, including unsuccessful
    linear solves. Default ``100 * (N + 1)``. There is also one initial
    function evaluation; the actual count is returned in ``nfev``.
epsfcn : float, optional
    Unused inherited argument; does not enable numerical derivatives.
factor : float, optional
    Damping multiplier after a singular linear solve. Default 100.
    This is not SciPy's initial-step-bound option.
diag : sequence, optional
    Unused inherited argument; custom variable scaling is unimplemented.
method : {"ls", "mle"}, optional
    Default "ls" minimizes half the residual sum of squares. "mle"
    minimizes the Laurence–Chromy Poisson objective using approximate
    curvature ``J.T @ diag(y / mu**2) @ J`` and adaptive damping.

Returns
-------
popt : ndarray
    Last accepted parameters.
cov_x : None
    This low-level solver does not compute covariance.
infodict : dict, optional
    With full_output, contains ``fvec`` (residuals for ls, predictions
    for mle), ``fjac`` at the returned parameters, and the actual number
    of function calls ``nfev``.
message : str, optional
    Termination description, returned with full_output.
status : int, optional
    With full_output: 1 for objective convergence, 2 for step convergence,
    4 for gradient convergence, or 5 for exhausted iterations. Without
    full_output, exhaustion is logged and the last accepted point returned.


### curve_fit
```python
curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-np.inf, np.inf), method=None, jac=None, **kwargs)
```
Fit a model using SciPy or the custom analytic-derivative solvers.

Parameters
----------
f : callable
    Model called as ``f(xdata, *params)``. For custom "mle", predictions
    must be strictly positive; use a suitable model parameterization.
xdata : array_like or object
    Independent variables passed to the model.
ydata : array_like
    Observations. Custom "mle" requires finite nonnegative Poisson counts,
    including zero-count bins. Other methods minimize squared residuals.
p0 : scalar or array_like or None, optional
    Initial guess for SciPy. If None, SciPy infers the parameter count and
    starts at ones. Custom methods first run an unweighted SciPy
    least-squares fit, then use its result as their initial parameters.
sigma : scalar or array_like or None, optional
    Passed to delegated SciPy methods. Custom methods require None;
    weighting is unimplemented, including a supplied all-ones sigma.
absolute_sigma : bool, optional
    Passed to SciPy. Custom covariance is unscaled for both True and
    False (default); custom covariance rescaling is unimplemented.
check_finite : bool, optional
    Check input arrays for NaNs and infinities. Default True.
bounds : 2-tuple of array_like, optional
    Parameter bounds passed to SciPy. Custom methods support only
    unbounded parameters (the default ``(-inf, inf)``).
method : {None, "lm", "trf", "dogbox", "ls", "mle"}, optional
    None (default), "lm", "trf", and "dogbox" delegate to SciPy curve_fit.
    "ls" selects custom unweighted least squares. "mle" selects the
    Laurence–Chromy Poisson objective, not least squares. "pyls" and
    unknown methods raise TypeError.
jac : callable or str or None, optional
    Custom methods require an analytic Jacobian ``jac(xdata, *params)``
    with shape (observations, parameters); None and numerical derivative
    selectors raise NotImplementedError. Delegated methods retain SciPy's
    supported numerical derivative options.
**kwargs
    Options for the delegated SciPy fit, or for both the SciPy initializer
    and custom ``lm``. See ``lm`` for custom tolerances and limitations.
    ``full_output=True`` requests a five-item result instead of two.
    Custom ``col_deriv`` must be True despite its legacy name. Inherited
    ``epsfcn`` and ``diag`` do not tune the custom iteration.

Returns
-------
popt : ndarray
    Fitted parameters.
pcov : ndarray
    SciPy's covariance for delegated methods. Custom methods return the
    unscaled pseudoinverse of ``J.T @ J`` at the fitted point, discarding
    numerically zero singular values. This has no validated Poisson
    confidence-interval interpretation and ignores ``absolute_sigma``.
infodict : dict or None, optional
    With full_output, diagnostics from SciPy or custom ``lm``. Delegated
    bounded fits and "trf"/"dogbox" retain the legacy placeholder None.
message : str, optional
    Termination message with full_output. The legacy delegated placeholder
    result uses "No error".
status : int, optional
    Termination status with full_output. The legacy delegated placeholder
    result uses 1. Failed custom convergence raises RuntimeError.


## dphtools/utils/lpsvd.py

Conversion of old IgorPro code.

LPSVD was developed by Tufts and Kumaresan (Tufts, D.; Kumaresan, R. IEEE Transactions on Acoustics,
Speech and signal Processing 1982, 30, 671 – 675.) as a method of harmonic inversion, i.e. decomposing
a time signal into a linear combination of (decaying) sinusoids.

The backward-prediction equations for damped signals are given by Kumaresan, R.;
Tufts, D. W. IEEE Transactions on Acoustics, Speech, and Signal Processing 1982,
30 (6), 833–840, equations (2)–(4), DOI: 10.1109/TASSP.1982.1163974.
https://www.math.ucdavis.edu/~saito/data/sonar/KumaresanTufts.pdf

A great reference that is easy to read for the non-EECS user is:
Barkhuijsen, H.; De Beer, R.; Bovée, W. M. M. .; Van Ormondt, D. J. Magn. Reson. (1969) 1985, 61, 465–481.

This particular implementation was adapted, in part, from matNMR by Jacco van Beek
http://matnmr.sourceforge.net/
and  Complex Exponential Analysis by Greg Reynolds
http://www.mathworks.com/matlabcentral/fileexchange/12439-complex-exponential-analysis/

Author: David Hoffman (dave.p.hoffman@gmail.com)
Date: Aug, 2015


### LPSVD
```python
LPSVD(signal, M=None, lfactor=1 / 2, removebias=True)
```
Perform linear prediction-singular value decomposition.

NOTE: Assumes signal to be a linear combination of damped sinusoids.

Parameters
----------
signal : ndarray
    The signal to be analyzed
M : int
    Model order, if None, it will be estimated
lfactor : float
    Set L = floor(len(signal) * lfactor) prediction coefficients and
    len(signal) - L prediction equations. The default uses half the samples
    for the coefficient count. Both matrix dimensions must accommodate
    the signal rank.
removebias    : bool
    If true bias will be removed from the singular values of A


### estimate_model_order
```python
estimate_model_order(s, N, L)
```
Estimate model order.

Adapted from from Complex Exponential Analysis by Greg Reynolds
http://www.mathworks.com/matlabcentral/fileexchange/12439-complex-exponential-analysis/
Use the MDL method as in Lin (1997) to compute the model
order for the signal. You must pass the vector of
singular values, i.e. the result of svd(T) and
N and L. This method is best explained by Scharf (1992).

Parameters
----------
s : ndarray
    singular values from SVD decomposition
N : int
L : int

Returns
-------
M : float
    Estimated model order


### calc_LPSVD_error
```python
calc_LPSVD_error(LPSVD_coefs, data)
```
Calculate LPSVD error.

A function that estimates the errors on the LPSVD parameters using the Cramer-Rao
lower bound (http://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).
This implementation is based on the work of Barkhuijsen et al (http://dx.doi.org/10.1016/0022-2364(86)90446-4)

Parameters
----------
LPSVD_coefs    : DataFrame
    Coefficients calculated from the LPSVD algorithm, we will add errors to this DataFrame
data : ndarray
    The data from which the LPSVD coefficients were calculated


### reconstruct_signal
```python
reconstruct_signal(LPSVD_coefs, signal, ampcutoff=0, freqcutoff=0, dampcutoff=0)
```
Reconstruct signal.

#A function that reconstructs the original signal in the time domain and frequency domain
#from the LPSVD algorithms coefficients, which are passed as LPSVD_coefs
#http://mathworld.wolfram.com/FourierTransformLorentzianFunction.html

WAVE LPSVD_coefs        #coefficients from the LPSVD algorithm
String name                #Name of the generated waves
Variable length            #Length of the time domain signal
Variable timeStep        #Sampling frequency with which the signal was recorded, in fs
Variable dataReal        #Should the output time domain data be real?
Variable ampcutoff        #Cutoff for the amplitudes of the components
Variable freqcutoff        #Cutoff for the frequency of the components
Variable dampcutoff        #Cutoff for the damping of the components


## dphtools/utils/registration.py

Classes for registering point sets.

Based on:
- Myronenko and Xubo Song - 2010 - Point Set Registration Coherent Point Drift
DOI: 10.1109/TPAMI.2010.46

Copyright (c) 2018, David Hoffman


### BaseCPD
Base class for the coherent point drift algorithm.

Based on:
Myronenko and Xubo Song - 2010 - Point Set Registration Coherent Point Drift
DOI: 10.1109/TPAMI.2010.46


#### BaseCPD.__init__
```python
__init__(self, X: np.ndarray, Y: np.ndarray)
```
Set up the registration class that will actually perform the CPD algorithm.

Parameters
----------
X : ndarray (N, D)
    Fixed point cloud, an N by D array of N original observations in an n-dimensional space
Y : ndarray (M, D)
    Moving point cloud, an M by D array of N original observations in an n-dimensional space


#### BaseCPD.scale
```python
scale(self)
```
Return the estimated scale of the transformation matrix.


#### BaseCPD.matches
```python
matches(self)
```
Return X, Y matches.


#### BaseCPD.estimate
```python
estimate(self)
```
Estimate the simple transform for matching pairs.


#### BaseCPD.plot
```python
plot(self, only2d=False)
```
Plot the results of the registration.


#### BaseCPD.transform
```python
transform(self, other: np.ndarray)
```
Transform `other` point cloud via the Y -> X registration.


#### BaseCPD.updateTY
```python
updateTY(self)
```
Update the transformed point cloud and distance matrix.


#### BaseCPD.estep
```python
estep(self)
```
Do expectation step were we calculate the posterior probability of the GMM centroids.


#### BaseCPD.updateB
```python
updateB(self)
```
Update B matrix.

This is the only method that needs to be overloaded for the various linear transformation subclasses,
more will need to be done for non-rigid transformation models.


#### BaseCPD.mstep
```python
mstep(self)
```
Maximization step.

Update transformation and variance these are the transposes of the equations on p. 2265 and 2266


#### BaseCPD.calc_var
```python
calc_var(self)
```
Calculate variance in transform.


#### BaseCPD.rmse
```python
rmse(self)
```
Return RMSE between X and transformed Y.


#### BaseCPD.calc_init_scale
```python
calc_init_scale(self)
```
Need to overloaded in child classes.


#### BaseCPD.norm_data
```python
norm_data(self)
```
Normalize data to mean 0 and unit variance.


#### BaseCPD.unnorm_data
```python
unnorm_data(self)
```
Undo the intial normalization.


#### BaseCPD.__call__
```python
__call__(self, tol=1e-06, dist_tol=0, maxiters=1000, init_var=None, weight=0, normalization=True)
```
Perform the actual registration.

Parameters
----------
tol : float
dist_tol : float
    Stop the iteration of the average distance between matching points is
    less than this number. This is really only necessary for synthetic data
    with no noise
maxiters : int
init_var : float
weight : float
B : ndarray (D, D)
translation : ndarray (1, D)


### TranslationCPD
Coherent point drift with a translation only transformation model.


#### TranslationCPD.updateB
```python
updateB(self)
```
Update step.

Translation only means that B should be identity.


#### TranslationCPD.calc_init_scale
```python
calc_init_scale(self)
```
For translation only we need to calculate a uniform scaling.


### SimilarityCPD
Coherent point drift with a similarity (translation, rotation and isotropic scaling) transformation model.


#### SimilarityCPD.calculateR
```python
calculateR(self)
```
Calculate the estimated rotation matrix, eq. (9).


#### SimilarityCPD.calculateS
```python
calculateS(self)
```
Calculate the scale factor, Fig 2 p. 2266.


#### SimilarityCPD.updateB
```python
updateB(self)
```
Update B: in this case is just the rotation matrix multiplied by the scale factor.


#### SimilarityCPD.calc_init_scale
```python
calc_init_scale(self)
```
Use one isotropic scale for both centered point clouds.


### RigidCPD
Coherent point drift with a rigid or Euclidean (translation and rotation) transformation model.


#### RigidCPD.calculateS
```python
calculateS(self)
```
No scaling for this guy.


### AffineCPD
Coherent point drift with a similarity (translation, rotation, shear and anisotropic scaling) transformation model.


#### AffineCPD.updateB
```python
updateB(self)
```
Solve for B using equations in Fig. 3 p. 2266.


#### AffineCPD.calc_init_scale
```python
calc_init_scale(self)
```
Calculate scale.


### choose_model
```python
choose_model(model)
```
Choose model if string.


### auto_weight
```python
auto_weight(X, Y, model, resolution=0.01, limits=0.05, **kwargs)
```
Automatically determine the weight to use in the CPD algorithm.

Parameters
----------
X : ndarray (N, D)
    Fixed point cloud, an N by D array of N original observations in an n-dimensional space
Y : ndarray (M, D)
    Moving point cloud, an M by D array of N original observations in an n-dimensional space
model : str or BaseCPD child class
    The transformation model to use, available types are:
        Translation
        Rigid
        Euclidean
        Similarity
        Affine
resolution : float
    the resolution at which to sample the weights
limits : float or length 2 iterable
    The limits of weight to search
kwargs : dictionary
    key word arguments to pass to the model function when its called.


### nearest_neighbors
```python
nearest_neighbors(fids0, fids1, r=100, transform=lambda x: x, coords=['x0', 'y0'])
```
Find nearest neighbors in both sets.


### align
```python
align(fids0, fids1, atol=1, rtol=0.001, diagnostics=False, model='translation', only2d=False, iters=100)
```
Align two slabs fiducials, assumes that z coordinate has been normalized.


### closest_point_matches
```python
closest_point_matches(X, Y, method='tree', **kwargs)
```
Keep determine the nearest neighbors in two point clouds.

Parameters
----------
X : ndarray (N, D)
Y : ndarray (M, D)

kwargs
------
r : float
    The search radius for nearest neighbors

Returns
-------
xpoints : ndarray
    indicies of points with neighbors in x
ypoints : ndarray
    indicies of points with neighbors in y


### to_augmented
```python
to_augmented(B, t)
```
Convert transform matrix and translation vector to an augmented transformation matrix.

https://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix


### from_augmented
```python
from_augmented(aug_B)
```
Convert the augmented matrix back into transformation matrix + tranlsation vector.


### propogate_transforms
```python
propogate_transforms(regs, initial=None)
```
Propagate transforms along slabs.


### apply_transform_to_slab
```python
apply_transform_to_slab(s, B, t, copy=True)
```
Apply a given transform to a slab.


## dphtools/utils/rolling_ball.py

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


### sq_norm
```python
sq_norm(v)
```
Squared norm.


### circumcircle
```python
circumcircle(points, simplex)
```
Get the circumcenter and circum radius of all the simplices, works for 2D only.

Compute the circumcenter and circumradius of a triangle (see their definitions
https://en.wikipedia.org/wiki/Circumscribed_circle#Circumcircle_equations)

http://mathworld.wolfram.com/Circumcircle.html


### get_alpha_complex
```python
get_alpha_complex(alpha, points, simplices)
```
Get alpha complex.


### rolling_ball_filter_accurate
```python
rolling_ball_filter_accurate(data, ball_radius, roll_along=-1, top=True, interpolator=interpolate.interp1d, **kwargs)
```
Filter data via a rolling ball algorithm.

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


### rolling_ball_filter
```python
rolling_ball_filter(data, ball_radius, spacing=None, top=False, **kwargs)
```
Filter data via a rolling ball algorithm.

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
