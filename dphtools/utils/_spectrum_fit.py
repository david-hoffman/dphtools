"""Joint unweighted fits of positive, unit-height spectrum components."""

import warnings
from types import SimpleNamespace

import numpy as np
from scipy.optimize import OptimizeWarning, least_squares
from scipy.signal import find_peaks
from scipy.special import voigt_profile


def _real_array(value, name):
    """Copy real numeric input into finite floating-point storage."""
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _estimate_rows(data, x, centers, baseline, family):
    """Estimate heights and widths without selecting additional components."""
    corrected = data - baseline
    floor = max(np.max(np.abs(data)) * 1e-3, np.finfo(float).eps)
    rows = []
    for center in centers:
        amplitude = max(np.interp(center, x, corrected), floor)
        index = np.argmin(np.abs(x - center))
        left = np.flatnonzero(corrected[:index] < amplitude / 2)
        right = np.flatnonzero(corrected[index + 1 :] < amplitude / 2) + index + 1
        # Interpolate the half-height crossings in physical coordinates.
        lo, hi = x[0], x[-1]
        if left.size:
            j = left[-1]
            lo = x[j] + (x[j + 1] - x[j]) * (amplitude / 2 - corrected[j]) / (
                corrected[j + 1] - corrected[j]
            )
        if right.size:
            j = right[0]
            hi = x[j - 1] + (x[j] - x[j - 1]) * (amplitude / 2 - corrected[j - 1]) / (
                corrected[j] - corrected[j - 1]
            )
        fwhm = max(hi - lo, np.min(np.diff(x)))
        # Overlap can inflate the measured width of the combined spectrum.
        separations = np.abs(centers - center)
        separations = separations[separations > 0]
        if separations.size:
            fwhm = min(fwhm, 2 * separations.min())
        if family == "gauss":
            rows.append([amplitude, center, fwhm / np.sqrt(8 * np.log(2))])
        elif family == "lorentz":
            rows.append([amplitude, center, fwhm / 2])
        else:
            # Split the initial broadening between the two independent widths.
            rows.append([amplitude, center, fwhm / 3.6, fwhm / 3.6])
    return np.array(rows)


def _covariance(residual, parameters, scales, data_scale, dof):
    """Calculate residual-scaled covariance in physical parameter coordinates."""
    # Differentiate on local parameter scales, rather than an absolute x-unit
    # floor. Work in dimensionless columns until restoring physical covariance.
    steps = np.cbrt(np.finfo(float).eps) * scales
    if not np.isfinite(steps).all() or np.any(steps <= 0):
        raise RuntimeError("Spectrum covariance differences are not representable")
    jacobian = []
    for index, step in enumerate(steps):
        delta = np.zeros_like(parameters)
        delta[index] = step
        upper, lower = parameters + delta, parameters - delta
        span = (upper[index] - lower[index]) / scales[index]
        jacobian.append((residual(upper) / data_scale - residual(lower) / data_scale) / span)
    jacobian = np.column_stack(jacobian)
    if not np.isfinite(jacobian).all():
        raise RuntimeError("Spectrum covariance Jacobian is not finite")
    norms = np.max(np.abs(jacobian), axis=0)
    jacobian /= np.where(norms > 0, norms, 1)
    try:
        _, singular, vectors = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Spectrum covariance decomposition failed") from error
    threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0]
    if np.any(singular <= threshold):
        warnings.warn("Spectrum parameter covariance could not be estimated", OptimizeWarning)
        return np.full((parameters.size, parameters.size), np.inf)
    error_scale = np.linalg.norm(residual(parameters) / data_scale) / np.sqrt(dof)
    factor = (vectors.T / singular) * (error_scale / norms)[:, None] * scales[:, None]
    covariance = factor @ factor.T
    if not np.isfinite(covariance).all():
        raise RuntimeError("Spectrum covariance is not representable")
    return covariance


def spectrum_fit(
    data,
    xdata=None,
    *,
    peak_type="gauss",
    guesses=None,
    background="constant",
    prominence=None,
    distance=None,
    max_nfev=10000,
    optimizer="lm",
):
    """Fit positive Gaussian, Lorentzian or Voigt peaks jointly to a spectrum.

    Parameters
    ----------
    data : array_like
        Nonempty finite real one-dimensional point observations. Not modified.
    xdata : array_like, optional
        Matching finite real strictly increasing coordinates, possibly nonuniform.
        Defaults to sample indices.
    peak_type : {'gauss', 'lorentz', 'voigt'}, optional
        Profile family for all components. Names are case insensitive;
        'gaussian' and 'lorentzian' are aliases.
    guesses : array_like, optional
        Nonempty one-dimensional centers, or full parameter rows. Gaussian rows
        are (height, center, sigma), Lorentzian rows are (height, center, gamma),
        and Voigt rows are (height, center, sigma, gamma). Heights and widths
        must be positive. Centers are free, even outside the observed interval.
        Omission detects local maxima. Supplied guesses are not modified.
    background : {'none', 'constant', 'linear'}, optional
        Jointly fitted background. A line is b0 + b1 * (x - xdata[0]).
    prominence : float, optional
        Nonnegative minimum prominence in data units for automatic discovery.
    distance : float, optional
        Minimum separation in samples, at least one, for automatic discovery.
        Neither discovery control may be combined with explicit guesses.
    max_nfev : int, optional
        Positive optimizer evaluation limit, counted by the selected solver.
    optimizer : {'lm'} or callable, optional
        Default SciPy Levenberg-Marquardt least squares uses internal log
        transforms for positive parameters and coordinate-span conditioning.
        Data and returned parameters retain physical units. A callable receives
        ``optimizer(residual, initial, bounds=(lower, upper), max_nfev=max_nfev)``.
        All callable parameters use physical units in fixed input row order,
        followed by background coefficients; residual is data minus model.
        Heights/widths have lower bounds zero and upper bounds infinity, while
        centers/background are unbounded. Final heights/widths must be positive.
        The callable must honor the limit and return finite real 1-D ``x`` and
        Boolean ``success``; ``message`` is optional.

    Returns
    -------
    result : object
        Attributes are ``peak_parameters`` (rows sorted by fitted center),
        ``background_parameters``, full ``covariance``, ``fitted`` and
        ``residuals`` (data minus fitted). Covariance follows flattened sorted
        peak rows then background, including cross correlations in physical units.
        Returned arrays do not share storage with caller inputs.

    Raises
    ------
    ValueError
        For invalid inputs/options, no eligible peaks, or samples not exceeding
        the number of fitted parameters.
    RuntimeError
        For numerical optimizer failure/exhaustion or malformed/nonphysical output.

    Notes
    -----
    Amplitudes are individual component peak heights, not areas. Gaussian sigma
    is standard deviation; Lorentzian gamma is half width at half maximum.
    Voigt is ``voigt_profile(x-center, sigma, gamma) / voigt_profile(0, sigma,
    gamma)`` and requires both widths positive. Centers/widths use x units.

    Minimize unweighted squared residuals at the supplied samples, including
    nonuniform coordinates. Covariance is the local inverse Jacobian information
    scaled by residual sum of squares / (samples - parameters); it is approximate
    and does not establish identifiability or global optimality. Singular local
    information emits OptimizeWarning and returns infinite covariance.
    Discovery promises local-maximum candidates, not endpoint or unresolved peaks.
    """
    data = _real_array(data, "data")
    if data.ndim != 1 or not data.size:
        raise ValueError("data must be a nonempty one-dimensional spectrum")
    x = np.arange(data.size, dtype=float) if xdata is None else _real_array(xdata, "xdata")
    if x.ndim != 1 or x.shape != data.shape or np.any(np.diff(x) <= 0):
        raise ValueError("xdata must match data and be strictly increasing")
    aliases = {
        "gauss": "gauss",
        "gaussian": "gauss",
        "lorentz": "lorentz",
        "lorentzian": "lorentz",
        "voigt": "voigt",
    }
    if not isinstance(peak_type, str) or peak_type.lower() not in aliases:
        raise ValueError("Unsupported peak_type")
    family = aliases[peak_type.lower()]
    backgrounds = {"none": 0, "constant": 1, "linear": 2}
    if not isinstance(background, str) or background not in backgrounds:
        raise ValueError("Unsupported background")
    nbackground = backgrounds[background]
    if (
        isinstance(max_nfev, (bool, np.bool_))
        or not isinstance(max_nfev, (int, np.integer))
        or max_nfev <= 0
    ):
        raise ValueError("max_nfev must be a positive integer")
    default_optimizer = isinstance(optimizer, str) and optimizer == "lm"
    if not default_optimizer and not callable(optimizer):
        raise ValueError("optimizer must be 'lm' or a callable")
    for name, value, minimum in (("prominence", prominence, 0), ("distance", distance, 1)):
        if value is not None:
            control = _real_array(value, name)
            if control.ndim != 0 or control < minimum:
                raise ValueError(f"{name} must be a scalar at least {minimum}")
    if guesses is not None and (prominence is not None or distance is not None):
        raise ValueError("Discovery controls require guesses=None")
    row_size = 4 if family == "voigt" else 3
    if guesses is None:
        indices, _ = find_peaks(data, prominence=prominence, distance=distance)
        guesses = x[indices]
    guesses = _real_array(guesses, "guesses")
    if guesses.ndim not in (1, 2) or not guesses.size:
        raise ValueError("Expected nonempty centers or full parameter rows")
    if guesses.ndim == 2 and guesses.shape[1] != row_size:
        raise ValueError("Full guess rows have the wrong number of parameters")
    npeak = len(guesses)
    size = npeak * row_size
    nparameter = size + nbackground
    if data.size <= nparameter:
        raise ValueError("More samples than fitted parameters are required")
    elapsed = x - x[0]
    slope = (data[-1] - data[0]) / elapsed[-1] if nbackground == 2 else 0
    intercept = np.min(data - slope * elapsed)
    coefficients = np.array([intercept, slope])[:nbackground]
    baseline = intercept + slope * elapsed if nbackground else np.zeros_like(data)
    rows = _estimate_rows(data, x, guesses, baseline, family) if guesses.ndim == 1 else guesses
    initial = np.concatenate((rows.ravel(), coefficients))
    positive = np.r_[np.tile(np.arange(row_size) != 1, npeak), np.zeros(nbackground, dtype=bool)]
    if np.any(initial[positive] <= 0):
        raise ValueError("Initial amplitudes and widths must be strictly positive")

    def residual(parameters):
        """Evaluate the unweighted residual in fixed physical row order."""
        model = np.zeros_like(data)
        for row in parameters[:size].reshape(npeak, row_size):
            amplitude, center, width = row[:3]
            d = x - center
            if family == "gauss":
                profile = np.exp(-0.5 * (d / width) ** 2)
            elif family == "lorentz":
                profile = 1 / (1 + (d / width) ** 2)
            else:
                profile = voigt_profile(d, width, row[3]) / voigt_profile(0, width, row[3])
            model += amplitude * profile
        if nbackground:
            model += parameters[size]
        if nbackground == 2:
            model += parameters[size + 1] * elapsed
        errors = data - model
        if not np.isfinite(errors).all():
            raise RuntimeError("Spectrum residual is not finite")
        return errors

    # SciPy also uses ValueError for numerical residual failures. A custom
    # callable's ordinary ValueError may instead indicate a programming error.
    optimization_errors = (
        ArithmeticError,
        ValueError if default_optimizer else np.linalg.LinAlgError,
    )
    try:
        if default_optimizer:
            # Make coordinate steps independent of x units and origin. A line's
            # private slope is its physical change across the observed span.
            parameter_scale = np.ones(nparameter)
            parameter_scale[:size].reshape(npeak, row_size)[:, 1:] = elapsed[-1]
            parameter_origin = np.zeros(nparameter)
            parameter_origin[:size].reshape(npeak, row_size)[:, 1] = x[0]
            if nbackground == 2:
                parameter_scale[-1] = 1 / elapsed[-1]

            def physical(parameters):
                """Undo private conditioning and positive-parameter log transforms."""
                values = parameters.copy()
                values[positive] = np.exp(values[positive])
                return values * parameter_scale + parameter_origin

            transformed = (initial - parameter_origin) / parameter_scale
            transformed[positive] = np.log(transformed[positive])
            solution = least_squares(
                lambda values: residual(physical(values)),
                transformed,
                method="lm",
                x_scale="jac",
                max_nfev=max_nfev,
            )
            parameters = physical(solution.x)
        else:
            lower = np.full(nparameter, -np.inf)
            lower[positive] = 0
            solution = optimizer(
                residual,
                initial.copy(),
                bounds=(lower, np.full(nparameter, np.inf)),
                max_nfev=max_nfev,
            )
            parameters = getattr(solution, "x", None)
    except optimization_errors as error:
        raise RuntimeError("Spectrum optimization failed") from error
    success = getattr(solution, "success", None)
    if not isinstance(success, (bool, np.bool_)) or not success:
        raise RuntimeError(
            f"Spectrum optimization failed: {getattr(solution, 'message', 'invalid status')}"
        )
    try:
        parameters = _real_array(parameters, "optimizer parameters")
    except (ValueError, TypeError) as error:
        raise RuntimeError("Malformed optimizer parameters") from error
    if parameters.shape != (nparameter,) or np.any(parameters[positive] <= 0):
        raise RuntimeError("Optimizer returned parameters outside the physical domain")
    scales = np.abs(parameters)
    peak_scales = scales[:size].reshape(npeak, row_size)
    # A center perturbation follows its component's width, even at center zero
    # or after translating the coordinate origin.
    peak_scales[:, 1] = np.max(peak_scales[:, 2:], axis=1)
    data_scale = max(np.max(np.abs(data)), np.max(peak_scales[:, 0]))
    if nbackground:
        scales[size] = data_scale
    if nbackground == 2:
        scales[size + 1] = data_scale / elapsed[-1]
    covariance = _covariance(residual, parameters, scales, data_scale, data.size - nparameter)
    order = np.argsort(parameters[:size].reshape(npeak, row_size)[:, 1], kind="stable")
    permutation = np.r_[
        np.arange(size).reshape(npeak, row_size)[order].ravel(), np.arange(size, nparameter)
    ]
    errors = residual(parameters)
    return SimpleNamespace(
        peak_parameters=parameters[:size].reshape(npeak, row_size)[order].copy(),
        background_parameters=parameters[size:].copy(),
        covariance=covariance[np.ix_(permutation, permutation)],
        fitted=data - errors,
        residuals=errors,
    )
