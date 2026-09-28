#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lm.py
"""
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
"""

import logging

import numpy as np
import scipy.optimize
from numpy import linalg as la
from scipy.linalg import solve_triangular

logger = logging.getLogger(__name__)


def _chi2_ls(f):
    """Sum of the squares of the residuals.

    Assumes that f returns residuals.

    Minimizing this will maximize the likelihood for a
    data model with gaussian deviates.
    """
    return 0.5 * (f**2).sum(0)


def _update_ls(x0, f, Dfun):
    """Hessian and gradient calculations for gaussian deviates."""
    # calculate the jacobian
    # j shape (ndata, nparams)
    j = Dfun(x0)
    # calculate the linear term of Hessian
    # a shape (nparams, nparams)
    a = j.T @ j
    # calculate the gradient
    # g shape (nparams,)
    g = j.T @ f
    return j, a, g


def _chi2_mle(f):
    """Return half the Poisson deviance, including zero-count bins."""
    f, y = f
    f, y = np.asarray(f), np.asarray(y)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Poisson counts must be finite and nonnegative")
    if not np.isfinite(f).all() or (f <= 0).any():
        return np.inf
    positive = y > 0
    return (f - y).sum() + (y[positive] * (np.log(y[positive]) - np.log(f[positive]))).sum()


def _update_mle(x0, f, Dfun):
    """Hessian and gradient calculations for poisson deviates."""
    # calculate the jacobian
    # j shape (ndata, nparams)
    f, y = f
    y_f = y / f
    y_f2 = y_f / f

    j = Dfun(x0)
    # calculate the linear term of Hessian
    # a shape (nparams, nparams)
    a = (j.T * y_f2) @ j
    # calculate the gradient
    # g shape (nparams,)
    g = j.T @ (1 - y_f)
    return j, a, g


def _wrap_func_mle(func, xdata, ydata, transform):
    """Return model predictions and unchanged Poisson counts."""
    if transform is None:

        def func_wrapped(params):
            # return function and data
            return np.asarray(func(xdata, *params)), ydata

    elif transform.ndim == 1:
        raise NotImplementedError
    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        raise NotImplementedError
    return func_wrapped


def _wrap_jac_mle(jac, xdata, transform):
    if transform is None:

        def jac_wrapped(params):
            return jac(xdata, *params)

    elif transform.ndim == 1:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return jac_wrapped


def _wrap_func_ls(func, xdata, ydata, transform):
    """Cost function as defined by Transtrum and Sethna."""
    if transform is None:

        def func_wrapped(params):
            return func(xdata, *params) - ydata

    elif transform.ndim == 1:

        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)

    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        def func_wrapped(params):
            return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)

    return func_wrapped


def _wrap_jac_ls(jac, xdata, transform):
    if transform is None:

        def jac_wrapped(params):
            return jac(xdata, *params)

    elif transform.ndim == 1:

        def jac_wrapped(params):
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))

    else:

        def jac_wrapped(params):
            return solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)

    return jac_wrapped


def make_lambda(j, d0):
    """Make the diagonal matrix which takes care of scaling.

    according to J. J. Moré's paper
    """
    # Calculate the norm of the jacobian columns
    ds = la.norm(j, axis=0)
    ds[0] = d0
    # return an increasing diagnonal matrix

    return np.diag([max(ds[i], ds[i - 1]) for i in range(1, len(ds))])


def lm(
    func,
    x0,
    args=(),
    Dfun=None,
    full_output=False,
    col_deriv=True,
    ftol=1.49012e-8,
    xtol=1.49012e-8,
    gtol=0.0,
    maxfev=None,
    epsfcn=None,
    factor=100,
    diag=None,
    method="ls",
):
    """Fit unweighted least squares or Poisson counts with analytic derivatives.

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
    """
    info = 0
    x0 = np.asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple) or args:
        raise NotImplementedError("Extra-argument forwarding has not been implemented")
    if not callable(Dfun):
        raise NotImplementedError("An analytic Jacobian is required")
    if not col_deriv:
        raise NotImplementedError("col_deriv=False has not been implemented")
    if maxfev is None:
        maxfev = 100 * (n + 1)

    errors = {
        1: "Relative objective reduction is at most {}".format(ftol),
        2: "Relative parameter step is at most {}".format(xtol),
        4: "Maximum absolute gradient component is at most {}".format(gtol),
        5: "Trial-iteration limit maxfev = {} reached.".format(maxfev),
    }

    def gtest(g):
        """Test if the gradient has converged."""
        if gtol:
            return np.abs(g).max() <= gtol
        else:
            return False

    def xtest(dx, x):
        """Check if the parameters have converged."""
        return la.norm(dx) <= xtol * (la.norm(x) + xtol)

    # set up update and chi2 for use
    if method == "ls":

        def update(x0, f):
            return _update_ls(x0, f, Dfun)

        def chi2(f):
            return _chi2_ls(f)

    elif method == "mle":

        def update(x0, f):
            return _update_mle(x0, f, Dfun)

        def chi2(f):
            return _chi2_mle(f)

    else:
        raise TypeError("Method {} not recognized".format(method))

    # get initial function, jacobian, hessian and gradient
    f = func(x0)
    nfev = 1
    chisq_old = chi2(f)
    if not np.isfinite(chisq_old):
        raise ValueError("Initial objective must be finite; Poisson predictions must be positive")
    j, a, g = update(x0, f)
    # initialize D.T @ D array
    dtd = np.diag(np.diag(a))
    # lambda
    lambda_ = np.sqrt(x0.T @ dtd @ x0)
    if lambda_ <= 0 or ~np.isfinite(lambda_):
        lambda_ = np.array(100.0)

    for ev in range(maxfev):
        logger.debug("Iteration #{}".format(ev))
        if gtest(g):
            info = 4
            break
        # Damping uses the diagonal curvature scale from accepted iterates.
        logger.debug("lambda_ = {}".format(lambda_))
        logger.debug("x = {}".format(x0))
        aug_a = a + lambda_ * dtd
        try:
            dx = la.solve(aug_a, -g)
        except la.LinAlgError:
            lambda_ *= factor
            continue

        if xtest(dx, x0):
            info = 2
            break
        trial_x = x0 + dx
        trial_f = func(trial_x)
        nfev += 1
        chisq_new = chi2(trial_f)

        actual_reduction = chisq_old - chisq_new
        # Quadratic model at the accepted point, for the half-deviance or
        # half-squared-residual objective. Do not linearize at the trial point.
        predicted_reduction = -g @ dx - 0.5 * (dx @ a @ dx)
        rho = 0.0
        if actual_reduction > 0 and predicted_reduction > 0:
            rho = actual_reduction / predicted_reduction
        if not np.isfinite(rho):
            rho = 0.0
        if rho > 1e-2:
            converged = actual_reduction <= ftol * chisq_old
            # update params, chisq and a and g
            x0, f, chisq_old = trial_x, trial_f, chisq_new
            j, a, g = update(x0, f)
            dtd = np.fmax(dtd, np.diag(np.diag(a)))
            lambda_ = max(lambda_ / 5, 1e-7)
            if converged:
                info = 1
                break
        else:
            lambda_ = min(lambda_ * 1.5, 1e7)

    else:
        # loop exited normally
        info = 5

    if method == "mle":
        # remember we return the data with f?
        f = f[0]

    logger.debug("Ended with {} function evaluations".format(nfev))

    infodict = dict(fvec=f, fjac=j, nfev=nfev)

    if info == 5 and not full_output:
        logger.warning(errors[info])

    errmsg = errors[info]
    logger.debug(errmsg)
    popt, cov_x = x0, None

    if full_output:
        return popt, cov_x, infodict, errmsg, info
    else:
        return popt, cov_x


def curve_fit(
    f,
    xdata,
    ydata,
    p0=None,
    sigma=None,
    absolute_sigma=False,
    check_finite=True,
    bounds=(-np.inf, np.inf),
    method=None,
    jac=None,
    **kwargs
):
    """Fit a model using SciPy or the custom analytic-derivative solvers.

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
    """
    # fix kwargs
    return_full = kwargs.pop("full_output", False)
    can_full_output = method not in {"trf", "dogbox"} and np.array_equal(bounds, (-np.inf, np.inf))

    if method in {"lm", "trf", "dogbox", None}:
        if can_full_output:
            kwargs["full_output"] = return_full

        res = scipy.optimize.curve_fit(
            f, xdata, ydata, p0, sigma, absolute_sigma, check_finite, bounds, method, jac, **kwargs
        )

        # user has requested that full_output be returned, but the method
        # isn't capable, fill in the blanks.
        if return_full and not can_full_output:
            return res[0], res[1], None, "No error", 1
        else:
            return res

    elif method == "ls":
        _wrap_func = _wrap_func_ls
        _wrap_jac = _wrap_jac_ls
    elif method == "mle":
        _wrap_func = _wrap_func_mle
        _wrap_jac = _wrap_jac_mle
    else:
        raise TypeError("Method {} not recognized".format(method))

    if np.any(np.asarray(bounds[0]) != -np.inf) or np.any(np.asarray(bounds[1]) != np.inf):
        raise NotImplementedError("Bounds has not been implemented")

    if sigma is not None:
        raise NotImplementedError("Weighting has not been implemented")
    else:
        transform = None

    if not callable(jac):
        raise NotImplementedError("An analytic Jacobian is required")
    if not kwargs.get("col_deriv", True):
        raise NotImplementedError("col_deriv=False has not been implemented")

    # initialize p0 with standard LM
    # The legacy custom col_deriv flag has different semantics from SciPy's.
    initial_kwargs = {key: value for key, value in kwargs.items() if key != "col_deriv"}
    res = scipy.optimize.curve_fit(
        f,
        xdata,
        ydata,
        p0,
        sigma,
        absolute_sigma,
        check_finite,
        bounds,
        None,
        jac,
        **initial_kwargs
    )

    # grab p0
    logger.debug("Initialized p0")
    p0 = res[0]

    # NaNs can not be handled
    if check_finite:
        ydata = np.asarray_chkfinite(ydata)
    else:
        ydata = np.asarray(ydata)

    if isinstance(xdata, (list, tuple, np.ndarray)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        if check_finite:
            xdata = np.asarray_chkfinite(xdata)
        else:
            xdata = np.asarray(xdata)

    func = _wrap_func(f, xdata, ydata, transform)
    if callable(jac):
        jac = _wrap_jac(jac, xdata, transform)

    res = lm(func, p0, Dfun=jac, full_output=1, method=method, **kwargs)
    popt, pcov, infodict, errmsg, info = res

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = la.svd(infodict["fjac"], full_matrices=False)
    threshold = np.finfo(float).eps * max(infodict["fjac"].shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s**2, VT)

    if info not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal parameters not found: " + errmsg)

    if return_full:
        return popt, pcov, infodict, errmsg, info
    else:
        return popt, pcov
