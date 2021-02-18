#!/usr/bin/env python
# -*- coding: utf-8 -*-
# fitfuncs.py
"""
Various functions for fitting things

Copyright (c) 2021, David Hoffman
"""

import matplotlib.pyplot as plt
import numpy as np

from .lm import curve_fit


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
        with np.errstate(divide="ignore"):
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
