#!/usr/bin/env python
# -*- coding: utf-8 -*-
# fitfuncs.py
"""
Various functions for fitting things.

Copyright (c) 2021, David Hoffman
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import signaltools as sig
from scipy.special import zeta
from scipy.stats import nbinom

from .lm import curve_fit


def multi_exp(xdata, *args):
    r"""Sum of exponentials.
    
    .. math:: y = bias + \sum_n A_i e^{-k_i x}
    """
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
    """Jacopian for multi_exp."""
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
    """Single exponential function.
    
    .. math:: y = amp e^{-rate xdata} + offset
    """
    return multi_exp(xdata, amp, rate, offset)


def _estimate_exponent_params(data, xdata):
    """Estimate exponent params."""
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
    """Not implemented."""
    raise NotImplementedError


def exponent_fit(data, xdata=None, offset=True):
    """Fit data to a single exponential function."""
    return multi_exp_fit(data, xdata, components=1, offset=offset)


def multi_exp_fit(data, xdata=None, components=None, offset=True, **kwargs):
    """Fit data to a multi-exponential function.

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
    """Estimate the best fit parameters for a power law by linearly fitting the loglog plot."""
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
    """Test power law params."""
    a, b = popt
    assert a > 0, "Scale invalid"
    assert b > 1, "Exponent invalid"
    assert xmin > 0, "xmin invalid"


def power_percentile(p, popt, xmin=1):
    """Percentile of a single power law function."""
    assert 0 <= p <= 1, "percentile invalid"
    _test_pow_law(popt, xmin)
    a, b = popt
    x0 = (1 - p) ** (1 / (1 - b)) * xmin
    return x0


def power_percentile_inv(x0, popt, xmin=1):
    """Given an x value what percentile of the power law function does it correspond to."""
    _test_pow_law(popt, xmin)
    a, b = popt
    p = 1 - (x0 / xmin) ** (1 - b)
    return p


def power_intercept(popt, value=1):
    """At what x value does the function reach value."""
    a, b = popt
    assert a > 0, f"a = {value}"
    assert value > 0, f"value = {value}"
    return (a / value) ** (1 / b)


def power_law(xdata, *args):
    """Multi-power law function."""
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
    """Jacobian for a multi-power law function."""
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


def powerlaw_prng(alpha, xmin=1, xmax=1e7):
    """Calculate a psuedo random variable drawn from a discrete power law distribution with scale parameter alpha and xmin."""
    # don't want to waste time recalculating this
    bottom = zeta(alpha, xmin)

    def P(x):
        """Cumulative distribution function."""
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
    """Class for fitting and testing power law distributions."""

    def __init__(self, data):
        """Object representing power law data.
        
        Pass in data, it will be automagically determined to be
        continuous (float/inexact datatype) or discrete (integer datatype)
        """
        self.data = data
        # am I discrete data
        self._discrete = np.issubdtype(data.dtype, np.integer)

    def fit(self, xmin=None, xmin_max=200, opt_max=False):
        """Fit the data, if xmin is none then estimate it."""
        if self._discrete:
            # discrete fitting
            if opt_max:
                # we should optimize the maximum x
                def func(m):
                    """Optimization function, m is max value."""
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
                    """Update self then run KS_test."""
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
        """Return data clipped to xmin."""
        return self.data[self.data >= self.xmin]

    def intercept(self, value=1):
        """Return the intercept calculated from power law values."""
        return power_intercept((self.C * len(self.data), self.alpha), value)

    def percentile(self, value):
        """Return the intercept calculated from power law values."""
        return power_percentile(value, (self.C * len(self.data), self.alpha), self.xmin)

    def gen_power_law(self):
        """x.append(xmin*pow(1.-random(),-1./(alpha-1.)))."""
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
        """Make a bunch of fake data and run the KS_test on it."""
        ks_data = self._KS_test_discrete()
        # normalizing constant
        ks_tests = []
        for i in range(num):
            fake_data = self.gen_power_law()
            power = PowerLaw(fake_data)
            power.fit(self.xmin)
            ks_tests.append(power._KS_test_discrete())

        ks_tests = np.asarray(ks_tests)
        self.ks_tests = ks_tests
        self.ks_data = ks_data
        return (ks_tests > ks_data).sum() / len(ks_tests)

    def _convert_to_probability_discrete(self):
        """Convert to a probability distribution."""
        y = np.bincount(self.data)
        x = np.arange(len(y))

        # calculate the normalization constant from xmin onwards
        N = y[self.xmin :].sum()
        y = y / N
        return x, y, N

    def _power_law_fit_discrete(self, x):
        """Compute the power_law fit."""
        return x ** (-self.alpha) / zeta(self.alpha, self.xmin)

    def _KS_test_discrete(self):
        """Kolmogorov–Smirnov or KS statistic."""
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
        """Fit a discrete power-law to data."""
        # update internal xmin
        self.xmin = xmin

        # clip data to be greater than xmin
        data = self.clipped_data

        # calculate the log sum of the data
        lsum = np.log(data).sum()
        # and length, don't want to recalculate these during optimization
        n = len(data)

        def nll(alpha, xmin):
            """Negative log-likelihood of discrete power law."""
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
        """Fit a continuous power-law to data."""
        data = self.data

        data = data[data >= xmin]

        alpha = 1 + len(data) / np.log(data / xmin).sum()
        C = (alpha - 1) * xmin ** (alpha - 1)

        self.C = C
        self.alpha = alpha
        self.alpha_std = (alpha - 1) / np.sqrt(len(data))

        return C, alpha

    def plot(self, ax=None, density=True, norm=False):
        """Plot data."""
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
    """Fit the data assuming it follows a zero-truncated Poisson model."""
    n = len(data)
    sum_x = data.sum()
    # ignore the constant offset
    # sum_log_x_fac = np.log(gamma(data)).sum()

    def negloglikelihood(lam):
        """Negative log-likelihood of ZTP."""
        # ignore the constant offset
        return n * np.log(np.exp(lam) - 1) - np.log(lam) * sum_x  # + sum_log_x_fac

    with np.errstate(divide="ignore", invalid="ignore"):
        opt = minimize_scalar(negloglikelihood)

    if not opt.success:
        raise RuntimeError("Fitting zero-truncated poisson failed")

    return opt.x


def NegBinom(a, m):
    """Convert scipy's definition to mean and shape."""
    r = a
    p = m / (m + r)
    return nbinom(r, 1 - p)


def negloglikelihoodNB(args, x):
    """Negative log likelihood for negative binomial."""
    a, m = args
    numerator = NegBinom(a, m).pmf(x)
    return -np.log(numerator).sum()


def negloglikelihoodZTNB(args, x):
    """Negative log likelihood for zero truncated negative binomial."""
    a, m = args
    denom = 1 - NegBinom(a, m).pmf(0)

    return len(x) * np.log(denom) + negloglikelihoodNB(args, x)


def fit_ztnb(data, x0=(0.5, 0.5)):
    """Fit the data assuming it follows a zero-truncated Negative Binomial model."""
    opt = minimize(negloglikelihoodZTNB, x0, (data,), bounds=((0, np.inf), (0, np.inf)))

    if not opt.success:
        raise RuntimeError("Fitting zero-truncated negative binomial", opt)

    return opt.x
