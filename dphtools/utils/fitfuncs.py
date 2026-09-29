#!/usr/bin/env python
# -*- coding: utf-8 -*-
# fitfuncs.py
"""
Various functions for fitting things.

Copyright (c) 2021, David Hoffman
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import signaltools as sig
from scipy.special import betaln, exprel, gammaln, zeta
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
            # Choose guess partitions independently of the x-axis origin.
            elapsed = xdata_fixed - xdata_fixed[0]
            split_points = np.logspace(
                np.log(elapsed[elapsed > 0].min()),
                np.log(elapsed.max()),
                components + 1,
                base=np.e,
            )
            # convert to indices
            split_idxs = np.searchsorted(elapsed, split_points)
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
        return zeta(alpha, x) / bottom

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


def _powerlaw_log_ratio(values, lower):
    """Compute log ratios without overflowing wide ratios or cancelling close ones."""
    values = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        close = np.log1p((values - lower) / lower)
        return np.where(np.isfinite(close), close, np.log(values) - np.log(lower))


def _powerlaw_log_mean(shape, width):
    """Return the mean log ratio for a bounded continuous power law."""
    argument = shape * width
    if argument < 1e-3:
        return width * (0.5 - argument / 12 + argument**3 / 720 - argument**5 / 30240)
    return 1 / shape - width * np.exp(-argument) / -np.expm1(-argument)


def _powerlaw_discrete_partition(alpha, lower, upper):
    """Evaluate the discrete partition and log moment, scaled by lower**alpha.

    Sum the first 64 masses directly, then use Euler--Maclaurin for the
    remaining integer support. Its integral, endpoint and six Bernoulli
    corrections are differentiated together, giving the likelihood score.
    A finite upper endpoint subtracts the corresponding tail corrections;
    it never renormalizes an unbounded model at the observed maximum.
    """
    if lower > 2**53 - 1 or (np.isfinite(upper) and upper > 2**53 - 1):
        raise RuntimeError("Integer support exceeds exact float64 spacing.")
    stop = lower + 64 if np.isinf(upper) else min(lower + 64, upper + 1)
    logs = _powerlaw_log_ratio(np.arange(lower, stop, dtype=float), lower)
    weights = np.exp(-alpha * logs)
    total, moment = weights.sum(), np.dot(weights, logs)
    if stop <= upper:
        start_log = float(_powerlaw_log_ratio(stop, lower))
        start_weight = np.exp(-alpha * start_log)
        shape = alpha - 1
        if np.isinf(upper):
            integral = start_weight * stop / shape
            integral_mean = start_log + 1 / shape
            endpoints = np.array([stop], dtype=float)
            signs = np.array([1.0])
        else:
            width = float(_powerlaw_log_ratio(upper + 1, stop))
            integral = start_weight * stop * width * exprel(-shape * width)
            integral_mean = start_log + _powerlaw_log_mean(shape, width)
            endpoints = np.array([stop, upper + 1], dtype=float)
            signs = np.array([1.0, -1.0])
        logs = _powerlaw_log_ratio(endpoints, lower)
        weights = signs * np.exp(-alpha * logs)
        total += integral + 0.5 * weights.sum()
        moment += integral * integral_mean + 0.5 * np.dot(weights, logs)
        rising = alpha / endpoints
        reciprocal = 1 / alpha
        coefficients = (
            1 / 12,
            -1 / 720,
            1 / 30240,
            -1 / 1209600,
            1 / 47900160,
            -691 / 1307674368000,
        )
        for index, coefficient in enumerate(coefficients):
            correction = coefficient * weights * rising
            log_correction = correction * (logs - reciprocal)
            total += correction.sum()
            moment += log_correction.sum()
            for offset in (2 * index + 1, 2 * index + 2):
                rising *= (alpha + offset) / endpoints
                reciprocal += 1 / (alpha + offset)
        # Reject unresolved tail corrections rather than passing a visibly
        # unconverged series evaluation to the likelihood solver.
        if np.abs(correction).sum() > 1e-13 * total or np.abs(log_correction).sum() > 1e-13 * max(
            moment, np.finfo(float).tiny
        ):
            raise RuntimeError("Discrete partition precision is insufficient.")
    if not np.isfinite(total + moment) or total <= 0 or moment < 0:
        raise RuntimeError("Discrete partition is outside numerical range.")
    return total, moment


class PowerLaw(object):
    """Class for fitting and testing power law distributions."""

    def __init__(self, data):
        """Represent integer discrete samples or real floating continuous samples."""
        if (
            not isinstance(data, np.ndarray)
            or data.ndim != 1
            or not data.size
            or data.dtype.kind not in "iuf"
            or not np.isfinite(data).all()
            or np.any(data < 0)
        ):
            raise ValueError("Data must be a nonempty finite nonnegative numeric NumPy vector.")
        self.data = data
        self._discrete = np.issubdtype(data.dtype, np.integer)

    def _cutoff(self, value, discrete=False):
        """Validate a positive finite scalar support bound."""
        scalar = np.asarray(value)
        if (
            scalar.ndim != 0
            or scalar.dtype.kind not in "iuf"
            or not np.isfinite(scalar)
            or scalar <= 0
        ):
            raise ValueError("Cutoffs must be positive finite real scalars.")
        if discrete and scalar != np.floor(scalar):
            raise ValueError("Discrete lower cutoffs must be integers.")
        return int(scalar) if discrete else float(scalar)

    def fit(self, xmin=None, xmin_max=200, opt_max=False):
        """Fit inclusive supports and select by ordinary Kolmogorov--Smirnov distance.

        Automatic lower candidates retain at least 50 observations. Upper
        optimization compares observed endpoints and infinite upper support.
        Only a successful selection replaces fitted state and diagnostics.
        """
        if xmin is None:
            cap = self._cutoff(xmin_max)
            lowers = [
                value.item()
                for value in np.unique(self.data)
                if 0 < value <= cap
                and np.count_nonzero(self.data >= value) >= 50
                and np.any(self.data > value)
            ]
        else:
            lowers = [self._cutoff(xmin, self._discrete)]
        candidates = []
        for lower in lowers:
            tail = self.data[self.data >= lower]
            uppers = list(np.unique(tail)) + [np.inf] if opt_max else [np.inf]
            for upper in uppers:
                retained = tail[tail <= upper]
                if opt_max and len(retained) < 50:
                    continue
                try:
                    c, alpha = self._fit_support(retained, lower, upper)
                except ValueError:
                    if opt_max:
                        continue
                    raise
                distance = self._ks_distance(retained, lower, upper, alpha)
                candidates.append((distance, len(retained), lower, upper, c, alpha))
        if not candidates:
            raise ValueError("No eligible power-law support has a finite alpha > 1 optimum.")
        minimum = min(candidate[0] for candidate in candidates)
        selected = min(
            (candidate for candidate in candidates if candidate[0] <= minimum + 1e-10),
            key=lambda candidate: (-candidate[1], candidate[2], -candidate[3]),
        )
        _, count, lower, upper, c, alpha = selected
        self.xmin, self.xmax, self.C, self.alpha = lower, upper, c, alpha
        self.ks_statistics = np.array([candidate[0] for candidate in candidates])
        error = (alpha - 1) / np.sqrt(count)
        if self._discrete:
            self.alpha_error = error
        else:
            self.alpha_std = error
        return self.C, self.alpha

    @property
    def clipped_data(self):
        """Return observations within both inclusive fitted bounds."""
        return self.data[(self.data >= self.xmin) & (self.data <= self.xmax)]

    def intercept(self, value=1):
        """Return the intercept calculated from power law values."""
        return power_intercept((self.C * len(self.data), self.alpha), value)

    def percentile(self, value):
        """Return the intercept calculated from power law values."""
        return power_percentile(value, (self.C * len(self.data), self.alpha), self.xmin)

    def _fit_support(self, data, lower, upper):
        """Fit only the exponent on fixed support without modifying instance state."""
        if not len(data) or not np.any(data > lower) or upper <= lower:
            raise ValueError("Support has no finite identifiable power-law optimum.")
        values, counts = np.unique(data, return_counts=True)
        mean = np.dot(counts / len(data), _powerlaw_log_ratio(values, lower))
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError("Sample log ratios are outside numerical range.")
        if self._discrete:

            def score(alpha):
                """Return the exact discrete likelihood score per observation."""
                total, moment = _powerlaw_discrete_partition(alpha, lower, upper)
                return moment / total - mean

            left = np.nextafter(1.0, 2.0) if np.isinf(upper) else 1.0
            if score(left) <= 0:
                if np.isfinite(upper):
                    raise ValueError("Bounded likelihood has no interior alpha > 1 optimum.")
                raise RuntimeError("Exponent is too close to one to represent.")
            right = 2.0
            while score(right) > 0:
                right = 1 + 2 * (right - 1)
                if not np.isfinite(right):
                    raise RuntimeError("Cannot bracket a finite power-law exponent.")
            try:
                alpha = brentq(score, left, right, xtol=5e-14, rtol=1e-14)
            except ValueError as error:
                raise RuntimeError("Discrete likelihood root could not be resolved.") from error
            total, _ = _powerlaw_discrete_partition(alpha, lower, upper)
            log_c = alpha * np.log(lower) - np.log(total)
        else:
            shape = 1 / mean
            log_fraction = 0.0
            if np.isfinite(upper):
                width = float(_powerlaw_log_ratio(upper, lower))
                if mean >= width / 2:
                    raise ValueError("Bounded likelihood has no interior alpha > 1 optimum.")
                try:
                    shape = brentq(
                        lambda value: _powerlaw_log_mean(value, width) - mean,
                        0.0,
                        shape,
                        xtol=np.finfo(float).tiny,
                        rtol=1e-14,
                    )
                except ValueError as error:
                    raise RuntimeError(
                        "Continuous likelihood root could not be resolved."
                    ) from error
                log_fraction = np.log(-np.expm1(-shape * width))
            alpha = 1 + shape
            log_c = np.log(shape) + shape * np.log(lower) - log_fraction
        with np.errstate(over="ignore", under="ignore"):
            c = np.exp(log_c)
        if not np.isfinite(c) or c <= 0 or not np.isfinite(alpha) or alpha <= 1:
            raise RuntimeError("Power-law parameters are outside finite numerical range.")
        return c, alpha

    def _cdf(self, values, lower, upper, alpha):
        """Evaluate the normalized fitted CDF, including discrete support gaps."""
        values = np.asarray(values, dtype=float)
        result = np.zeros(values.shape)
        inside = (values >= lower) & (values < upper)
        result[values >= upper] = 1.0
        if self._discrete:
            total, _ = _powerlaw_discrete_partition(alpha, lower, upper)
            for index in np.flatnonzero(inside):
                start = int(np.floor(values.flat[index])) + 1
                tail, _ = _powerlaw_discrete_partition(alpha, start, upper)
                log_survival = np.log(tail / total) - alpha * float(
                    _powerlaw_log_ratio(start, lower)
                )
                result.flat[index] = -np.expm1(log_survival)
        else:
            shape = alpha - 1
            fraction = (
                1.0 if np.isinf(upper) else -np.expm1(-shape * _powerlaw_log_ratio(upper, lower))
            )
            result[inside] = (
                -np.expm1(-shape * _powerlaw_log_ratio(values[inside], lower)) / fraction
            )
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise RuntimeError("Power-law CDF is outside numerical range.")
        return result

    def _ks_distance(self, data, lower, upper, alpha):
        """Compare both sides of each empirical jump with the matching model sides."""
        values, counts = np.unique(data, return_counts=True)
        after = counts.cumsum() / len(data)
        before = after - counts / len(data)
        right = self._cdf(values, lower, upper, alpha)
        left = (
            self._cdf(values.astype(float) - 1, lower, upper, alpha) if self._discrete else right
        )
        return float(max(np.max(np.abs(after - right)), np.max(np.abs(before - left))))

    def gen_power_law(self):
        """Draw one independent observation per retained sample on fitted support."""
        uniform = np.random.random(len(self.clipped_data))
        if self._discrete:
            cache = {}

            def cdf(points):
                """Cache repeated integer CDF evaluations during inversion."""
                for point in np.unique(points):
                    if point not in cache:
                        cache[point] = self._cdf(
                            np.array([point]), self.xmin, self.xmax, self.alpha
                        )[0]
                return np.array([cache[point] for point in points])

            upper = self.xmin if np.isinf(self.xmax) else int(self.xmax)
            while cdf([upper])[0] < uniform.max():
                upper *= 2
                if upper > 2**53 - 1:
                    raise RuntimeError("A sampled integer exceeds exact numerical support.")
            left = np.full(len(uniform), self.xmin - 1, dtype=np.int64)
            right = np.full(len(uniform), upper, dtype=np.int64)
            while np.any(right - left > 1):
                active = right - left > 1
                middle = (left[active] + right[active]) // 2
                below = cdf(middle) < uniform[active]
                left[active] = np.where(below, middle, left[active])
                right[active] = np.where(below, right[active], middle)
            return right
        shape = self.alpha - 1
        fraction = (
            1.0
            if np.isinf(self.xmax)
            else -np.expm1(-shape * _powerlaw_log_ratio(self.xmax, self.xmin))
        )
        with np.errstate(over="ignore"):
            samples = self.xmin * np.exp(-np.log1p(-uniform * fraction) / shape)
        if not np.isfinite(samples).all() or np.any(samples > self.xmax):
            raise RuntimeError("A sampled value exceeds finite numerical range.")
        return samples

    def calculate_p(self, num=1000):
        """Return a fixed-support, refitted-exponent conditional bootstrap fraction."""
        if isinstance(num, (bool, np.bool_)) or not isinstance(num, (int, np.integer)) or num <= 0:
            raise ValueError("Bootstrap iterations must be a positive integer.")
        ks_data = self._ks_distance(self.clipped_data, self.xmin, self.xmax, self.alpha)
        ks_tests = []
        for index in range(num):
            try:
                samples = self.gen_power_law()
                _, alpha = self._fit_support(samples, self.xmin, self.xmax)
                ks_tests.append(self._ks_distance(samples, self.xmin, self.xmax, alpha))
            except (ValueError, RuntimeError) as error:
                raise RuntimeError(
                    "Bootstrap replicate {} failed: {}".format(index + 1, error)
                ) from error
        self.ks_data = ks_data
        self.ks_tests = np.asarray(ks_tests)
        return np.count_nonzero(self.ks_tests >= self.ks_data) / num

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
        return self.C * x ** (-self.alpha)

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
    """Fit a zero-truncated Poisson rate by conditional maximum likelihood.

    Parameters
    ----------
    data : ndarray
        Nonempty one-dimensional array of finite positive integer-valued
        observations with real numeric storage. The input is not modified.

    Returns
    -------
    rate : float
        Positive rate satisfying ``rate / (1 - exp(-rate)) = mean(data)``.

    Raises
    ------
    ValueError
        If observations are invalid or all ones, for which no positive rate
        attains the likelihood supremum.
    RuntimeError
        If an eligible sample cannot be fitted numerically.
    """
    if (
        not isinstance(data, np.ndarray)
        or data.ndim != 1
        or data.size == 0
        or data.dtype.kind not in "iuf"
    ):
        raise ValueError("Expected a nonempty one-dimensional real numeric NumPy array")
    if not np.isfinite(data).all() or np.any(data <= 0) or np.any(data != np.floor(data)):
        raise ValueError("Observations must be finite positive integer-valued counts")
    if np.all(data == 1):
        raise ValueError("All-ones data has no positive maximum-likelihood rate")

    try:
        mean = float(np.mean(data, dtype=np.float64))
        if not np.isfinite(mean) or mean <= 1:
            raise RuntimeError("The sample mean cannot be represented for fitting")
        return _fit_ztp_mean(mean)
    except (ValueError, ArithmeticError) as error:
        raise RuntimeError("Fitting zero-truncated Poisson failed") from error


def _fit_ztp_mean(mean):
    """Fit the conditional Poisson rate for a finite binary64 mean above one."""

    def mean_residual(lam):
        """Evaluate the likelihood equation, including its limit at zero."""
        return (1.0 if lam == 0 else lam / -np.expm1(-lam)) - mean

    # The conditional mean increases from 1 and exceeds lam for lam > 0.
    # Thus [0, mean] brackets the unique positive likelihood optimum.
    rate = brentq(mean_residual, 0.0, mean)
    if not np.isfinite(rate) or rate <= 0:
        raise RuntimeError("Fitting zero-truncated Poisson returned an invalid rate")
    return rate


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
    """Fit a zero-truncated negative binomial by conditional maximum likelihood.

    Parameters
    ----------
    data : ndarray
        Nonempty one-dimensional real numeric array of finite positive
        integer-valued counts. The input is not modified.
    x0 : pair of float, optional
        Positive finite initial shape and untruncated mean. The shape seeds
        the global profile search; the mean is profiled out analytically.
        The pair is not modified and does not select a different estimator.

    Returns
    -------
    parameters : ndarray
        Finite positive shape and untruncated mean, in that order.

    Raises
    ------
    ValueError
        If observations or the initial pair are invalid, or all counts are one.
    RuntimeError
        If a finite optimum cannot be resolved against both limiting models,
        or the global profile search and numerical checks do not converge.

    Notes
    -----
    The entire positive shape domain is searched in compact coordinates,
    including the optimized logarithmic-series and truncated-Poisson limits.
    Competing interior maxima are refined on two successively finer meshes.
    This is a numerical global check, not a proof of profile unimodality.
    """
    if (
        not isinstance(data, np.ndarray)
        or data.ndim != 1
        or data.size == 0
        or data.dtype.kind not in "iuf"
    ):
        raise ValueError("Expected a nonempty one-dimensional real numeric NumPy array")
    if not np.isfinite(data).all() or np.any(data <= 0) or np.any(data != np.floor(data)):
        raise ValueError("Observations must be finite positive integer-valued counts")
    initial = np.asarray(x0)
    if (
        initial.shape != (2,)
        or initial.dtype.kind not in "iuf"
        or not np.isfinite(initial).all()
        or np.any(initial <= 0)
    ):
        raise ValueError("Initial shape and mean must be a finite positive numeric pair")
    if np.all(data == 1):
        raise ValueError("All-ones data has no finite positive maximum-likelihood parameters")

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            seed = float(initial[0] / (1.0 + initial[0]))
            counts, frequencies = np.unique(data, return_counts=True)
            return _fit_ztnb_profile(counts, frequencies, seed)
    except (ValueError, ArithmeticError) as error:
        raise RuntimeError(
            "Could not establish a finite zero-truncated negative-binomial fit"
        ) from error


def _fit_ztnb_profile(counts, frequencies, seed, maxiter=500):
    """Resolve the conditional likelihood profile of an integer-count histogram.

    The distinct positive integer counts and their positive integer observation
    multiplicities are not modified. The seed is a compactified initial shape.
    Positive integer maxiter is forwarded to each bounded scalar refinement as
    its stopping option. Actual iteration/evaluation counts follow SciPy's
    solver semantics; this is not an aggregate work or time limit and does not
    constrain the mean root solves or mesh evaluations.
    """
    counts = counts.astype(np.float64)
    weights = frequencies / sum(map(int, frequencies))
    mean = counts @ weights
    if not np.isfinite(mean) or mean <= 1:
        raise RuntimeError("The sample mean cannot be represented for fitting")

    root_options = dict(xtol=np.finfo(float).tiny, rtol=8 * np.finfo(float).eps)
    rate = brentq(lambda z: 1 / exprel(-z) - mean, 0, mean, **root_options)
    t0 = brentq(lambda t: exprel(t) - mean, 0, 2 * np.log(mean) + 2, **root_options)
    log_counts = np.log(counts)
    log_factorials = weights @ gammaln(counts + 1)
    boundaries = np.array(
        [
            mean * np.log(-np.expm1(-t0)) - np.log(t0) - weights @ log_counts,
            mean * np.log(rate) - log_factorials - rate - np.log(-np.expm1(-rate)),
        ]
    )
    # Allow for cancellation in log probabilities; replication must not change
    # the estimator, so all likelihoods and tolerances are per observation.
    tolerance = 1024 * np.finfo(float).eps * (1 + log_factorials + mean * np.log1p(mean))

    def profile(w):
        """Evaluate an interior shape after matching the conditional mean."""
        shape = w / (1 - w)
        # With t=log(1+m/a), z=a*t, write t=(1-w)*h and z=w*h.
        # E[X|X>0]=exprel(t)/exprel(-z). This remains stable at both
        # boundaries, and t<=t0, z<=rate bracket the unique mean root.
        upper = min(t0 / (1 - w), rate / w) * (1 + 1e-12)
        h = brentq(
            lambda h: exprel((1 - w) * h) / exprel(-w * h) - mean,
            0,
            upper,
            **root_options,
        )
        t, z = (1 - w) * h, w * h
        # (a)_k/k! = 1/(k*B(a,k)); no PMF underflow or large gamma values.
        likelihood = (
            -weights @ (log_counts + betaln(shape, counts))
            + mean * np.log(-np.expm1(-t))
            - z
            - np.log(-np.expm1(-z))
        )
        return likelihood, np.array([shape, z * exprel(t)])

    def objective(w):
        """Attach the exact limiting likelihoods to the compact interval."""
        if w == 0:
            return -boundaries[0]
        if w == 1:
            return -boundaries[1]
        return -profile(w)[0]

    solutions = []
    for size in (129, 513):
        # Cluster near both boundaries and also sample the caller's start.
        mesh = np.unique(np.r_[np.sin(np.linspace(0, np.pi / 2, size)) ** 2, seed])
        values = np.array([objective(w) for w in mesh])
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite negative-binomial profile likelihood")
        minima = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])) + 1
        # Refine edge intervals even when their best sampled value is a limit:
        # a near-boundary interior maximum must not be replaced by a shape cap.
        minima = np.unique(np.r_[1, minima, len(mesh) - 2])
        candidates = [(values.min(), mesh[values.argmin()])]
        for index in minima:
            optimum = minimize_scalar(
                objective,
                bounds=(mesh[index - 1], mesh[index + 1]),
                method="bounded",
                options={"xatol": 1e-13, "maxiter": maxiter},
            )
            if not optimum.success:
                raise RuntimeError(
                    f"Negative-binomial profile refinement failed: {optimum.message}"
                )
            candidates.append((optimum.fun, optimum.x))
        solutions.append(min(candidates))

    best, w = solutions[-1]
    if w in (0, 1) or -best - boundaries.max() <= tolerance:
        boundary = ("logarithmic-series", "Poisson")[boundaries.argmax()]
        raise RuntimeError(
            f"Could not establish a finite optimum: the {boundary} boundary is competitive "
            "at numerical precision"
        )
    likelihood, parameters = profile(w)
    previous_value, previous_w = solutions[0]
    if (
        previous_w in (0, 1)
        or abs(best - previous_value) > tolerance
        or not np.allclose(parameters, profile(previous_w)[1], rtol=1e-4, atol=0)
    ):
        raise RuntimeError("Global negative-binomial profile refinement did not converge")

    # Check stationarity independently of the local optimizer's success flag.
    # Differences in log(shape) avoid an automatically tiny raw score at infinity.
    shape, fitted_mean = parameters
    adjacent_shapes = shape * np.exp(np.array([-1e-4, 1e-4]))
    adjacent = np.array([profile(a / (1 + a))[0] for a in adjacent_shapes])
    conditional_mean = fitted_mean / -np.expm1(-shape * np.log1p(fitted_mean / shape))
    if (
        not np.isfinite(parameters).all()
        or np.any(parameters <= 0)
        or not np.isclose(conditional_mean, mean, rtol=1e-10, atol=0)
        or abs(adjacent[1] - adjacent[0]) / 2e-4 > 1e-6
        or adjacent.max() > likelihood + tolerance
    ):
        raise RuntimeError("Negative-binomial likelihood equations could not be resolved")

    return parameters
