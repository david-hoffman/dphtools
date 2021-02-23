#!/usr/bin/env python
# -*- coding: utf-8 -*-
# histstats.py
"""
Small utility functions for calculating histogram stats.

https://en.wikipedia.org/wiki/Standardized_moment

Copyright (c) 2016, David Hoffman
"""

from functools import partial

import numpy as np


def _standard_bins(weights, bins):
    """Standardize bins input."""
    # add checks for bins here
    if bins is None:
        bins = np.arange(len(weights))
    return bins


def hist_mean(weights, bins=None):
    """Histogram mean."""
    bins = _standard_bins(weights, bins)
    return (weights * bins).sum() / weights.sum()


def hist_var(weights, bins=None):
    """Histogram variance."""
    bins = _standard_bins(weights, bins)
    mean = hist_mean(weights, bins)
    return ((weights * (bins - mean)) ** 2).sum() / weights.sum()


def hist_moment(weights, bins=None, k=3):
    """Generalized histogram moment.
    
    Defaults to the third one
    """
    bins = _standard_bins(weights, bins)
    mean = hist_mean(weights, bins)
    std = np.sqrt(hist_var(weights, bins))
    mu_k = ((weights * (bins - mean)) ** k).sum() / weights.sum()
    return mu_k / std ** k


hist_skew = partial(hist_moment, k=3)
hist_kurtosis = partial(hist_moment, k=4)
