#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_fitfuncs.py
"""
Testing for fitfuncs

Copyright (c) 2021, David Hoffman
"""


import unittest

import numpy as np
from dphtools.utils.fitfuncs import exponent, exponent_fit
from numpy.testing import assert_allclose


class TestFFTPad(unittest.TestCase):
    """This is not even close to testing edge cases"""

    def setUp(self):
        self.x = np.linspace(0, 10)
        self.params = (10, 3, 5)
        self.data = exponent(self.x, *self.params)
        self.data_noisy = np.random.randn(self.x.size)

    def test_positive(self):
        """Test a decaying signal"""
        popt, pcov = exponent_fit(self.data, self.x)
        assert_allclose(popt, self.params, rtol=1e-3)

    def test_negative(self):
        """Test a rising signal"""
        popt, pcov = exponent_fit(-self.data, self.x)
        amp, k, offset = self.params
        new_params = -amp, k, -offset
        assert_allclose(popt, new_params, rtol=1e-3)
