#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_dphutils.py
"""
Testing for utils.

Copyright (c) 2021, David Hoffman
"""

import unittest
from itertools import product

import numpy as np
import pytest

# import the package to test
from dphtools.utils import (
    anscombe,
    anscombe_inv,
    fft_gaussian_filter,
    fft_pad,
    padding_slices,
    radial_profile,
    slice_maker,
    win_nd,
)
from numpy.fft import fftshift, ifftshift
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.fft import next_fast_len
from scipy.ndimage.filters import gaussian_filter


rng = np.random.default_rng(12345)


class TestFFTPad(unittest.TestCase):
    def setUp(self):
        pass

    def test_new_shape_no_size(self):
        """Test the make a new shape with even and odd numbers when no size is specified, i.e. test auto padding."""
        oldshape = (2 * 17, 17)
        data = np.zeros(oldshape)
        newshape = tuple(next_fast_len(s) for s in oldshape)
        newdata = fft_pad(data)
        assert newshape == newdata.shape

    def test_new_shape_one_size(self):
        """
        Make sure the new shape has the same dimensions when one is given.
        """
        oldshape = (10, 20, 30)
        data = rng.standard_normal(oldshape)
        newsize = 50
        newdata = fft_pad(data, newsize)
        assert (newsize,) * newdata.ndim == newdata.shape

    def test_new_shape_multiple(self):
        """
        Make sure the new shape has the same dimensions when one is given.
        """
        oldshape = (10, 20, 30, 40)
        data = rng.standard_normal(oldshape)
        newsize = (50, 40, 30, 100)
        newdata = fft_pad(data, newsize)
        assert newsize == newdata.shape

    def test_smaller_shape(self):
        """Test that cropping works as expected."""
        oldshape = rng.integers(10, 200)
        newshape = rng.integers(5, oldshape)
        data = np.ones(oldshape)
        assert data.shape == oldshape
        pad_data = fft_pad(data, newshape)
        assert pad_data.shape == newshape

    def test_right_position_cases(self):
        """make sure that center stays centered (for ffts) all cases."""
        cases = (
            (14, 34),  # even -> even
            (14, 35),  # even -> odd
            (17, 34),  # odd -> even
            (17, 35),  # odd -> odd
        )
        # same cases
        same = ((34, 34), (35, 35))  # odd -> odd  # even -> even
        # try the cropping version too
        rev_cases = tuple((j, i) for i, j in cases)
        for oldshape, newshape in cases + same + rev_cases:
            data = np.zeros(oldshape)
            data[0] = 1
            data_centered = ifftshift(data)
            data_padded = fft_pad(data_centered, newshape)
            assert fftshift(data_padded)[0] == 1

    def test_right_position_multidimensional(self):
        """make sure that center stays centered (for ffts) fuzzy test to see if I missed anything."""
        for i in range(10):
            dims = rng.integers(1, 4)
            oldshape = rng.integers(10, 100, dims)
            newshape = rng.integers(10, 100, dims)
            data = np.zeros(oldshape)
            zero_loc = (0,) * dims
            data[zero_loc] = 1
            data_centered = ifftshift(data)
            data_padded = fft_pad(data_centered, newshape)
            assert fftshift(data_padded)[zero_loc] == 1


def test_radprof_complex():
    """Testing rad prof for complex values."""
    result = radial_profile(np.ones((11, 11)) + np.ones((11, 11)) * 1j)
    avg = np.ones(8) + np.ones(8) * 1j
    assert_allclose(result[0], avg)
    std = np.zeros(8) + np.zeros(8) * 1j
    assert_allclose(result[1], std)


def test_win_nd():
    """Testing the size of win_nd."""
    shape = (128, 65, 17)
    result = win_nd(shape)
    assert shape == result.shape


def test_anscombe():
    """Test anscombe function."""
    # https://en.wikipedia.org/wiki/Anscombe_transform
    data = rng.poisson(100, (128, 128, 128))
    assert_almost_equal(data.mean(), 100, 1), "Data not generated properly!"
    ans_data = anscombe(data)
    assert_almost_equal(ans_data.var(), 1, 2)
    in_ans_data = anscombe_inv(ans_data)
    assert_almost_equal((in_ans_data - data).var(), 0, 4)


# need to move these into a test class
def test_fft_gaussian_filter():
    """Test the gaussian filter."""
    data = rng.standard_normal((128, 128, 128))
    sigmas = (rng.random(data.ndim) + 1) * 2
    fftg = fft_gaussian_filter(data, sigmas)
    # The fft_convolution is equivalent to wrapping around
    # and its inherently more accurate so we need to truncate
    # the kernel for the gaussian filter further out.
    fftc = gaussian_filter(data, sigmas, mode="wrap", truncate=32)
    assert_allclose(fftg, fftc, atol=1e-8, rtol=1e-6, err_msg="sigmas = {}".format(sigmas))


def _turn_slices_into_list(slice_list):
    """Take output of slice_maker and turn into list for testing."""
    result = []
    for s in slice_list:
        result += [s.start, s.stop]
    return np.array(result)


def test_slice_maker_negative():
    """Make sure slice_maker doesn't return negative indices."""
    slices = _turn_slices_into_list(slice_maker((10, -10), 10))
    assert (slices >= 0).all(), slices


def test_slice_maker_complex_input():
    """test complex in all positions."""
    for y0, x0, width in product(*(((10, 10j),) * 3)):
        if np.isrealobj((y0, x0, width)):
            continue
        with pytest.raises(TypeError):
            slice_maker((y0, x0), width)


def test_slice_maker_float_input():
    """Make sure floats are rounded properly."""
    for i in range(10):
        y0, x0, width = rng.random(3) * 100
        slice_list = _turn_slices_into_list(slice_maker((y0, x0), width))
        assert np.issubdtype(slice_list.dtype, int)


def test_slice_maker_center():
    """Make sure slices center y0, x0 at fft center."""
    for i in range(10):
        data = np.zeros((256, 256))
        center_loc = tuple(rng.integers(64, 256 - 64, 2))
        data[center_loc] = 1
        width = rng.integers(16, 32)
        slices = slice_maker(center_loc, width)
        data_crop = data[slices]
        print(data_crop)
        print(ifftshift(data_crop))
        assert ifftshift(data_crop)[0, 0] == 1, ifftshift(data_crop)


def test_padding_slices():
    """Make sure we can reverse things."""
    oldshape = tuple(rng.integers(64, 256 - 64, 2))
    newshape = tuple(rng.integers(s, s * 2) for s in oldshape)
    data = rng.standard_normal(oldshape)
    new_data = fft_pad(data, newshape)
    padding, slices = padding_slices(newshape, oldshape)
    assert np.all(data == new_data[slices])
