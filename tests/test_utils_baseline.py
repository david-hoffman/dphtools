"""SETUP-001 L1/L3/L5: supplemental public array and geometry baselines.

Expectations come from the approved API packet and elementary arithmetic.
These tests were added after the implementation; they are not test-first evidence.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dphtools import utils


@pytest.mark.parametrize("operation, divisor", [("sum", 1), ("mean", 4)])
@pytest.mark.parametrize("shape_args", [{"new_shape": (2, 2)}, {"bin_size": 2}])
def test_bin_values(operation, divisor, shape_args):
    data = np.arange(16).reshape(4, 4)
    expected = np.array([[10, 18], [42, 50]]) / divisor
    assert_allclose(utils.bin_ndarray(data, operation=operation, **shape_args), expected)


def test_bin_anisotropic_volume():
    data = np.arange(24).reshape(2, 3, 4)
    expected = np.array([[[6], [22], [38]], [[54], [70], [86]]])
    assert_array_equal(utils.bin_ndarray(data, bin_size=(1, 1, 4)), expected)


def test_bin_unit_size_preserves_data():
    data = np.array([2, -1, 4, 9])
    assert_array_equal(utils.bin_ndarray(data, bin_size=1), data)


def test_bin_rejects_nondivisible_target():
    with pytest.raises(ValueError):
        utils.bin_ndarray(np.arange(12).reshape(3, 4), new_shape=(2, 2))


def test_scale_nan_and_affine_values():
    data = np.array([[-2.0, 0.0], [6.0, np.nan]])
    assert_allclose(utils.scale(data), [[0.0, 0.25], [1.0, np.nan]], equal_nan=True)


@pytest.mark.parametrize("dtype, maximum", [(np.uint8, 255), (np.uint16, 65535)])
def test_scale_unsigned_full_range(dtype, maximum):
    scaled = utils.scale(np.array([-4.0, 0.0, 8.0]), dtype=dtype)
    assert scaled.dtype == dtype
    assert_array_equal(scaled, [0, maximum // 3, maximum])


@pytest.mark.parametrize("value", [3.0, 2.0 + 4.0j])
def test_radial_profile_constant_with_explicit_center(value):
    mean, std = utils.radial_profile(np.full((5, 5), value), center=(2, 2))
    assert mean.ndim == 1
    assert std.shape == mean.shape
    assert mean.size > 0
    assert_allclose(mean, value)
    assert_allclose(std, 0)


@pytest.mark.parametrize("data, expected", [([0], 0), ([1, 7, 7, 2, 7], 7), ([3, 0, 0], 0)])
def test_integer_mode(data, expected):
    assert utils.mode(np.array(data)) == expected


@pytest.mark.parametrize(
    "center, width, expected",
    [
        ((30, 20), 10, (slice(25, 35), slice(15, 25))),
        ((30, 20), 25, (slice(18, 43), slice(8, 33))),
        ((10, 20), (4, 6), (slice(8, 12), slice(17, 23))),
        ((0, 1), 4, (slice(0, 2), slice(0, 3))),
    ],
)
def test_slice_maker_documented_centers_and_clipping(center, width, expected):
    assert utils.slice_maker(center, width) == expected


@pytest.mark.parametrize("old_size, new_size", [(4, 7), (5, 8), (8, 5), (7, 4)])
def test_fft_pad_preserves_centered_coordinates(old_size, new_size):
    # Existing tests specify the FFT center. Here every retained sample is checked.
    old_center = (old_size + 1) // 2
    new_center = (new_size + 1) // 2
    data = np.arange(old_size) - old_center
    expected = np.full(new_size, -99)
    for output_index in range(new_size):
        source_index = output_index - new_center + old_center
        if 0 <= source_index < old_size:
            expected[output_index] = data[source_index]
    assert_array_equal(
        utils.fft_pad(data, new_size, mode="constant", constant_values=-99), expected
    )


def test_fft_convolution_centered_identity_kernel():
    data = np.arange(25.0).reshape(5, 5)
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1
    assert_allclose(utils.fftconvolve_fast(data, kernel), data, atol=1e-12)


def test_window_numeric_tensor_product():
    # Symmetric Hann(3) = [0, 1, 0], Hann(5) = [0, 1/2, 1, 1/2, 0].
    expected = np.zeros((3, 5))
    expected[1] = [0, 0.5, 1, 0.5, 0]
    assert_allclose(utils.win_nd((3, 5)), expected, atol=1e-15)


def test_window_forwards_window_arguments():
    # Periodic Hann(4) samples one complete period without its repeated endpoint.
    assert_allclose(utils.win_nd((4,), sym=False), [0, 0.5, 1, 0.5], atol=1e-15)


def test_anscombe_formula():
    data = np.array([0.0, 1.0, 4.0, 20.0])
    assert_allclose(utils.anscombe(data), 2 * np.sqrt(data + 3 / 8))


@pytest.mark.parametrize("sigma", [0, 1.5, (1.0, 2.0)])
def test_gaussian_filter_constant_and_shape(sigma):
    data = np.full((8, 10), 3.25)
    result = utils.fft_gaussian_filter(data, sigma)
    assert result.shape == data.shape
    assert_allclose(result, data, atol=1e-12)


def test_gaussian_filter_zero_sigma_preserves_nonconstant_data():
    data = np.array([[0.0, 1.0, 3.0, -2.0], [4.0, -1.0, 2.0, 0.0]])
    assert_allclose(utils.fft_gaussian_filter(data, 0), data, atol=1e-12)


def test_gaussian_filter_fourier_mode_at_known_frequency():
    # A normalized Gaussian attenuates cos(2*pi*f*x) by exp(-2*pi**2*sigma**2*f**2).
    x = np.arange(32)
    frequency = 2 / 32
    sigma = 2.0
    wave = np.cos(2 * np.pi * frequency * x)
    expected = wave * np.exp(-2 * np.pi**2 * sigma**2 * frequency**2)
    assert_allclose(utils.fft_gaussian_filter(wave, sigma), expected, atol=1e-10)


@pytest.mark.parametrize("number, factors", [(2, [2]), (10, [2, 5]), (72, [2, 2, 2, 3, 3])])
def test_prime_factors(number, factors):
    assert_array_equal(utils.find_prime_facs(number), factors)


def test_get_max_returns_x_at_y_maximum():
    assert utils.get_max(np.array([2.0, 5.0, 9.0]), np.array([1.0, 7.0, 3.0])) == 5


def test_get_max_along_columns():
    x = np.array([[2.0, 2.0], [5.0, 5.0], [9.0, 9.0]])
    y = np.array([[8.0, 0.0], [1.0, 3.0], [4.0, 7.0]])
    assert_array_equal(utils.get_max(x, y, axis=0), [2.0, 9.0])


def test_plane_fit_zero_surface_has_zero_coefficients():
    x, y = np.meshgrid(np.arange(-2.0, 3.0), np.arange(-1.0, 3.0))
    z = np.zeros_like(x)
    coefficients = utils.plane_fit(x.ravel(), y.ravel(), z.ravel())
    assert_allclose(coefficients, [0, 0, 0], atol=1e-12)


def test_remove_tilt_leaves_a_level_surface():
    x, y = np.meshgrid(np.arange(-2.0, 3.0), np.arange(-1.0, 3.0))
    level = utils.remove_tilt(x.ravel(), y.ravel(), (2 * x - 3 * y + 4).ravel())
    # The packet does not define the residual height datum; only tilt is asserted.
    assert np.ptp(level) < 1e-12


def test_plane_normal_is_perpendicular_to_two_tangents():
    x, y = np.meshgrid(np.arange(-2.0, 3.0), np.arange(-1.0, 3.0))
    normal = utils.find_normal(x.ravel(), y.ravel(), (2 * x - 3 * y + 4).ravel())
    assert np.linalg.norm(normal) > 0
    assert_allclose([np.dot(normal, [1, 0, 2]), np.dot(normal, [0, 1, -3])], 0, atol=1e-12)


def test_rotation_maps_source_to_target_and_preserves_lengths():
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    matrix = utils.rot_matrix(source, target)
    assert_allclose(matrix @ source, target, atol=1e-12)
    assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert_allclose(np.linalg.det(matrix), 1, atol=1e-12)


def test_quadratic_fit_uses_documented_coefficient_order():
    x, y = np.meshgrid(np.arange(-2.0, 3.0), np.arange(-2.0, 3.0))
    z = 2 * x**2 + 3 * y**2 + 4 * x + 5 * y + x * y + 6
    coefficients, _ = utils.fit_quadratic(x.ravel(), y.ravel(), z.ravel())
    assert_allclose(coefficients, [2, 3, 4, 5, 1, 6], atol=1e-12)


def test_quadratic_center_from_sampled_surface():
    x, y = np.meshgrid(np.arange(-2.0, 3.0), np.arange(-2.0, 3.0))
    z = (x - 0.5) ** 2 + 2 * (y + 0.25) ** 2
    center, _ = utils.find_center(x.ravel(), y.ravel(), z.ravel())
    assert_allclose(center, [0.5, -0.25], atol=1e-12)


def test_quadratic_center_with_cross_term_and_exact_coefficients():
    # Gradient [4*x + y - 2, x + 6*y + 11] vanishes at (1, -2).
    center, errors = utils.find_center_quad_coefs(
        np.array([2.0, 3.0, -2.0, 11.0, 1.0, 9.0]), np.zeros(6)
    )
    assert_allclose(center, [1, -2], atol=1e-12)
    assert_allclose(errors, 0, atol=1e-12)


def test_extended_depth_of_focus_preserves_identical_planes():
    image = np.array([[0.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 4.0]])
    assert_array_equal(utils.edf(np.stack([image, image, image])), image)


def test_split_tiles_preserve_values_without_assuming_tile_order():
    data = np.arange(48).reshape(6, 8)
    tiles = utils.split_img(data, (2, 4))
    expected = [data[y : y + 2, x : x + 4] for y in (0, 2, 4) for x in (0, 4)]
    assert tiles.shape == (6, 2, 4)
    assert sorted(tuple(tile.ravel()) for tile in tiles) == sorted(
        tuple(tile.ravel()) for tile in expected
    )


def test_crop_for_split_retains_pixels_and_divisible_shape():
    data = np.arange(63).reshape(7, 9)
    cropped = utils.crop_image_for_split(data, (2, 4))
    assert cropped.shape == (6, 8)
    assert any(
        np.array_equal(cropped, data[y : y + 6, x : x + 8]) for y in range(2) for x in range(2)
    )
    tiles = utils.split_img(cropped, (2, 4))
    assert_array_equal(np.sort(tiles.ravel()), np.sort(cropped.ravel()))


def test_square_tiles_split_combine_round_trip():
    data = np.arange(64).reshape(8, 8)
    assert_array_equal(utils.combine_img(utils.split_img(data, (4, 4))), data)


def test_one_has_no_prime_factors():
    assert utils.find_prime_facs(1).size == 0


@pytest.mark.parametrize("shape, tile_shape", [((6, 10), (3, 5)), ((6, 12), (2, 4))])
def test_rectangular_tiles_in_square_grid_split_combine_round_trip(shape, tile_shape):
    # Respect U1's square tile grid: 2x2 or 3x3, with rectangular individual tiles.
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape) - 20
    combined = utils.combine_img(utils.split_img(data, tile_shape))
    assert_array_equal(combined, data)


def test_quadratic_fit_and_center_recover_a_rotated_surface():
    x, y = np.meshgrid(np.arange(-3.0, 4.0), np.arange(-3.0, 4.0))
    u, v = x - 0.75, y + 0.5
    z = 2 * u**2 + 3 * v**2 + u * v + 7
    # The Hessian [[4,1],[1,6]] is positive definite, so (0.75,-0.5)
    # is the unique minimum. Fit and center are both real public entry points.
    center, _ = utils.find_center(x.ravel(), y.ravel(), z.ravel())
    assert_allclose(center, [0.75, -0.5], atol=1e-12)


def test_gaussian_filter_anisotropic_fourier_mode():
    y, x = np.meshgrid(np.arange(16), np.arange(20), indexing="ij")
    fy, fx = 2 / 16, 3 / 20
    sy, sx = 0.75, 1.5
    wave = np.cos(2 * np.pi * (fy * y + fx * x))
    attenuation = np.exp(-2 * np.pi**2 * ((sy * fy) ** 2 + (sx * fx) ** 2))
    assert_allclose(utils.fft_gaussian_filter(wave, (sy, sx)), wave * attenuation, atol=1e-12)
