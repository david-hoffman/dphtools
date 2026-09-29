"""Supplemental D1 contracts through public plotting APIs and real Agg draws."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyQuadMesh, QuadMesh
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from numpy.testing import assert_allclose, assert_array_equal

from dphtools import display


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def assert_image_pixels_and_limits(fig, expected):
    """Observe native images or cell-valued meshes, including mesh row/column layout."""
    observations = []
    for axis in fig.axes:
        observations.extend((artist, artist.get_array()) for artist in axis.images)
        for artist in axis.collections:
            if isinstance(artist, (QuadMesh, PolyQuadMesh)):
                coordinates = artist.get_coordinates()
                cell_shape = tuple(size - 1 for size in coordinates.shape[:2])
                values = artist.get_array()
                if np.isfinite(coordinates).all() and values.size == np.prod(cell_shape):
                    observations.append((artist, values.reshape(cell_shape)))
    for artist, values in observations:
        low, high = artist.get_clim()
        if (
            np.array_equal(values, expected)
            and not np.ma.getmaskarray(values).any()
            and np.isfinite([low, high]).all()
            and low < high
        ):
            return
    pytest.fail("No displayed image or mesh preserves the pixels with usable color limits.")


@pytest.mark.parametrize("color", [0, 0.375, np.float64(1)])
def test_scalar_colorline_draws_uniform_color_without_changing_segments(color):
    x = np.array([-2.0, 0.0, 3.0, 4.0])
    y = np.array([1.0, -1.0, 2.0, 0.0])
    original_x, original_y = x.copy(), y.copy()
    cmap = ListedColormap(["red", "green", "blue", "cyan"])
    fig, ax = plt.subplots()
    collection = display.colorline(x, y, z=color, cmap=cmap, norm=Normalize(0, 1), ax=ax)
    fig.canvas.draw()
    assert collection in ax.collections
    assert_allclose(
        collection.get_segments(),
        [[[-2, 1], [0, -1]], [[0, -1], [3, 2]], [[3, 2], [4, 0]]],
    )
    # A LineCollection may share one RGBA value or repeat it per segment.
    colors = collection.get_colors()
    expected = to_rgba({0: "red", 0.375: "green", 1: "cyan"}[float(color)])
    assert len(colors) in (1, 3)
    assert_allclose(colors, np.broadcast_to(expected, colors.shape))
    assert_array_equal(x, original_x)
    assert_array_equal(y, original_y)


@pytest.mark.parametrize("explicit_axis", [False, True])
def test_recolor_resolves_a_registered_name_and_preserves_line_data(explicit_axis):
    name = "coverage_continuation_constant_color"
    cmap = ListedColormap(["#2a7fbd"], name=name)
    matplotlib.colormaps.register(cmap)
    try:
        fig, (selected, other) = plt.subplots(1, 2)
        lines = [
            selected.plot([0, 2, 5], values, color="red")[0]
            for values in ([1, 4, -2], [5, 0, 3], [-1, -2, 7])
        ]
        other_line = other.plot([0, 1], [2, 3], color="green")[0]
        originals = [line.get_xydata().copy() for line in lines]
        plt.sca(other if explicit_axis else selected)
        display.recolor(name, **({"ax": selected} if explicit_axis else {}))
        fig.canvas.draw()
        for line, original in zip(lines, originals):
            assert_allclose(to_rgba(line.get_color()), to_rgba("#2a7fbd"))
            assert_array_equal(line.get_xydata(), original)
        assert_allclose(to_rgba(other_line.get_color()), to_rgba("green"))
    finally:
        matplotlib.colormaps.unregister(name)


@pytest.mark.parametrize("name", ["viridis", "magma"])
def test_recolor_builtin_names_produce_palette_colors_in_a_real_draw(name):
    fig, ax = plt.subplots()
    lines = [
        ax.plot([0, 1, 3], [index, index + 2, index - 1], color="red")[0] for index in range(3)
    ]
    original = [line.get_xydata().copy() for line in lines]
    display.recolor(name, ax=ax)
    fig.canvas.draw()
    # No sampling positions are specified. Require colors from the named
    # registered palette, rather than prescribing their spacing along it.
    cmap = matplotlib.colormaps[name]
    palette = cmap(np.linspace(0, 1, cmap.N))
    for line, points in zip(lines, original):
        rgba = to_rgba(line.get_color())
        assert np.any(np.all(np.isclose(palette, rgba, rtol=0, atol=1e-12), axis=1))
        assert_array_equal(line.get_xydata(), points)


@pytest.mark.parametrize("gamma", [0.5, 2.0])
def test_power_norm_masked_matrix_preserves_mask_values_and_inverse(gamma):
    values = np.ma.array(
        [[2.0, 4.0, 10.0], [3.0, 8.0, 6.0]], mask=[[False, True, False], [False, False, True]]
    )
    original_data, original_mask = values.data.copy(), values.mask.copy()
    norm = display.SymPowerNorm(gamma, vmin=2, vmax=10)
    scaled = norm(values)
    expected = ((original_data - 2) / 8) ** gamma
    assert scaled.shape == values.shape
    assert_array_equal(np.ma.getmaskarray(scaled), original_mask)
    assert_allclose(np.ma.getdata(scaled)[~original_mask], expected[~original_mask], atol=1e-12)
    restored = norm.inverse(scaled)
    assert_array_equal(np.ma.getmaskarray(restored), original_mask)
    assert_allclose(
        np.ma.getdata(restored)[~original_mask], original_data[~original_mask], atol=1e-12
    )
    assert_array_equal(values.data, original_data)
    assert_array_equal(values.mask, original_mask)


@pytest.mark.parametrize("gamma", [0.5, 2.0])
def test_power_norm_inverse_independently_preserves_a_masked_input(gamma):
    scaled = np.ma.array([0.0, 0.25, 0.5, 1.0], mask=[False, True, False, False])
    original = scaled.copy()
    result = display.SymPowerNorm(gamma, vmin=-3, vmax=5).inverse(scaled)
    expected = -3 + 8 * scaled.data ** (1 / gamma)
    assert_array_equal(np.ma.getmaskarray(result), scaled.mask)
    assert_allclose(np.ma.getdata(result)[~scaled.mask], expected[~scaled.mask], atol=1e-12)
    assert_array_equal(scaled.data, original.data)
    assert_array_equal(scaled.mask, original.mask)


@pytest.mark.parametrize("gamma", [0.5, 2.0])
@pytest.mark.parametrize("call_clip", [False, True], ids=["constructor-clip", "call-clip"])
def test_power_norm_clipping_preserves_mask_and_input(gamma, call_clip):
    values = np.ma.array(
        [-8.0, 2.0, 4.0, 10.0, 18.0, -100.0], mask=[False, False, False, False, False, True]
    )
    original = values.copy()
    norm = display.SymPowerNorm(gamma, vmin=2, vmax=10, clip=not call_clip)
    result = norm(values, **({"clip": True} if call_clip else {}))
    assert_array_equal(np.ma.getmaskarray(result), original.mask)
    assert_allclose(np.ma.getdata(result)[:5], np.array([0, 0, 0.25, 1, 1]) ** gamma)
    assert_array_equal(values.data, original.data)
    assert_array_equal(values.mask, original.mask)


@pytest.mark.parametrize("bounds", [{}, {"vmin": 2}, {"vmax": 10}])
def test_power_norm_autoscaling_ignores_masked_outliers(bounds):
    values = np.ma.array([-1000.0, 2.0, 4.0, 10.0, 1000.0], mask=[True, False, False, False, True])
    original = values.copy()
    norm = display.SymPowerNorm(2, **bounds)
    result = norm(values)
    assert (norm.vmin, norm.vmax) == (2, 10)
    assert_array_equal(np.ma.getmaskarray(result), original.mask)
    assert_allclose(np.ma.getdata(result)[1:4], [0, 1 / 16, 1])
    assert_array_equal(values.data, original.data)
    assert_array_equal(values.mask, original.mask)


@pytest.mark.parametrize(
    "value",
    [3.0, [-2.0, 3.0, 8.0], np.ma.array([1.0, 7.0], mask=[False, True])],
    ids=["scalar", "vector", "masked"],
)
def test_power_norm_equal_finite_bounds_map_unmasked_values_to_zero(value):
    original = np.ma.array(value, copy=True)
    result = display.SymPowerNorm(0.5, vmin=3, vmax=3)(value)
    assert np.shape(result) == np.shape(value)
    assert_array_equal(np.ma.getmaskarray(result), np.ma.getmaskarray(original))
    assert_allclose(np.ma.asarray(result).compressed(), 0, atol=0)
    assert_array_equal(np.ma.getdata(value), original.data)
    assert_array_equal(np.ma.getmaskarray(value), np.ma.getmaskarray(original))


def test_power_norm_rejects_reversed_bounds():
    with pytest.raises(ValueError):
        display.SymPowerNorm(2, vmin=5, vmax=1)([2.0, 3.0])


@pytest.mark.parametrize("bounds", [{}, {"vmin": 1}, {"vmax": 5}])
def test_power_norm_inverse_rejects_missing_calibration(bounds):
    with pytest.raises(ValueError):
        display.SymPowerNorm(2, **bounds).inverse([0.0, 0.5, 1.0])


@pytest.mark.parametrize("gamma, masked", [(0.5, False), (2.0, True)])
def test_power_norm_image_and_colorbar_draw_with_independent_pixel_colors(gamma, masked):
    pixels = np.ma.array([[2.0, 4.0, 6.0], [8.0, 9.0, 10.0]], mask=False)
    if masked:
        pixels.mask[0, 1] = True
    original = pixels.copy()
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad=(1, 0, 0, 0))
    norm = display.SymPowerNorm(gamma, vmin=2, vmax=10)
    fig, ax = plt.subplots()
    artist = ax.imshow(pixels, norm=norm, cmap=cmap)
    colorbar = fig.colorbar(artist, ax=ax)
    fig.canvas.draw()
    expected_rgba = cmap(np.ma.array(((pixels.data - 2) / 8) ** gamma, mask=pixels.mask))
    assert_allclose(artist.to_rgba(pixels), expected_rgba, atol=1e-12)
    assert_array_equal(artist.get_array().data, original.data)
    assert_array_equal(np.ma.getmaskarray(artist.get_array()), original.mask)
    assert np.isfinite(colorbar.ax.get_ylim()).all()
    assert_allclose(colorbar.ax.get_ylim(), [2, 10], atol=1e-12)
    assert_array_equal(pixels.data, original.data)
    assert_array_equal(pixels.mask, original.mask)


@pytest.mark.parametrize(
    "data", [[np.ones((2, 3))], np.ones((2, 3)), 7], ids=["list", "array", "scalar"]
)
def test_display_grid_rejects_non_dictionary_input(data):
    # D1 requires ordinary validation, without selecting exact error text.
    with pytest.raises((TypeError, ValueError)):
        display.display_grid(data)


@pytest.mark.parametrize("shape", [(), (7,), (2, 3, 4, 5)])
def test_mip_rejects_data_outside_two_or_three_dimensions(shape):
    with pytest.raises((TypeError, ValueError)):
        display.mip(np.zeros(shape))


@pytest.mark.parametrize("entry", ["make_grid", "display_grid"])
def test_empty_grids_cannot_claim_populated_results(entry, record_property):
    # Both rejection and an empty usable layout satisfy the deliberately narrow
    # contract. Retain the rejection diagnostic for review instead of hiding it.
    try:
        result = display.make_grid(0) if entry == "make_grid" else display.display_grid({})
    except Exception as error:
        record_property("empty_grid_rejection", f"{type(error).__name__}: {error}")
    else:
        record_property("empty_grid_return_type", type(result).__name__)
        for number in plt.get_fignums():
            fig = plt.figure(number)
            assert not any(ax.has_data() for ax in fig.axes)
            fig.canvas.draw()


@pytest.mark.parametrize("dtype", [np.float64, np.uint16])
def test_auto_adjust_distributed_background_and_bright_features_preserves_pixels(dtype):
    rng = np.random.default_rng(1827)
    row, column = np.mgrid[:48, :64]
    pixels = 100 + rng.normal(0, 8, (48, 64))
    pixels += 180 * np.exp(-((row - 13) ** 2 + (column - 19) ** 2) / 18)
    pixels += 350 * np.exp(-((row - 32) ** 2 + (column - 47) ** 2) / 12)
    pixels = pixels.astype(dtype)
    original = pixels.copy()
    limits = display.auto_adjust(pixels)
    assert isinstance(limits, dict)
    low, high = limits["vmin"], limits["vmax"]
    assert np.isfinite([low, high]).all() and low < high
    fig, ax = plt.subplots()
    artist = ax.imshow(pixels, **limits)
    fig.canvas.draw()
    assert_array_equal(artist.get_array(), original)
    display.display_grid({"distributed background": pixels}, auto=True)
    grid = plt.gcf()
    grid.canvas.draw()
    assert_image_pixels_and_limits(grid, original)
    assert_array_equal(pixels, original)
