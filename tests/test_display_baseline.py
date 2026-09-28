"""SETUP-001 L2/L3/L5: Agg-rendered plot data and documented geometry."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Rectangle
from numpy.testing import assert_allclose, assert_array_equal

from dphtools import display


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "x, y, expected",
    [
        ([0, 2, 5], [1, 3, -1], [[[0, 1], [2, 3]], [[2, 3], [5, -1]]]),
        ([0, 2], [1, 3], [[[0, 1], [2, 3]]]),
    ],
)
def test_make_segments_connects_adjacent_points(x, y, expected):
    assert_array_equal(display.make_segments(x, y), expected)


def test_make_segments_single_point_has_no_lines():
    assert display.make_segments([2], [3]).shape == (0, 2, 2)


@pytest.mark.parametrize("autoscale", [False, True])
def test_colorline_plots_coordinates_and_requested_colors(autoscale):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    display.colorline(
        [0, 2, 5], [1, 3, -1], z=[0.2, 0.8], linewidth=2, alpha=0.4, ax=ax, autoscale=autoscale
    )
    assert len(ax.collections) == 1
    collection = ax.collections[0]
    assert_allclose(collection.get_segments(), [[[0, 1], [2, 3]], [[2, 3], [5, -1]]])
    assert_allclose(collection.get_array(), [0.2, 0.8])
    assert_allclose(collection.get_linewidths(), [2])
    assert collection.get_alpha() == 0.4
    if not autoscale:
        assert_allclose(ax.get_xlim(), [-1, 1])
        assert_allclose(ax.get_ylim(), [-1, 1])
    fig.canvas.draw()


def test_colorline_default_axes_and_color_values_render():
    fig, ax = plt.subplots()
    display.colorline([0, 1, 2], [2, 0, 3])
    assert_allclose(ax.collections[0].get_segments(), [[[0, 2], [1, 0]], [[1, 0], [2, 3]]])
    fig.canvas.draw()


@pytest.mark.parametrize("count, rows", [(1, None), (5, 2), (4, 2)])
def test_make_grid_contains_requested_number_of_usable_axes(count, rows):
    fig, axes = display.make_grid(count, nrows=rows)
    axes = np.asarray(axes).ravel()
    assert len(axes) >= count
    assert all(ax.figure is fig for ax in axes)
    for index, ax in enumerate(axes[:count]):
        ax.plot([0, 1], [index, index + 1])
    display.clean_grid(fig, axes)
    assert len(fig.axes) == count
    fig.canvas.draw()


@pytest.mark.parametrize("contours", [False, True])
def test_display_grid_contains_each_named_image(contours):
    images = {"first": np.arange(20).reshape(4, 5), "second": np.arange(20, 40).reshape(4, 5)}
    display.display_grid(images, showcontour=contours, nrows=1, cmap="viridis")
    fig = plt.gcf()
    plotted = [
        (ax.get_title(), np.asarray(image.get_array())) for ax in fig.axes for image in ax.images
    ]
    assert len(plotted) == 2
    for title, pixels in plotted:
        assert title in images
        assert_array_equal(pixels, images[title])
    fig.canvas.draw()


def assert_image_planes(fig, expected):
    """Allow display transposes; do not invent undocumented axis orientation."""
    actual = [np.asarray(image.get_array()) for ax in fig.axes for image in ax.images]
    assert len(actual) == len(expected)
    remaining = list(actual)
    for plane in expected:
        match = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if np.array_equal(candidate, plane) or np.array_equal(candidate, plane.T)
            ),
            None,
        )
        assert match is not None
        remaining.pop(match)


@pytest.mark.parametrize("allaxes", [False, True])
def test_slice_plot_uses_requested_center(allaxes):
    data = np.arange(60).reshape(3, 4, 5)
    fig, _ = display.slice_plot(data, center=(1, 2, 3), allaxes=allaxes)
    assert_image_planes(fig, [data[1, :, :], data[:, 2, :], data[:, :, 3]])
    fig.canvas.draw()


@pytest.mark.parametrize("projection", [np.amax, np.mean])
@pytest.mark.parametrize("allaxes", [False, True])
def test_mip_shows_each_requested_projection(projection, allaxes):
    data = np.arange(60).reshape(3, 4, 5)
    fig, _ = display.mip(data, func=projection, allaxes=allaxes, cmap="viridis")
    assert_image_planes(fig, [projection(data, axis=axis) for axis in range(3)])
    fig.canvas.draw()


def test_mip_accepts_two_dimensional_image():
    data = np.arange(20).reshape(4, 5)
    fig, _ = display.mip(data, plt_kwds={"color": "red"})
    assert_image_planes(fig, [data])
    fig.canvas.draw()


def test_recolor_preserves_data_and_sets_requested_alpha():
    fig, ax = plt.subplots()
    (first,) = ax.plot([0, 1], [1, 2])
    (second,) = ax.plot([0, 1], [3, 4])
    display.recolor(ListedColormap(["red"]), ax=ax, new_alpha=0.4)
    for line, expected_y in [(first, [1, 2]), (second, [3, 4])]:
        # Alpha can be stored in the RGBA color or as a separate artist property.
        assert_allclose(to_rgba(line.get_color(), alpha=line.get_alpha()), [1, 0, 0, 0.4])
        assert_array_equal(line.get_xdata(), [0, 1])
        assert_array_equal(line.get_ydata(), expected_y)
    fig.canvas.draw()


@pytest.mark.parametrize("wavelength", [380, 440, 490, 510, 580, 645, 750])
def test_visible_wavelength_returns_finite_rgb(wavelength):
    # The packet promises an approximation, but specifies no numerical color table.
    color = np.asarray(display.wavelength_to_rgb(wavelength))
    assert color.shape == (3,)
    assert np.isfinite(color).all()
    assert (color >= 0).all()
    assert color.max() > 0


def test_scalebar_length_uses_pixel_size():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((20, 20)))
    display.add_scalebar(ax, scalebar_size=2, pixel_size=0.25, unit="µm")
    bars = [child for artist in ax.artists for child in artist.findobj(Rectangle)]
    assert any(np.isclose(bar.get_width(), 8) for bar in bars)
    fig.canvas.draw()


def test_power_normalization_positive_range_matches_documented_formula():
    norm = display.SymPowerNorm(2, vmin=0, vmax=4)
    assert_allclose(norm([0, 2, 4]), [0, 0.25, 1])


@pytest.mark.parametrize("gamma", [0.5, 1, 2])
def test_power_normalization_inverse_recovers_values(gamma):
    norm = display.SymPowerNorm(gamma, vmin=-2, vmax=4)
    values = np.array([-2.0, -1.0, 0.0, 1.0, 4.0])
    assert_allclose(norm.inverse(norm(values)), values, atol=1e-12)


def test_power_normalization_autoscale_and_clip():
    norm = display.SymPowerNorm(1, clip=True)
    norm.autoscale(np.array([-2.0, 4.0]))
    assert (norm.vmin, norm.vmax) == (-2, 4)
    assert_allclose(norm([-8, -2, 1, 4, 10]), [0, 0, 0.5, 1, 1])
    norm.autoscale_None(np.array([-10.0, 10.0]))
    assert (norm.vmin, norm.vmax) == (-2, 4)


def test_power_normalization_autoscales_only_missing_limit():
    norm = display.SymPowerNorm(1, vmin=-5)
    norm.autoscale_None(np.array([-2.0, 4.0]))
    assert (norm.vmin, norm.vmax) == (-5, 4)


def test_rectangle_is_centered_on_requested_coordinates():
    rectangle = display.make_rec(y=10, x=20, width=6, height=4, linewidth=2)
    assert_allclose(rectangle.get_xy(), [17, 8])
    assert rectangle.get_width() == 6
    assert rectangle.get_height() == 4
    assert rectangle.get_linewidth() == 2


def test_rectangle_from_slice_has_slice_dimensions():
    rectangle = display.make_rec_from_slice((slice(2, 6), slice(3, 9)), linewidth=2)
    assert rectangle.get_width() == 6
    assert rectangle.get_height() == 4


@pytest.mark.parametrize("log", [False, True])
def test_drift_plot_converts_sample_spacing_and_pixel_displacements(log):
    import pandas as pd

    frame = pd.DataFrame({"x0": [-2.0, -1.0, 0.0, 1.0, 2.0], "y0": [2.0, 1.0, 0.0, -1.0, -2.0]})
    fig, axes = display.drift_plot(frame, dt=0.5, dx=2.0, log=log)
    real_axis, fourier_axis, scatter_axis = axes
    assert len(real_axis.lines) == 2
    for line, coordinate in zip(real_axis.lines, ["x0", "y0"]):
        assert_allclose(np.diff(line.get_xdata()), 0.5)
        assert_allclose(line.get_ydata(), frame[coordinate].to_numpy() * 2, atol=1e-12)
    assert fourier_axis.figure is scatter_axis.figure is fig
    fig.canvas.draw()


@pytest.mark.parametrize("log", [False, True])
def test_histogram_and_cumulative_plot_have_valid_distribution_ranges(log):
    fig, ax = plt.subplots()
    display.hist_and_cumulative(np.array([1.0, 2.0, 2.0, 3.0]), ax=ax, log=log)
    assert len(fig.axes) == 2
    histogram_axis, cumulative_axis = fig.axes
    assert histogram_axis.patches
    assert all(np.isfinite(patch.get_path().vertices).all() for patch in histogram_axis.patches)
    assert len(cumulative_axis.lines) == 1
    line = cumulative_axis.lines[0]
    assert np.all(np.diff(line.get_xdata()) >= 0)
    assert np.all(np.diff(line.get_ydata()) >= 0)
    assert np.all((line.get_ydata() >= 0) & (line.get_ydata() <= 1))
    fig.canvas.draw()


@pytest.mark.parametrize("gamma, expected", [(0.5, [0, 0.5, 1]), (2, [0, 1 / 16, 1])])
def test_power_normalization_shifted_positive_range(gamma, expected):
    # The public formula says to map linearly into [0,1] before exponentiation.
    norm = display.SymPowerNorm(gamma, vmin=1, vmax=5)
    assert_allclose(norm([1, 2, 5]), expected)


@pytest.mark.parametrize("axis, expected_shape", [(0, (4, 5)), (1, (3, 5)), (2, (3, 4))])
def test_take_slice_default_preserves_constant_and_removes_selected_axis(axis, expected_shape):
    # G2: every possible plane has the same values, regardless of midpoint convention.
    data = np.full((3, 4, 5), -3.25)
    result = display.take_slice(data, axis=axis)
    assert result.shape == expected_shape
    assert_array_equal(result, np.full(expected_shape, -3.25))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_take_slice_explicit_coordinate_vector_selects_requested_plane(axis):
    data = np.arange(60).reshape(3, 4, 5)
    expected = (data[0, :, :], data[:, 1, :], data[:, :, 4])[axis]
    assert_array_equal(display.take_slice(data, axis=axis, midpoint=(0, 1, 4)), expected)
