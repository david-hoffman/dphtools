"""Five owner-approved display policies, observed through public artist APIs."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from numpy.testing import assert_allclose, assert_array_equal

from dphtools import display


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def assert_unlabelled_bar_length(ax, previous_artists, expected_length):
    """Require a calibrated added path to contribute to the actual Agg draw."""
    ax.figure.canvas.draw()
    painted = np.asarray(ax.figure.canvas.buffer_rgba()).copy()
    added = [artist for artist in ax.findobj() if artist not in previous_artists]
    assert not [
        artist.get_text() for artist in added if isinstance(artist, Text) and artist.get_text()
    ]
    lengths = []
    matched_paint = False
    for artist in added:
        if isinstance(artist, (Patch, Line2D)) and artist.get_visible():
            bounds = artist.get_path().get_extents(artist.get_transform())
            length = bounds.transformed(ax.transData.inverted()).width
            lengths.append(length)
            if np.isclose(length, expected_length, rtol=0, atol=1e-9):
                # Toggling only this matched path handles separate/embedded
                # alpha and nested artists without prescribing their storage.
                artist.set_visible(False)
                try:
                    ax.figure.canvas.draw()
                    contributes_paint = np.any(
                        np.asarray(ax.figure.canvas.buffer_rgba()) != painted
                    )
                finally:
                    artist.set_visible(True)
                    ax.figure.canvas.draw()
                if contributes_paint:
                    matched_paint = True
                    break
    assert (
        matched_paint
    ), f"No added painted bar has data-coordinate length {expected_length}; found {lengths}"


@pytest.mark.parametrize(
    "data",
    [
        {"single sample": np.array([3.25])},
        {"signal": np.array([4.0, -2.0, 7.5, 1.0]), "offset": np.array([8, 3, -1, 5, 2])},
    ],
    ids=["single-sample", "different-length-series"],
)
def test_display_grid_plots_one_dimensional_values_against_sample_indices(data):
    original = {key: values.copy() for key, values in data.items()}
    display.display_grid(data)
    fig = plt.gcf()
    fig.canvas.draw()
    axes = [ax for ax in fig.axes if ax.lines]
    assert len(axes) == len(original)
    assert {ax.get_title() for ax in axes} == set(original)
    for ax in axes:
        assert len(ax.lines) == 1
        expected = original[ax.get_title()]
        assert_array_equal(ax.lines[0].get_xdata(), np.arange(expected.size))
        assert_array_equal(ax.lines[0].get_ydata(), expected)
    assert set(data) == set(original)
    for key in original:
        assert_array_equal(data[key], original[key])


@pytest.mark.parametrize("count", [1, 4])
def test_make_grid_rejects_explicit_zero_rows(count):
    with pytest.raises(ValueError):
        display.make_grid(count, nrows=0)


def test_make_grid_positive_rows_remain_usable():
    fig, axes = display.make_grid(4, nrows=2)
    axes = np.asarray(axes).ravel()
    assert len(axes) >= 4
    for index, ax in enumerate(axes[:4]):
        assert ax.figure is fig
        (line,) = ax.plot([0, 1, 2], [index, index + 2, index - 1])
        assert_array_equal(line.get_xydata(), [[0, index], [1, index + 2], [2, index - 1]])
    fig.canvas.draw()


@pytest.mark.parametrize("count, expected_alpha", [(1, 1), (4, 8 / 9), (9, 3 / 4)])
def test_recolor_best_uses_selected_count_and_requested_palette(count, expected_alpha):
    # A constant palette fixes expected RGB without imposing sample positions.
    cmap = ListedColormap(["#2679b5"])
    fig, (ax, other) = plt.subplots(1, 2)
    lines = [
        ax.plot([0, 2, 5], [index, index + 3, index - 2], color="red", alpha=0.25)[0]
        for index in range(count)
    ]
    originals = [line.get_xydata().copy() for line in lines]
    (other_line,) = other.plot([0, 1], [7, 9], color="green", alpha=0.4)
    plt.sca(other)
    display.recolor(cmap, ax=ax, new_alpha="best")
    fig.canvas.draw()
    for line, original in zip(lines, originals):
        assert_allclose(
            to_rgba(line.get_color(), alpha=line.get_alpha()),
            (*to_rgba("#2679b5")[:3], expected_alpha),
            rtol=0,
            atol=1e-12,
        )
        assert_array_equal(line.get_xydata(), original)
    assert_allclose(to_rgba(other_line.get_color(), other_line.get_alpha()), to_rgba("green", 0.4))
    assert_array_equal(other_line.get_xydata(), [[0, 7], [1, 9]])


def test_recolor_best_on_empty_axes_is_a_noop():
    fig, ax = plt.subplots()
    display.recolor(ListedColormap(["blue"]), ax=ax, new_alpha="best")
    fig.canvas.draw()
    assert not ax.has_data()
    assert (
        not list(ax.lines)
        + list(ax.collections)
        + list(ax.images)
        + list(ax.patches)
        + list(ax.artists)
    )


@pytest.mark.parametrize("gamma", [0.8, 1.6])
@pytest.mark.parametrize("wavelength", [200.0, 379.0, 751.0, 1100.0])
def test_wavelength_outside_visible_interval_is_black(wavelength, gamma):
    color = np.asarray(display.wavelength_to_rgb(wavelength, gamma=gamma))
    assert color.shape == (3,)
    assert_array_equal(color, [0, 0, 0])


@pytest.mark.parametrize("gamma", [0.8, 1.6])
@pytest.mark.parametrize("wavelength", [380.0, 750.0])
def test_wavelength_interval_includes_both_endpoints(wavelength, gamma):
    # Preserve the existing finite, nonblack endpoint convention without
    # inventing a new in-range color table.
    color = np.asarray(display.wavelength_to_rgb(wavelength, gamma=gamma))
    assert color.shape == (3,)
    assert np.isfinite(color).all()
    assert (color >= 0).all()
    assert color.max() > 0


@pytest.mark.parametrize(
    "size, pixel_size, expected_length, location",
    [
        (2.0, 0.25, 8.0, "upper right"),
        (7.5, 1.5, 5.0, "lower left"),
        (1.2, 0.8, 1.5, "lower right"),
    ],
)
def test_scalebar_unit_none_draws_calibrated_bar_without_text(
    size, pixel_size, expected_length, location
):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(np.zeros((20, 50)), aspect="auto")
    fig.canvas.draw()
    previous = set(ax.findobj())
    display.add_scalebar(ax, scalebar_size=size, pixel_size=pixel_size, unit=None, loc=location)
    assert_unlabelled_bar_length(ax, previous, expected_length)
