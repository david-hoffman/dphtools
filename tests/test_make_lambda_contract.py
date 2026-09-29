"""Owner-approved public Jacobian column scaling, independent of the solver."""

import inspect
import math
from decimal import Decimal
from fractions import Fraction
from numbers import Real

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dphtools.utils.lm import make_lambda


def _make_unchanged(jacobian, previous):
    original_j = jacobian.copy()
    original_previous = np.asarray(previous).copy()
    try:
        return make_lambda(jacobian, previous)
    finally:
        assert jacobian.dtype == original_j.dtype
        assert_array_equal(jacobian, original_j)
        assert np.asarray(previous).dtype == original_previous.dtype
        assert_array_equal(np.asarray(previous), original_previous)


def _scalar_epsilon(value):
    # Scalar precision survives object storage; container dtype need not.
    epsilon = np.finfo(np.float64).eps
    if isinstance(value, np.floating):
        epsilon = max(epsilon, float(np.finfo(value.dtype).eps))
    return epsilon


def _assert_scaling(result, expected):
    matrix = np.asarray(result)
    expected_shape = (len(expected), len(expected))
    assert matrix.shape == expected_shape, f"expected shape {expected_shape}, got {matrix.shape}"
    assert np.isrealobj(matrix), f"expected a real matrix, got dtype {matrix.dtype}"
    # At most three terms per norm: eight eps allows input/reference rounding,
    # products, summation and square root. Respect the returned float precision;
    # the binary64 floor also allows comparison via Python float and math.sqrt.
    for row in range(len(expected)):
        for column in range(len(expected)):
            value = matrix[row, column]
            assert isinstance(
                value, (Real, Decimal)
            ), f"entry [{row}, {column}] must be real numeric, got {type(value).__name__}"
            assert np.isrealobj(value), f"entry [{row}, {column}] must be real, got {value}"
            if row != column or expected[column] == 0:
                # Structural zeros are exact in the original representation.
                assert value == 0, f"entry [{row}, {column}] must be zero, got {value}"
            else:
                try:
                    if isinstance(value, (int, np.integer)):
                        exact = Fraction(int(value), 1)
                    else:
                        exact = Fraction(*value.as_integer_ratio())
                    # Normalize before float conversion; neither wide nor tiny
                    # finite magnitudes need to fit in binary64 themselves.
                    ratio = float(exact / Fraction(expected[column]))
                except (AttributeError, OverflowError, TypeError, ValueError) as exc:
                    raise AssertionError(
                        f"entry [{row}, {column}] must be finite real numeric: {value}"
                    ) from exc
                assert math.isfinite(ratio), f"nonfinite normalized entry [{row}, {column}]"
                # No absolute tolerance: losing a tiny nonzero norm must fail.
                epsilon = _scalar_epsilon(value)
                assert math.isclose(ratio, 1.0, rel_tol=8 * epsilon, abs_tol=0), (
                    f"diagonal [{column}] expected {expected[column]}, got {value}; "
                    f"relative tolerance {8 * epsilon}"
                )


def test_make_lambda_preserves_public_signature():
    parameters = inspect.signature(make_lambda).parameters
    assert tuple(parameters) == ("j", "d0")
    for parameter in parameters.values():
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.parametrize(
    "jacobian, previous, expected",
    [
        (np.array([[3, 0, 0], [0, 4, 12]]), np.array([5, 2, 6]), [5, 4, 12]),
        (np.array([[3, 0], [4, 0], [0, 12]], dtype=np.float32), 2, [5, 12]),
        (np.array([[0.0, 3, 0], [0, 4, 0]]), np.array([0.0, 0, 7]), [0, 5, 7]),
    ],
    ids=["wide-independent-previous", "tall-scalar", "zero-column-and-zero-scale"],
)
def test_rectangular_column_norms_and_independent_previous_scales(jacobian, previous, expected):
    _assert_scaling(_make_unchanged(jacobian, previous), expected)


@pytest.mark.parametrize("previous", [0, 2, 2.5, np.float32(0.5), np.array(1.0)])
def test_scalar_and_repeated_vector_give_the_same_correct_scaling(previous):
    jacobian = np.array([[3.0, 0, 1], [4, 12, 1]])
    expected = [max(float(previous), norm) for norm in (5, 12, math.sqrt(2))]
    scalar_result = _make_unchanged(jacobian, previous)
    vector_result = _make_unchanged(jacobian, np.full(3, previous))
    # Both must match an independent oracle; agreement alone would be inadequate.
    _assert_scaling(scalar_result, expected)
    _assert_scaling(vector_result, expected)


def test_row_permutation_signs_and_positive_scaling_preserve_the_norm_rule():
    jacobian = np.array([[3.0, 0], [4, 0], [0, 12]])
    previous = np.array([6.0, 2])
    transformed = jacobian[[2, 0, 1]] * np.array([[-1], [1], [-1]])
    _assert_scaling(_make_unchanged(jacobian, previous), [6, 12])
    _assert_scaling(_make_unchanged(transformed, previous), [6, 12])
    _assert_scaling(_make_unchanged(jacobian / 8, previous / 8), [0.75, 1.5])


@pytest.mark.parametrize(
    "jacobian, expected",
    [
        (np.array([[3e200, 0], [4e200, 0]]), [5e200, 0]),
        (np.array([[3e-200, 0], [4e-200, 0]]), [5e-200, 0]),
        (np.array([[3_000_000_000, 0], [4_000_000_000, 0]], dtype=np.int64), [5e9, 0]),
    ],
    ids=["large-representable-norm", "tiny-nonzero-norm", "integer-square-range"],
)
def test_representable_norms_survive_intermediate_arithmetic_range(jacobian, expected):
    # Scaled 3-4-5 triangles: reference error from stored float inputs < 1e-15.
    _assert_scaling(_make_unchanged(jacobian, 0), expected)


@pytest.mark.parametrize(
    "jacobian",
    [
        np.array(3.0),
        np.array([3.0, 4]),
        np.ones((1, 1, 1)),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[True, False]]),
        np.array([[3 + 0j, 4 + 0j]]),
        np.array([[3 + 1j, 4 + 0j]]),
        np.array([["3", "4"]]),
        np.array([[3, 4]], dtype=object),
        np.array([[np.nan, 4]]),
        np.array([[np.inf, 4]]),
        np.array([[-np.inf, 4]]),
    ],
    ids=[
        "scalar",
        "vector",
        "three-dimensional",
        "zero-rows",
        "zero-columns",
        "boolean",
        "complex-real-values",
        "complex",
        "strings",
        "object",
        "nan",
        "positive-inf",
        "negative-inf",
    ],
)
def test_invalid_jacobian_raises_value_error_and_preserves_inputs(jacobian):
    with pytest.raises(ValueError):
        _make_unchanged(jacobian, 0)


@pytest.mark.parametrize(
    "previous",
    [
        np.array([[1.0, 2]]),
        np.array([]),
        np.array([1.0]),
        np.array([1.0, 2, 3]),
        -1,
        np.nan,
        np.inf,
        -np.inf,
        np.array([1.0, -1]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        True,
        np.array([True, False]),
        1 + 0j,
        1 + 1j,
        np.array([1 + 0j, 2 + 0j]),
        "1",
        np.array(["1", "2"]),
        np.array([1, "bad"], dtype=object),
    ],
    ids=[
        "matrix",
        "empty-vector",
        "short-vector",
        "long-vector",
        "negative-scalar",
        "nan-scalar",
        "positive-inf-scalar",
        "negative-inf-scalar",
        "negative-vector",
        "nan-vector",
        "inf-vector",
        "boolean-scalar",
        "boolean-vector",
        "complex-real-scalar",
        "complex-scalar",
        "complex-vector",
        "string-scalar",
        "string-vector",
        "object-vector",
    ],
)
def test_invalid_previous_scale_raises_value_error_and_preserves_inputs(previous):
    with pytest.raises(ValueError):
        _make_unchanged(np.array([[3.0, 0], [4, 12]]), previous)


def _assert_wide_norm(result, scale):
    matrix = np.asarray(result)
    # One column requires exactly one diagonal entry and has no off-diagonals.
    assert matrix.shape == (1, 1), f"expected shape (1, 1), got {matrix.shape}"
    value = matrix[0, 0]
    assert not isinstance(value, (bool, np.bool_)), "expected a real numeric norm"
    try:
        if isinstance(value, (int, np.integer)):
            numerator, denominator = int(value), 1
        else:
            # Floating, Decimal and Fraction scalars expose their exact ratio.
            # Nonfinite and nonreal scalars cannot supply a finite integer ratio.
            numerator, denominator = value.as_integer_ratio()
        normalized = float(Fraction(numerator, denominator) / Fraction(*scale.as_integer_ratio()))
    except (AttributeError, OverflowError, TypeError, ValueError) as exc:
        raise AssertionError(
            "expected a finite real norm with a bounded normalized value"
        ) from exc
    epsilon = _scalar_epsilon(value)
    # Only the normalized ratio is converted to float; wide magnitudes survive.
    assert math.isclose(
        normalized, math.sqrt(2), rel_tol=8 * epsilon, abs_tol=0
    ), f"norm / M expected sqrt(2), got {normalized}; relative tolerance {8 * epsilon}"


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_beyond_float64_norm_succeeds_widely_or_reports_runtime_error(policy, record_property):
    jacobian = np.full((2, 1), 1.4e308, dtype=np.float64)
    original = jacobian.copy()
    previous_policy = np.geterr()
    try:
        with np.errstate(over=policy, invalid=policy):
            try:
                result = make_lambda(jacobian, 0)
            except RuntimeError as exc:
                # Numerical inability is allowed only around this extreme call.
                record_property("boundary_outcome", "RuntimeError")
                record_property("boundary_diagnostic", str(exc))
            else:
                record_property("boundary_outcome", "returned_matrix")
                _assert_wide_norm(result, original[0, 0])
    finally:
        assert np.geterr() == previous_policy
        assert jacobian.dtype == original.dtype
        assert_array_equal(jacobian, original)
