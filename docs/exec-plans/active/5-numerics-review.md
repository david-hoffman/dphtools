# Issue 5 numerical review

## Accepted Invariants

- `divisors` is computed from image shape and tile side lengths.
- The product of `divisors` is the number of output tiles.
- `np.prod(divisors)` preserves the intended scalar product operation.
- Existing tests assert output shape and randomized cropped dimensions.

## Rejected Or Uncertain Claims

- No claim is made about broader NumPy 2 compatibility outside `split_img`.
- No claim is made about memory layout beyond the existing tests.

## Required Follow-Up Tests

- Add focused coverage for `display.take_slice` under NumPy 2 because it still references `np.int`.
- Add a compatibility issue for FFT `s`/`axes` deprecation warnings before they become errors.

## Confidence

High for the arithmetic change. The operation is dimensionless tile-count calculation, and no tolerance changes are involved.

## What Would Change This Conclusion

If downstream code relies on an exact scalar type from `np.product`, verify whether `np.prod` returns a materially different type for the same `divisors` input.
