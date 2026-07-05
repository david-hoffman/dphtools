# Issue 5 adversarial review

## Blocking Findings

None.

## Non-Blocking Findings

- `dphtools/display.py` still contains `np.int`, which is another NumPy 2 compatibility risk. It is outside issue 5's `split_img` scope and should be handled by a follow-up issue with focused tests.
- FFT calls in `dphtools/utils/__init__.py` emit NumPy 2 deprecation warnings about passing `s` without `axes`. That is outside this compatibility fix and should be tracked separately.

## Commands Run

```bash
python -m pytest -q tests/test_utils.py::test_split_img tests/test_utils.py::test_split_img_random
python -m pytest -q tests
```

## Confidence

High for the narrow `split_img` compatibility fix. The change replaces a removed NumPy alias with the current equivalent API and the existing focused tests pass.

## What Would Change This Conclusion

Evidence that `np.prod(divisors)` differs from `np.product(divisors)` for supported NumPy versions or dtypes would require rework.
