# Issue 5 red-test design

## Acceptance Criteria As Testable Claims

- `split_img` should tile cropped image arrays without relying on removed NumPy aliases.
- Existing `split_img` shape tests should pass under NumPy 2.x.
- No tests should be skipped, loosened, or deleted.

## Tests Used

Existing tests were sufficient:

```bash
python -m pytest -q tests/test_utils.py::test_split_img tests/test_utils.py::test_split_img_random
```

These tests failed on the base commit for the intended reason.
