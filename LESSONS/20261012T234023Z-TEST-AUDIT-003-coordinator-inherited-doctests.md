# Runtime docstrings affect doctest inventory

- ID: 20261012T234023Z-TEST-AUDIT-003-coordinator-inherited-doctests
- Date: 2026-10-12T23:40:23Z
- Task/role: TEST-AUDIT-003 / coordinator
- Status: confirmed
- Observation: Searching literal doctest examples found seven functions, but full collection includes an eighth inherited docstring test.
- Evidence: At base `04cbac9b59fa695218ddbafb6978529eafd8a2bc`, `dphtools/utils/__init__.py:338` appends `np.pad.__doc__` to `fft_pad.__doc__`. `/private/tmp/dphtools-test-audit-003-after-collection.txt` collects `dphtools/utils/__init__.py::dphtools.utils.fft_pad` among eight doctest nodes.
- Lesson: Use actual doctest collection to inventory executable public documentation; source searches alone miss runtime-composed docstrings.
