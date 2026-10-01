#!/usr/bin/env python3
"""Check the actual installed release using its disposable environment interpreter."""

from importlib import metadata
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import numpy, pandas, scipy, skimage, dphtools
from dphtools import utils

assert (
    metadata.version("dphtools") == sys.argv[1] == dphtools.__version__
), "installed version mismatch"
for module in (dphtools, utils, numpy, pandas, scipy, matplotlib, skimage):
    assert (
        Path(str(module.__file__)).resolve().is_relative_to(Path(sys.prefix).resolve())
    ), "import escaped clean environment"
assert (
    utils.bin_ndarray(numpy.arange(4).reshape(2, 2), new_shape=(1, 1), operation="sum").item() == 6
), "array sum mismatch"
print("Installed release smoke passed")
