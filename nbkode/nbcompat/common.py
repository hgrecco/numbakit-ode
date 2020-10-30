"""
    nbkode.nbcompat.common
    ~~~~~~~~~~~~~~~~~~~~~~

    Common methods.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from .nb_to_import import numba


@numba.njit()
def isclose(a, b, atol, rtol):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


@numba.njit
def clip(x, xmin, xmax):
    return min(max(x, xmin), xmax)
