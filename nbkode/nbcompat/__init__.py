"""
    nbkode.nbcompat
    ~~~~~~~~~~~~~~~

    Numba compatible functions. Hopefully, some of these will be unnecessary
    in the future as Numba continues to grow.

    Current functions and their origin:
    - newton: scipy.optimize.newton
    - isclose: numpy.isclose
    - clip: numpy.clip

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import os

from .common import clip, isclose
from .newton import j_newton, newton

if os.environ.get("NBKODE_NONUMBA", 0):
    from . import numbasub as numba
else:
    import numba

__all__ = ["clip", "isclose", "j_newton", "newton", "numba"]
