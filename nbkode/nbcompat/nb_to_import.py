"""
    nbkode.nb_to_import
    ~~~~~~~~~~~~~~~~~~~

    Select which numba to import based on NBKODE_NONUMBA environmental variable.

    We keep this here to allow importing within nbkode.nbcompat module
    and avoid circular imports.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import os

if os.environ.get("NBKODE_NONUMBA", 0):
    from . import numbasub as numba  # noqa: F401

    def is_jitted(func):
        return True

    NO_NUMBA = True

else:
    import numba  # noqa: F401
    from numba.extending import is_jitted  # noqa: F401

    numba.jitclass = numba.experimental.jitclass

    NO_NUMBA = False
