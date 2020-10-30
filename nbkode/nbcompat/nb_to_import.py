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
    from . import numbasub as numba
else:
    import numba
