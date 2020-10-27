"""
    benchmarks.common
    ~~~~~~~~~~~~~~~~~

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from enum import IntEnum, auto


class NumbaStepModes(IntEnum):
    NUMBA_DISABLED = auto()
    EXTERNAL_LOOP = auto()
    INTERNAL_LOOP = auto()
