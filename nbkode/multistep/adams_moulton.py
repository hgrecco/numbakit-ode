"""
    nbkode.adams_moulton
    ~~~~~~~~~~~~~~~~~~~~

    Methods of the Adams–Moulton family (implicit methods).
    - AdamsMoulton1
    - AdamsMoulton2
    - AdamsMoulton3
    - AdamsMoulton4
    - AdamsMoulton5

    See: https://en.wikipedia.org/wiki/Linear_multistep_method

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


import numpy as np

from ..util import classproperty
from .core import ImplicitMultistep


class AdamsMoulton(ImplicitMultistep, abstract=True):
    GROUP = "Adams-Moulton"

    @classproperty
    def A(cls):
        A = np.zeros_like(cls.B)
        A[-1] = -1
        return A


class AdamsMoulton1(AdamsMoulton):
    B = np.array([0.0])
    Bn = 1.0


class AdamsMoulton2(AdamsMoulton):
    B = np.array([1 / 2])
    Bn = 1 / 2


class AdamsMoulton3(AdamsMoulton):
    B = np.array([-1 / 12, 2 / 3])
    Bn = 5 / 12


class AdamsMoulton4(AdamsMoulton):
    B = np.array([19, -5, 1]) / 24
    Bn = 9 / 24


class AdamsMoulton5(AdamsMoulton):
    B = np.array([646, -264, 106, -19]) / 720
    Bn = 251 / 720
