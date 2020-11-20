import numpy as np

from ..util import classproperty
from .core import ImplicitMultistep


class BDF(ImplicitMultistep, abstract=True):
    GROUP = "BDF"

    @classproperty
    def B(cls):
        return np.zeros_like(cls.A)


class BDF1(BDF):
    A = np.array([-1.0])
    Bn = 1.0


class BDF2(BDF):
    A = np.array([1, -4]) / 3
    Bn = 2 / 3


class BDF3(BDF):
    A = np.array([-2, 9, -18]) / 11
    Bn = 6 / 11


class BDF4(BDF):
    A = np.array([3, -16, 36, -48]) / 25
    Bn = 12 / 25


class BDF5(BDF):
    A = np.array([-12, 75, -200, 300, -300]) / 137
    Bn = 60 / 137


class BDF6(BDF):
    A = np.array([10, -72, 225, -400, 450, -360]) / 147
    Bn = 60 / 147
