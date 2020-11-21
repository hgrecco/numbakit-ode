"""
    benchmarks.nbcompat
    ~~~~~~~~~~~~~~~~~~~

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from numba import njit
from scipy.optimize import newton

from nbkode.nbcompat.zeros import _j_newton as j_newton


@njit
def func(x):
    return x ** 3 - 1


@njit
def fprime(x):
    return 3 * x ** 2


@njit
def njit_newton(func, x0, fprime):
    for _ in range(50):
        fder = fprime(x0)
        fval = func(x0)
        newton_step = fval / fder
        x = x0 - newton_step
        if abs(x - x0) < 1.48e-8:
            return x
        x0 = x


class Suite:

    param_names = "variant", "numba"
    params = (["scipy", "simple", "nbkode"], [True, False])

    value = 1.5

    def setup(self, variant, numba):
        self.func = func
        self.fprime = fprime

        if variant == "scipy":
            self.newton = newton
        elif variant == "simple":
            self.newton = njit_newton
        elif variant == "nbkode":
            self.newton = j_newton

        if not numba:
            self.func = self.func.py_func
            self.fprime = self.fprime.py_func
            if variant != "scipy":
                self.newton = self.newton.py_func

    def time_newton(self, variant, numba):
        self.newton(self.func, self.value)

    def time_newton_fprime(self, variant, numba):
        self.newton(self.func, self.value, fprime=self.fprime)
