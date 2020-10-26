"""
    benchmarks.against_scipy
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Comparisons using SciPy as a gold standard.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
from scipy import integrate

from nbkode import runge_kutta
from nbkode.nbcompat import numba

RK23s = ("nbkode RungeKutta23", "SciPy RK23")
RK45s = ("nbkode RungeKutta45", "SciPy RK45")

by_name = {
    "nbkode RungeKutta23": runge_kutta.RungeKutta23,
    "nbkode RungeKutta45": runge_kutta.RungeKutta45,
    "SciPy RK23": integrate.RK23,
    "SciPy RK45": integrate.RK45,
}


y0 = np.atleast_1d(1.0)

sol = None


def time_f1_instantiate1_1step(solver_cls_name, jit_before):
    global sol

    solver_cls = by_name[solver_cls_name]

    if solver_cls.__module__.startswith("nbkode"):

        if jit_before:

            @numba.njit()
            def f(t, y, k):
                return k * y

        else:

            def f(t, y, k):
                return k * y

        _ = f(0.0, y0, -0.01)

        sol = solver_cls(f, 0.0, y0, args=(-0.01,))
    else:
        k = -0.01

        if jit_before:

            @numba.njit()
            def f(t, y):
                return k * y

        else:

            def f(t, y):
                return k * y

        _ = f(0.0, y0)

        sol = solver_cls(f, 0.0, y0, t_bound=10_000_000_000)
    sol.step()


time_f1_instantiate1_1step.params = (RK23s + RK45s, [True, False])
time_f1_instantiate1_1step.param_names = ["solver", "jit_before"]


def time_f1_rk23_10k(solver_cls, jit_before):
    for n in range(10_000):
        sol.step()


time_f1_rk23_10k.params = (RK23s, [True, False])
time_f1_rk23_10k.param_names = ["solver", "jit_before"]
time_f1_rk23_10k.setup = time_f1_instantiate1_1step


def time_f1_rk45_10k(solver_cls, jit_before):
    for n in range(10_000):
        sol.step()


time_f1_rk45_10k.params = (RK45s, [True, False])
time_f1_rk45_10k.param_names = ["solver", "jit_before"]
time_f1_rk45_10k.setup = time_f1_instantiate1_1step
