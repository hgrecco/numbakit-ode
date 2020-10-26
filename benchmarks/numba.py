"""
    benchmarks.against_scipy
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Comparisons of the different methods with and without numba.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import os

import numpy as np

# Do not import this automatically as we want to be able to disable Numba
# and this requires not importing nbkode soon.
names = [
    "AdamsBashforth1",
    "AdamsBashforth2",
    "AdamsBashforth3",
    "AdamsBashforth4",
    "AdamsBashforth5",
    # 'AdamsMoulton1', 'AdamsMoulton2', 'AdamsMoulton3', 'AdamsMoulton4', 'AdamsMoulton5',
    "ForwardEuler",  #'BackwardEuler',
    "RungeKutta23",
    "RungeKutta45",
]

y0 = np.atleast_1d(1.0)

sol = None


def time_instantiate1_1step(solver_cls_name, numba_enabled):
    global sol

    if not numba_enabled:
        os.environ["NBKODE_NONUMBA"] = "1"

    import nbkode

    solvers = {solver.__name__: solver for solver in nbkode.get_solvers()}
    solver_cls = solvers[solver_cls_name]

    def f(t, x, k):
        return k * x

    sol = solver_cls(f, 0.0, y0, args=(-0.01,))

    sol.step()


time_instantiate1_1step.params = (names, [True, False])
time_instantiate1_1step.param_names = ["solver", "numba_enabled"]


def time_1d_10k(solver_cls_name, numba_disabled):
    for n in range(10_000):
        sol.step()


time_1d_10k.setup = time_instantiate1_1step
time_1d_10k.params = (names, [True, False])
time_1d_10k.param_names = ["solver", "numba_enabled"]
