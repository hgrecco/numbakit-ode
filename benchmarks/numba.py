"""
    benchmarks.against_scipy
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Comparisons of the different methods with and without numba.

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import os

import numpy as np

from .common import NumbaStepModes

# Do not import this automatically as we want to be able to disable Numba
# and this requires not importing nbkode soon.
names = [
    "AdamsBashforth1",
    "AdamsBashforth2",
    "AdamsBashforth3",
    "AdamsBashforth4",
    "AdamsBashforth5",
    "AdamsMoulton1",
    "AdamsMoulton2",
    "AdamsMoulton3",
    "AdamsMoulton4",
    "AdamsMoulton5",
    "ForwardEuler",
    "BackwardEuler",
    "RungeKutta23",
    "RungeKutta45",
    "DOP853",
]

BOOLEANS = (True, False)
NUMBA_MODES = tuple(NumbaStepModes.__members__.keys())


y0 = np.atleast_1d(1.0)


func = None


def define_func(numba_enabled):
    global func
    if not numba_enabled:
        os.environ["NBKODE_NONUMBA"] = "1"

    # Leave the import inside to ensure proper NBKODE_NONUMBA
    import nbkode  # noqa: F401
    from nbkode.nbcompat import numba

    @numba.njit()
    def f(t, x, k):
        return k * x

    _ = f(0.0, y0, -0.01)
    func = f


sol = None


def define_sol(integrator):
    global sol

    # Leave the import inside to ensure proper NBKODE_NONUMBA
    import nbkode

    solvers = {solver.__name__: solver for solver in nbkode.get_solvers()}
    solver_cls = solvers[integrator]

    sol = solver_cls(func, 0.0, y0, params=(-0.01,))


###############
# Instantiate
###############


def setup_instantiate(integrator, numba_enabled):
    define_func(numba_enabled)


def time_instantiate(integrator, numba_enabled):
    return define_sol(integrator)


time_instantiate.setup = setup_instantiate
time_instantiate.params = (names, BOOLEANS)
time_instantiate.param_names = ("integrator", "numba_enabled")


###############
# First Step
###############


def setup_time_f1_first_step(integrator, numba_enabled):
    define_func(numba_enabled)
    define_sol(integrator)


def time_f1_first_step(integrator, numba_enabled):
    sol.step()


time_f1_first_step.setup = setup_time_f1_first_step
time_f1_first_step.params = (names, BOOLEANS)
time_f1_first_step.param_names = ("integrator", "numba_enabled")


###############
# Run 10k
###############


def setup_time_f1_run10k(integrator, other):
    numba_enabled = other != NumbaStepModes.NUMBA_DISABLED.name
    define_func(numba_enabled)
    define_sol(integrator)
    sol.step()
    # warm up _nsteps
    sol.step(n=2)


def time_f1_run10k(integrator, other):
    if other == NumbaStepModes.INTERNAL_LOOP.name:
        sol.step(n=10_000)
    else:
        for n in range(10_000):
            sol.step()


time_f1_run10k.setup = setup_time_f1_run10k
time_f1_run10k.params = (names, NUMBA_MODES)
time_f1_run10k.param_names = ("integrator", "other")
