
Migrating from SciPy
====================

If you are have been using ScipPy to integrate ODEs, moving to numbakit-ode
is very easy.

Migrating from SciPy class based API
------------------------------------

Instantiating the solver class is the mostly the same (except for the
class name, see table below) So `RK23(fun, t0, y0)` becomes
`RungeKutta45(fun, t0, y0)`. The `step` method available in the SciPy
solver instance is also available in the numbakit-ode instance
and has the same syntax (numbakit-ode `step` method provides a
few extra parameters but the defaults match the SciPy semantics)

======== ==============
 SciPy       nbkode
======== ==============
  RK23    RungeKutta23
  RK45    RungeKutta45
 DOP853      DOP853
  BDF        BDF<N>*
  Radau       N/A
  LSODA       N/A
======== ==============

* We provide BDF, with N=1 to 5
N/A: not available (yet :-))

All extra parameters can be given as named arguments using the same
name as in SciPy.


Migrating from SciPy solve_ivp
------------------------------

SciPy `solve_ivp` creates a Solver instances and step through the solution
until the desired timepoint. numbakit-ode takes a different approach:
it provides feature rich solver class that you keep alive while you need it.

To **run until a specific time**:

.. code-block:: python

    >>> out = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
    >>> print(out.t)
    >>> print(out.y)

in numbakit-ode becomes:

.. code-block:: python

    >>> sol = nbkode.RungeKutta45(exponential_decay, 0, [2, 4, 8])
    >>> t, y = sol.run(10)
    >>> print(t)
    >>> print(y)

Resuming the integration in SciPy involves calling `solve_ivp` again:

.. code-block:: python

    >>> out2 = solve_ivp(exponential_decay, [out.t, 20], out.y)
    >>> print(out2.t)
    >>> print(out2.y)

but bear in mind that for some solvers this is not exactly the same as
resuming as the internal state of the solver (which might contain previous
evaluations of the function or jacobian) are lost.

That is why in numbakit-ode we rather keep the solver alive and then
call it again if necessary:

.. code-block:: python

    >>> t2, y = sol.run(10)
    >>> print(t2)
    >>> print(yt)

To **evaluate at specific timepoints** in SciPy:

.. code-block:: python

    >>> out = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], t_eval=[0, 1, 2, 4, 10])
    
in numbakit-ode becomes:

    >>> sol = nbkode.RungeKutta45(exponential_decay, 0, [2, 4, 8])
    >>> t, y = sol.run([0, 1, 2, 4, 10])

To use **events** in SciPy:

    >>> out = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)
    >>> print(out.t_events)
    >>> print(out.y_events)

in numbakit-ode becomes:

    >>> sol = nbkode.RungeKutta45(exponential_decay, 0, [2, 4, 8])
    >>> t, y, t_events, y_events = sol.run_events(100, events=hit_ground)
    >>> print(t_events)
    >>> print(y_events)

Keep in mind that in `numbakit-ode` time always move forward.

