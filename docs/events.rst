

Events
======

`numbakit-ode` provides an events API very similar to the one provided
by SciPy solve_ivp. Briefly, the events API allows to register the time
and state in which certain conditions are met. It also provides a way
to define a termination condition.

As the API is the one from SciPy, we copy here from the SciPy docs, adapting
where necessary:

Each event occurs at the zeros of a continuous function of time and state.
Each function must have the signature `event(t, y)` and return a float.
The solver will find an accurate value of `t` at which `event(t, y(t)) = 0`
using a root-finding algorithm. By default, all zeros will be found.

The solver looks for a sign change over each step, so if multiple zero crossings
occur within one step, events may be missed. Additionally each event function
might have the following attributes::

 - **terminal**: bool, optional
   Whether to terminate integration if this event occurs.
   Implicitly `False` if not assigned.
 - **direction**: float, optional
   Direction of a zero crossing. If direction is positive, event will only
   trigger when going from negative to positive, and vice versa if direction
   is negative. If 0, then either direction will trigger event.
   Implicitly 0 if not assigned.

You can assign attributes like `event.terminal = True` to any function in Python.
To integrate the ODE using the events API, use the Solver `run_events` method
(see below).


Example
-------

Cannon fired upward with terminal event upon impact. The terminal and
direction fields of an event are applied by monkey patching a
function. Here `y[0]` is position and `y[1]`  is velocity.

The projectile starts at position 0 with velocity +10. Note that the
integration never reaches t=100 because the event is terminal.

.. doctest::

    >>> import numpy as np
    >>> import nbkode
    >>> def upward_cannon(t, y):
    ...    return np.asarray([y[1], -0.5])
    >>> def hit_ground(t, y):
    ...    return y[0]
    >>> hit_ground.terminal = True
    >>> hit_ground.direction = -1
    >>> sol = nbkode.RungeKutta45(upward_cannon, t0=0, y0=np.asarray([0, 10]))
    >>> t, y, t_events, y_events = sol.run_events(100, events=hit_ground)
    >>> print(t_events) # hit ground time   # doctest: +SKIP

`run_events` works like `run` but it has an extra argument (`events`) that
can accept a callable (or a list of callables).
It also has two additional outputs::

 - `t_events`: list of list of floats
   Contains for each callable a list of floats at which an event of
   that event was detected.
 - `y_events`: list of list of arrays
   For each value of t_events, the corresponding value of the solution.

Use events to find position, which is 100, at the apex of the cannonball’s
trajectory. Apex is not defined as terminal, so both apex and hit_ground
are found.

.. doctest::

    >>> def apex(t, y):
    ...     return y[1]
    >>> sol = nbkode.RungeKutta45(upward_cannon, t0=0, y0=np.asarray([0, 10]))
    >>> t, y, t_events, y_events = sol.run_events(100, events=(hit_ground, apex))
    >>> print(t_events[0]) # hit ground time  # doctest: +SKIP
    >>> print(t_events[1]) # apex time  # doctest: +SKIP

 .. note:: Important

    Event callables are compiled into the tight loop that runs the simulation
    using Numba. Therefore, just like the ODE rhs, these callables **must**
    compatible with Numba jit in nopython mode (`njit`)