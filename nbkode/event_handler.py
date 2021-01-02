"""
    nbkode.events
    ~~~~~~~~~~~~

    Methods for dealing with events.

    Adapted from: https://github.com/scipy/scipy/blob/v1.5.4/scipy/integrate/_ivp/ivp.py

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from typing import Callable, Iterable

import numpy as np
from numpy import ndarray

from .nbcompat import NO_NUMBA, is_jitted, numba, zeros


@numba.njit()
def _dummy_event(t, y):
    return y[0]


if NO_NUMBA:
    TypedList = list
else:
    TypedList = numba.typed.List
    _event_type = numba.types.float64(numba.types.float64, numba.types.float64[:])


@numba.njit()
def event_at_sol(t, func, interpolate, rhs, cache, *args):
    """Helper function to find the root for the event function along the solution.

    Parameters
    ----------
    t : float
        timepoint
    func : callable
        jitted event function.
    interpolate : callable
        jitted interpolation function of the solution
    rhs : callable
        right hand side of the dynamical system.
    cache : AlignedBuffer
        cache of the solver.
    args
        extra arguments required for the interpolation function.

    Returns
    -------
    float

    """
    return func(t, interpolate(t, rhs, cache, *args))


# TODO: Remove this UGLY HACK
@numba.njit()
def _get_tl(val):
    """Helper function to create typed lists. Should be removed when
    we can make empty list to work.
    """
    out = TypedList()
    out.append(val)
    return out


@numba.njit()
def _empty_list(val):
    """Helper function to create typed lists. Should be removed when
    we can make empty list to work.
    """
    out = _get_tl(val)
    out.pop()
    return out


if NO_NUMBA:
    _EVENT_SPEC = None
else:
    _EVENT_SPEC = [
        ("func", _event_type.as_type()),
        ("is_terminal", numba.bool_),
        ("direction", numba.int_),
        ("last_t", numba.float64),
        ("last_value", numba.float64),
        ("t", numba.types.ListType(numba.float64)),
        ("y", numba.types.ListType(numba.float64[::1])),
    ]


@numba.jitclass(_EVENT_SPEC)
class Event:
    """Helper class to deal with event.

    An event occurs at the zeros of a continuous function of time and state.

    Parameters
    ----------
    func : callable
        jitted function to calculate the event function.
    terminal: bool
        Whether to terminate integration if this event occurs.
    direction: int
        Direction of a zero crossing. If `direction` is positive,
        `event` will only trigger when going from negative to positive,
        and vice versa if `direction` is negative. If 0, then either
        direction will trigger event.
    init_t
    init_value

    Attributes
    ----------
    t : numba typed list of Event (length N)
    y : numba typed list of Event (length N)
    """

    def __init__(self, func, is_terminal, direction, init_t, init_y):
        self.func = func
        self.is_terminal = is_terminal
        self.direction = direction
        self.last_t = init_t
        self.last_value = func(init_t, init_y)

        self.t = _empty_list(init_t)
        self.y = _empty_list(init_y)

    def evaluate(self, interpolate, rhs, cache, *args):
        t = cache.t
        y = cache.y

        value = self.func(t, y)

        up = (self.last_value <= 0.0) & (value >= 0.0)
        down = (self.last_value >= 0.0) & (value <= 0.0)
        either = up | down

        trigger = (
            up & (self.direction > 0)
            | down & (self.direction < 0)
            | either & (self.direction == 0)
        )

        if trigger:
            if value == 0:
                root = t
            else:
                root = zeros.bisect(
                    event_at_sol,
                    self.last_t,
                    t,
                    args=(self.func, interpolate, rhs, cache, *args),
                )

            # This is required to avoid duplicates
            if not (self.t and self.t[-1] == root):
                self.t.append(root)
                self.y.append(interpolate(root, rhs, cache, *args))

        self.last_t = t
        self.last_value = value

        return trigger and self.is_terminal

    @property
    def last_event(self):
        if self.t:
            return self.t[-1], self.y[-1]
        return np.nan, np.empty(0) * np.nan


if NO_NUMBA:
    _EVENT_HANDLER_SPEC = None
else:
    _EVENT_HANDLER_SPEC = [
        ("events", numba.types.ListType(Event.class_type.instance_type)),
        (
            "last_event",
            numba.types.Tuple((numba.types.float64, numba.types.float64[:])),
        ),
    ]


@numba.jitclass(_EVENT_HANDLER_SPEC)
class EventHandler:
    """Helper class to deal with multiple events.

    N is the number of events to be tracked.

    Parameters
    ----------
    events : numba typed list of Event (length N)

    """

    def __init__(self, events):
        self.events = events
        self.last_event = np.nan, np.empty(0) * np.nan

    def evaluate(self, interpolate, rhs, cache, *args):
        """
        Parameters
        ----------
        interpolate : callable
            interpolator function.
        rhs : callable
            Right-hand side of the system. The calling signature is ``fun(t, y)``.
            Here ``t`` is a scalar, and the ndarray ``y`` hasna shape (n,);
            then ``fun`` must return array_like with shape (n,).
        cache : AlignedBuffer
        args
            extra arguments provided to interpolate.

        Returns
        -------
        bool
            True if it should terminate.
        """

        terminate = False
        min_t = -np.inf
        for ndx, event in enumerate(self.events):
            if event.evaluate(interpolate, rhs, cache, *args):
                terminate = True
                t, y = event.last_event
                if t > min_t:
                    self.last_event = t, y

        return terminate


def build_handler(events: Iterable[Callable], t: float, y: ndarray) -> EventHandler:
    """Standardize event functions and extract is_terminal and direction."""

    if callable(events):
        events = (events,)

    evs = TypedList()

    if events is not None:
        for ndx, event in enumerate(events):
            try:
                is_terminal = event.terminal
            except AttributeError:
                is_terminal = False

            try:
                direction = int(np.sign(event.direction))
            except AttributeError:
                direction = 0

            if not is_jitted(event):
                event = numba.njit()(event)

            evs.append(Event(event, is_terminal, direction, t, y))

    return EventHandler(evs)
