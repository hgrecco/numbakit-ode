from numpy import empty, ndarray

from .nbcompat import numba

spec = [
    ("ts", numba.float64[::1]),
    ("ys", numba.float64[:, ::1]),
    ("fs", numba.float64[:, ::1]),
]


@numba.experimental.jitclass(spec)
class AlignedBuffer:
    def __init__(self, capacity: int, t: ndarray, y: ndarray, f: ndarray):
        self.ts = empty(capacity)
        self.ys = empty((capacity, y.size))
        self.fs = empty((capacity, y.size))

        self.t = t
        self.y = y
        self.f = f

    @property
    def t(self):
        return self.ts[-1]

    @t.setter
    def t(self, value):
        self.ts[:-1] = self.ts[1:]
        self.ts[-1] = value

    @property
    def y(self):
        return self.ys[-1]

    @y.setter
    def y(self, value):
        self.ys[:-1] = self.ys[1:]
        self.ys[-1] = value

    @property
    def f(self):
        return self.fs[-1]

    @f.setter
    def f(self, value):
        self.fs[:-1] = self.fs[1:]
        self.fs[-1] = value
