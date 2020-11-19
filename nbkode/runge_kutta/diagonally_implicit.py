import numpy as np

from ..util import classproperty
from .core import RungeKutta


class DIRK(RungeKutta):
    """Diagonally Implicit Runge-Kutta (DIRK) method.

    In DIRK methods, the upper triangle of the A matrix is 0.
    In contrast to fully-implicit methods (FIRK), where an
    implicit system of equations needs to be solved for all ki
    simultaneously, in DIRK methods an equation for a single ki
    needs to be solved at stage.
    """

    def __init_subclass__(cls, *args, **kwargs) -> None:
        assert cls.diagonally_implicit
        super().__init_subclass__(*args, **kwargs)

    @classproperty
    def diagonally_implicit(cls):
        return np.all(np.triu(cls.A, k=1) == 0)


class SDIRK(DIRK):
    """Singly Diagonally Implicit Runge-Kutta (DIRK) method."""

    def __init_subclass__(cls, *args, **kwargs) -> None:
        assert cls.singly_diagonally_implicit
        super().__init_subclass__(*args, **kwargs)

    @classproperty
    def singly_diagonally_implicit(cls):
        return np.unique(np.diag(cls.A)).size == 1
