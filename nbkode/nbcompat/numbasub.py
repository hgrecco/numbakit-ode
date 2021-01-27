"""
    nbkode.nbcompat.numbasub
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Noop decorators disguised as numba.

    Copied from https://github.com/ptooley/numbasub

    See also: https://stackoverflow.com/questions/3888158

    :copyright: 2020 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import functools


def optional_arg_decorator(fn):
    @functools.wraps(fn)
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return fn(args[0])

        else:

            def real_decorator(decoratee):
                return fn(decoratee, *args, **kwargs)

            return real_decorator

    return wrapped_decorator


@optional_arg_decorator
def __noop(func, *args, **kwargs):
    return func


autojit = __noop
generated_jit = __noop
guvectorize = __noop
jit = __noop
jitclass = __noop
njit = __noop
vectorize = __noop
typeof = __noop


class WithGetItem:
    def __class_getitem__(cls, item):
        return None


b1 = WithGetItem
bool_ = WithGetItem
boolean = WithGetItem
byte = WithGetItem
c16 = WithGetItem
c8 = WithGetItem
char = WithGetItem
complex128 = WithGetItem
complex64 = WithGetItem
double = WithGetItem
f4 = WithGetItem
f8 = WithGetItem
ffi = WithGetItem
ffi_forced_object = WithGetItem
float32 = WithGetItem
float64 = WithGetItem
float_ = WithGetItem
i1 = WithGetItem
i2 = WithGetItem
i4 = WithGetItem
i8 = WithGetItem
int16 = WithGetItem
int32 = WithGetItem
int64 = WithGetItem
int8 = WithGetItem
int_ = WithGetItem
intc = WithGetItem
intp = WithGetItem
long_ = WithGetItem
longlong = WithGetItem
none = WithGetItem
short = WithGetItem
u1 = WithGetItem
u2 = WithGetItem
u4 = WithGetItem
u8 = WithGetItem
uchar = WithGetItem
uint = WithGetItem
uint16 = WithGetItem
uint32 = WithGetItem
uint64 = WithGetItem
uint8 = WithGetItem
uintc = WithGetItem
uintp = WithGetItem
ulong = WithGetItem
ulonglong = WithGetItem
ushort = WithGetItem
void = WithGetItem
