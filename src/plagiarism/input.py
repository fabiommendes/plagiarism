import os
import re

YN_REGEX = re.compile(r'''
    [\(\[]\s*
        (?P<first>[yn]|yes|no)\s*(?P<second>/\s*([yn]|yes|no))?
    \s*[\)\]]\s*$''', re.VERBOSE | re.IGNORECASE)


class Proxy:
    """
    A proxy element.
    """

    def __init__(self, func, *args, **kwargs):
        self.__func_ = func
        self.__args_ = args
        self.__kwargs_ = kwargs
        self.__has_value_ = False

    def __touch(self):
        if not self.__has_value_:
            self.__value = self.__func_(*self.__args_, **self.__kwargs_)
            self.__has_value_ = True

    def __getattr__(self, attr):
        self.__touch()
        return getattr(self.__value, attr)

    def __bool__(self):
        self.__touch()
        return bool(self.__value)

    def __eq__(self, other):
        self.__touch()
        return self.__value == other

    @classmethod
    def _make_proxy_names(cls):
        def method_factory(name):
            def method(self, *args, **kwargs):
                self.__touch()
                return getattr(self._Proxy__value, name)(*args, **kwargs)

            return method

        names = set(dir(float) + dir(int) + dir(list))
        names = {x for x in names if x.startswith('__')}
        names.difference_update(dir(object))
        for name in names:
            if hasattr(cls, name):
                continue
            method = method_factory(name)
            setattr(cls, name, method)


Proxy._make_proxy_names()


def ask(value, func, *args, **kwargs):
    """
    Execute function with the given arguments if value is None and return.
    Otherwise, return value.
    """
    if value is None:
        return func(*args, **kwargs)
    else:
        return value


def ask_lazy(value, func, *args, **kwargs):
    """
    Like ask(), but returns a lazy proxy object that calls the given function
    on demand.
    """
    if value is None:
        return Proxy(func, *args, **kwargs)
    else:
        return value


def get_input(*args):
    """
    A mockable input() function.
    """
    return input(*args)


def default_from_yn_message(msg):
    match = YN_REGEX.search(msg)
    if not match:
        raise ValueError('please provide a default or a valid message '
                         'specifier.')
    first = match.group('first')[0]
    second = match.group('second')
    if not second:
        return {'y': True, 'n': False}[first.lower()]
    second = second[1:].strip()[0]
    return bool({'yn': 1, 'YN': 1, 'yN': 0, 'Yn': 1,
                 'ny': 0, 'NY': 0, 'nY': 1, 'Ny': 0}[first + second])


def yn_input(msg, default=None):
    """
    Asks a yes/no question.
    """

    if default is None:
        default = default_from_yn_message(msg)
    raw_data = get_input(msg).lower()
    data = raw_data.lower()
    if not data:
        return default
    else:
        if data in ['y', 'yes']:
            return True
        elif data in ['n', 'no']:
            return False
        else:
            return yn_input(msg)


def do_print(*args, **kwargs):
    """
    A mockable print()
    """
    return print(*args, **kwargs)


def no_print(*args, **kwargs):
    """
    Do-nothing: useful to disable printing, while keeping its interface.
    """


def clear():
    """
    Clear console.
    """
    os.system('clear' if os.name == 'nt' else 'clear')
