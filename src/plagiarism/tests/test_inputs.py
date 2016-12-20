import pytest
from mock import mock

from plagiarism.input import *


def test_yn_message():
    assert default_from_yn_message('foo [y/n] ') is True
    assert default_from_yn_message('foo [y] ') is True
    assert default_from_yn_message('foo [Y] ') is True
    assert default_from_yn_message('foo [n] ') is False
    assert default_from_yn_message('foo [y/N] ') is False
    assert default_from_yn_message('foo [Y/n] ') is True
    assert default_from_yn_message('foo [ Y / n ] ') is True

    with pytest.raises(ValueError):
        default_from_yn_message('foo')


def test_proxy():
    assert Proxy(list, [1, 2, 3]).index(1) == 0
    assert bool(Proxy(list)) is False
    assert len(Proxy(list, [1, 2, 3])) == 3
    assert Proxy(list, [1, 2, 3])[2] == 3
    assert Proxy(list, [1, 2, 3]) + [4] == [1, 2, 3, 4]


def test_ask_function():
    for func in ask, ask_lazy:
        assert func(None, tuple) == ()
        assert func(None, tuple, (1, 2, 3)) == (1, 2, 3)
        assert func(False, tuple) is False


def test_yn_question():
    with mock.patch('plagiarism.input.get_input', return_value=''):
        assert yn_input('foo [y/n]') is True
        assert yn_input('foo [y/N]') is False

    with mock.patch('plagiarism.input.get_input', return_value='y'):
        assert yn_input('foo [y/n]') is True
        assert yn_input('foo [y/N]') is True

    with mock.patch('plagiarism.input.get_input', return_value='no'):
        assert yn_input('foo [y/n]') is False
        assert yn_input('foo [y/N]') is False
