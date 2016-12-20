from plagiarism.text import *
from plagiarism.text import text_diff, two_column, line_strip, line_rstrip, \
    line_lstrip


def test_dedent():
    assert dedent('foo') == 'foo'
    assert dedent('  foo') == 'foo'
    assert dedent('foo\nbar') == 'foo\nbar'
    assert dedent('foo\n  bar') == 'foo\nbar'
    assert dedent('foo\n\nbar') == 'foo\n\nbar'
    assert dedent('  foo\n\nbar') == 'foo\n\nbar'


def test_remove_accents():
    assert remove_accents('fábio') == 'fabio'


def test_remove_punctuation():
    assert remove_punctuation('word, word. word!') == 'word word word'


def test_strip_punctuation():
    assert strip_punctuation('word, word. word!') == 'word, word. word'


def test_diff():
    t1 = 'foo\nbar\nfoobar'
    t2 = 'foo\nham-spam\nfoobar'
    assert text_diff(t1, t2) == '  foo\n- bar\n+ ham-spam\n  foobar'


def test_two_column():
    t1 = 'foo\nbar\nfoobar'
    t2 = 'foo\nham-spam\nfoobar'
    assert two_column(t1, t2, 20) == 'foo      | foo      \n' \
                                     'bar      | ham-spam \n' \
                                     'foobar   | foobar   '


def test_line_strip():
    assert line_strip('foo \n  bar  \n foobar') == 'foo\nbar\nfoobar'
    assert line_rstrip('foo \n  bar  \n foobar') == 'foo\n  bar\n foobar'
    assert line_lstrip('foo \n  bar  \n foobar') == 'foo \nbar  \nfoobar'
