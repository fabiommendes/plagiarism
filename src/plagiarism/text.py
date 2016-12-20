"""
Functions that manipulate text input.
"""

import re

import itertools
from unidecode import unidecode
from difflib import Differ


text_differ = Differ()
WHITESPACE_RE = re.compile(r'^[ \t]*')
PUNCTUATION = '.,:;?!)]}-%#/\\'
PUNCTUATION_SYMBOLS = set(PUNCTUATION)

__all__ = [
    'strip_punctuation', 'remove_punctuation', 'remove_accents', 'dedent',
]


def strip_punctuation(word):
    """
    Remove punctuation from the end of word.
    """

    return word.strip(PUNCTUATION)


def remove_punctuation(text):
    """
    Remove punctuation from all text.
    """

    blacklist = PUNCTUATION_SYMBOLS
    return ''.join(c for c in text if c not in blacklist)


def remove_accents(text):
    """
    Remove accents from text.
    """

    return unidecode(text)


def dedent(text):
    """
    Remove indentation from text.
    """

    lines = []
    for line in text.splitlines():
        _, i = WHITESPACE_RE.search(line).span()
        lines.append(line[i:])
    return '\n'.join(lines)


def line_rstrip(text):
    """
    Strips whitespace from the end of lines.
    """
    return '\n'.join(line.rstrip() for line in text.splitlines())


def line_lstrip(text):
    """
    Strips whitespace from the beginning of each line.
    """
    return '\n'.join(line.lstrip() for line in text.splitlines())


def line_strip(text):
    """
    Strips whitespace from beginning and end of each line.
    """
    return '\n'.join(line.strip() for line in text.splitlines())


def text_diff(text1, text2, strip=False):
    """
    Returns a string with the difference between text1 and text2 in a similar
    way to the diff command.

    If strip=True, strips trailing whitespace from all lines.
    """

    if strip:
        text1, text2 = map(line_rstrip, [text1, text2])
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = text_differ.compare(lines1, lines2)
    diff = (line.rstrip() for line in diff)
    return '\n'.join(diff)


def two_column(text1, text2, size=120):
    """
    Display two strings as in a two column layout.
    """

    # Replace tabs for spaces
    text1 = text1.replace('\t', '    ')
    text2 = text2.replace('\t', '    ')

    # Split lines
    colsize = size // 2 - 1
    lines1 = [line[:colsize] for line in text1.splitlines()]
    lines2 = [line[:colsize] for line in text2.splitlines()]
    fmt = '{:<%d}| {:<%d}' % (colsize, colsize)
    lines = itertools.zip_longest(lines1, lines2, fillvalue='')
    lines = (fmt.format(*item) for item in lines)
    return '\n'.join(lines)
