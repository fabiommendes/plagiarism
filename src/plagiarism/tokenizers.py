import re
import string
import tokenize
from Stemmer import Stemmer, algorithms as _stemmer_algorithms
from collections import deque
from functools import lru_cache

import functools

from plagiarism.stopwords import get_stop_words
from plagiarism.text import strip_punctuation

__all__ = [
    'split_to_words', 'stemmize', 'split_programming_tokens',
    'split_python_tokens'
]


KEEP_LETTERS_TABLE = {
    i: ' '
    for i in range(1, 256)
    if chr(i) not in string.ascii_letters
    }


@lru_cache(maxsize=50)
def get_stemmer(language=None):
    """
    Return stemmer for given language.

    The default language is english.
    """

    return Stemmer(language or 'english')


def split_to_words(text, casefold=True):
    """
    Convert text in a sequence of case-folded tokens.

    Remove punctuation and digits.
    """

    if casefold:
        text = text.casefold()
    text = text.translate(KEEP_LETTERS_TABLE)
    return text.split()


def stemmize(text, language=None, stop_words=None, ngrams=1):
    """
    Receive a string of text and return a list of stems.

    Args:
        text (str):
            A string of text to stemize.
        language:
            Default language for stemmer and stop words.
        stop_words (list):
            List of stop tokens.
        ngrams (int):
            If given, uses n-grams instead of tokens.
    """
    stemmer = get_stemmer(language)

    if stop_words is None:
        stop_words = get_stop_words(language)
    stop_stems = set(stemmer.stemWords(stop_words))
    words = text.casefold().split()
    words = stemmer.stemWords([strip_punctuation(word) for word in words])
    data = [w for w in words if w and w not in stop_stems]
    if ngrams == 1:
        return data
    else:
        result = []
        for i in range(len(data) - ngrams + 1):
            words = data[i:i + ngrams]
            result.append(' '.join(words))
        return result


def split_python_tokens(text, exclude=('ENCODING',)):
    """
    Uses python lexer to split source into a curated list of real python tokens.
    """

    import tokenize

    exclude = {getattr(tokenize, tok) if isinstance(tok, str) else tok
               for tok in exclude}
    lines = iter(text.encode('utf8').splitlines())
    tokens = tokenize.tokenize(lambda: next(lines))
    tokens = (tok.string.strip() for tok in tokens if tok.type not in exclude)
    return [tok for tok in tokens if tok]


def split_programming_tokens(text):
    """
    Split text in tokens that should work for most programming languages.
    """

    ws_regex = re.compile(r'[ \f\t]+')
    regexes = [
        # Whitespace
        ws_regex,

        # Single char symbols
        re.compile(r'[\n()[\]{\},;#]'),

        # Common comments
        re.compile(r'(//|/\*|\*/|\n\r)'),

        # Operators
        re.compile(r'[*+-/\^&|!]+'),

        # Strings
        re.compile(
            r'''('[^\n'\\]*(?:\\.[^\n'\\]*)*'|"[^\n"\\]*(?:\\.[^\n"\\]*)*")'''),
        re.compile(r"""('''[^'\\]*(?:\\.['\\]*)*'''|""" +
                   r'''"""[^"\\]*(?:\\.[^"\\]*)*""")'''),

        # Numbers
        re.compile(tokenize.Number),

        # Names and decorators
        re.compile(tokenize.Name),
        re.compile('@' + tokenize.Name),
    ]
    tokens = []
    lines = deque(text.splitlines())
    while lines:
        stream = lines.popleft()
        while stream:
            if stream[0].isspace():
                idx = ws_regex.match(stream).span()[1]
                stream = stream[idx:]
                continue

            pos = endpos = len(stream)
            for regex in regexes:
                match = regex.search(stream)
                if match is None:
                    continue
                idx, end = match.span()
                if end == 0:
                    continue
                elif idx == 0:
                    token, stream = stream[:idx], stream[idx:]
                    tokens.append(token)
                    break
                elif idx < pos or idx == pos and end > endpos:
                    pos, endpos = idx, end

            if pos == 0:
                tokens.append(stream[:endpos])
                stream = stream[endpos:]
            elif pos == endpos:
                tokens.append(stream)
                stream = ''
            else:
                tokens.append(stream[:pos])
                tokens.append(stream[pos:endpos])
                stream = stream[endpos:]

    return [tok for tok in tokens if tok and not tok.isspace()]


def tokenize_all(documents, tokenizer=None, **kwargs):
    """
    Tokenize a sequence of documents.

    Args:
        documents:
            List of documents
        tokenizer:
            Tokenizer function or function name.
        **kwargs:
            Extra arguments passed to the tokenizer function.
    """

    if not callable(tokenizer):
        tokenizer = tokenizer or 'words'
        tokenizer = TOKENIZER_DICT[tokenizer.replace('_', '-')]
    return [tokenizer(doc, **kwargs) for doc in documents]

TOKENIZER_DICT = {
    'words': split_to_words,
    'split-to-words': split_to_words,
    'python': split_python_tokens,
    'python-tokens': split_python_tokens,
    'split-python-tokens': split_python_tokens,
    'code': split_programming_tokens,
    'programming': split_programming_tokens,
    'programming-tokens': split_programming_tokens,
    'split-programming-tokens': split_programming_tokens,
    'stemmize': stemmize,
}

# Language support
for _lang in _stemmer_algorithms():
    _func = functools.partial(stemmize, language='language')
    TOKENIZER_DICT[_lang] = _func
