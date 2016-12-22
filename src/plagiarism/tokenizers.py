import re
import string
import tokenize as _tokenize
from Stemmer import Stemmer, algorithms as _stemmer_algorithms
from collections import deque
from functools import lru_cache

import functools

from plagiarism.stopwords import get_stop_words
from plagiarism.text import strip_punctuation

__all__ = [
    'split_words', 'stemmize', 'split_programming_tokens',
    'split_python_tokens', 'tokenize', 'tokenize_all',
]

PUNCTUATION_TABLE = str.maketrans({
    c: ' ' for c in string.punctuation
})
STOP_SIGNS_TABLE = str.maketrans({
    c: '.' for c in '.!?'
})
PUNCTUATION_TABLE_NO_STOP_SIGN = str.maketrans({
    c: ' ' for c in string.punctuation.replace('.', '')
})


def tokenize(text, tokenizer=None, **kwargs):
    """
    Common interface to all tokenizers.

    Args:
        text:
            Text to tokenize.
        tokenizer:
            A tokenizer function or function name.
        **kwargs:
            Extra arguments passed to the tokenizer.

    Valid tokenize methods are:
        'words': :func:`split_to_words`
        'python': :func:`split_python_tokens`
        'code': :func:`split_programming_tokens`
        'stems': :func:`stemmize`
    """
    if not callable(tokenizer):
        tokenizer = tokenizer or 'words'
        tokenizer = TOKENIZER_DICT[tokenizer.replace('_', '-')]
    return tokenizer(text, **kwargs)


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

    See also:
        :func:`tokenize`.
    """

    if not callable(tokenizer):
        tokenizer = tokenizer or 'words'
        tokenizer = TOKENIZER_DICT[tokenizer.replace('_', '-')]
    return [tokenizer(doc, **kwargs) for doc in documents]


def stemmize(text, language=None, stop_words=None):
    """
    Receive a string of text and return a list of stems.

    Args:
        text (str):
            A string of text to stemmize.
        language:
            Default language for stemmer and stop words.
        stop_words:
            A list of stop words. Set to False if you don't want to include the
            default list of stop words for the given language.
    """

    stemmer = get_stemmer(language)
    words = split_words(text)
    words = stemmer.stemWords(words)
    if stop_words is False:
        return words
    if stop_words is None:
        stop_words = get_stop_words(language)
    else:
        stop_words = get_stop_words(stop_words)
    stop_stems = set(stemmer.stemWords(stop_words))
    return [word for word in words if word not in stop_stems]


def split_words(text, casefold=True, keep_stops=False):
    """
    Convert text in a sequence of case-folded tokens.

    Remove punctuation and digits.
    """

    if casefold:
        text = text.casefold()

    if keep_stops:
        text = text.translate(STOP_SIGNS_TABLE)
        text = text.replace('.', ' . ')
        text = text.translate(PUNCTUATION_TABLE_NO_STOP_SIGN)
    else:
        text = text.translate(PUNCTUATION_TABLE)
    return text.split()


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
        re.compile(_tokenize.Number),

        # Names and decorators
        re.compile(_tokenize.Name),
        re.compile('@' + _tokenize.Name),
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


@lru_cache(maxsize=50)
def get_stemmer(language=None):
    """
    Return stemmer for given language.

    The default language is english.
    """

    return Stemmer(language or 'english')

TOKENIZER_DICT = {
    'words': split_words,
    'split-words': split_words,
    'python': split_python_tokens,
    'python-tokens': split_python_tokens,
    'split-python-tokens': split_python_tokens,
    'code': split_programming_tokens,
    'programming': split_programming_tokens,
    'programming-tokens': split_programming_tokens,
    'split-programming-tokens': split_programming_tokens,
    'stemmize': stemmize,
    'stems': stemmize,
}

# Language support
for _lang in _stemmer_algorithms():
    _func = functools.partial(stemmize, language='language')
    TOKENIZER_DICT[_lang] = _func
