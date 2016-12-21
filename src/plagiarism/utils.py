import collections
import contextlib
import time
from math import log


def count_all(documents, method='total'):
    """
    Return a counter considering all words in all documents.

    Args:
        documents:
            List of documents. Each document is a sequence of words.
        method:
            'total' (default):
                Counts the total number of times each word appears in the
                document
            'doc':
                Counts the number of documents in which the word appears.
            'doc-freq':
                Counts the fraction of documents in which the word appears.
            'log-doc-freq':
                The weight of each token is computed as log(N / n) in which N
                is the total number of documents and n is the number of
                documents with the given token.
    """

    if method == 'log-doc-freq':
        counter = count_all(documents, 'doc-freq')
        counter.update({tok: -log(f) for tok, f in counter.items()})
        return counter

    counter = collections.Counter()
    size = len(documents)
    for doc in documents:
        if method in ['total']:
            counter.update(count(doc, method))
        elif method == 'doc':
            used_words = set()
            for word in doc:
                if word not in used_words:
                    counter[word] += 1
                    used_words.add(word)
        elif method == 'doc-freq':
            used_words = set()
            for word in doc:
                if word not in used_words:
                    counter[word] += 1 / size
                    used_words.add(word)
        else:
            raise ValueError('invalid method: %r' % method)
    return counter


def count(document: collections.Sequence, method='total'):
    """
    Count words in document.
    Args:
        document:
            A sequence of words/tokens.
        method:
            'total':
                Counts the total number of each word.
            'freq':
                Relative frequency of words.

    Returns:
        A collections.Counter object.

    """

    if method == 'total':
        return collections.Counter(document)
    elif method == 'freq':
        counter = collections.Counter()
        inv_size = 1 / len(document)
        for word in document:
            counter[word] += inv_size
        return counter
    else:
        raise ValueError('invalid method: %r' % method)


def tokens_all(documents):
    """
    Return a list of tokens from all documents.
    """

    tokens = set()
    for doc in documents:
        tokens.update(doc)
    return sorted(tokens)


class Instant:
    """
    Fake number-like object. Used at timeit() context manager.
    """
    @property
    def value(self):
        if self._tf is None:
            return time.time() - self._t0
        return self._tf - self._t0

    @value.setter
    def value(self, value):
        self._tf = self._t0 + value

    def __init__(self, t0=None, tf=None):
        self._t0 = t0 or time.time()
        self._tf = tf

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return 'Instant(%s)' % self.value

    def __str__(self):
        return str(self.value)

    def __rsub__(self, other):
        return other - self.value

    def __sub__(self, other):
        return self.value - other

    def __cmp__(self, other, *args):
        return self.value.__cmp__(other, *args)

    @contextlib.contextmanager
    def update(self):
        """
        Add elapsed time inside with block to interval object.
        """

        t0 = time.time()
        try:
            yield self
        finally:
            dt = time.time() - t0
            self.value += dt


@contextlib.contextmanager
def timeit():
    """
    Context manager used to time execution inside a with block.

    Example:
        >>> with timeit() as dt:
        ...     time.sleep(0.5)
        >>> print('%.1f secs' % dt)
        0.5 secs
    """

    dt = Instant()
    try:
        yield dt
    finally:
        dt._tf = time.time()