import collections
from collections import Counter

import numpy as np
from math import log

from plagiarism import compute_weights
from plagiarism.math_utils import similarity
from plagiarism.tokenizers import stemmize
from plagiarism.utils import count_all, sorted_tokens_all
from plagiarism.weigths import apply_weights

__all__ = [
    'bag_of_words', 'bag_of_documents', 'vectorize', 'unvectorize',
    'similarity_matrix', 'SimilarPair', 'most_similar', 'common_tokens_all',
]


def bag_of_words(document, method='boolean', weights=None, func=None,
                 tokenizer=None, **kwargs):
    """
    Convert a text to a Counter object.

    Args:
        document:
            Can be a string of text or a list of stems. If data is a string, it
            will be converted to a list of stems using the given tokenizer
            function.
        method:
            'boolean', 'bool' (default):
                Existing tokens receive a value of 1.
            'frequency', 'freq':
                Weight corresponds to the relative frequency of each tokens
            'count':
                Weight corresponds to the number of times the word appears on
                text.
        func (callable):
            If given, a function to apply to each result.
        weights:
            Weights given to each component of the bag of words. If must be
            a dictionary and if token is not present in dictionary, the weight
            is implicitly equal to 1. Pass a collections.defaultdict object if
            you need a different default.
        tokenizer:
            The function used to tokenize string inputs.
    """

    if isinstance(document, str):
        tokenizer = tokenizer or stemmize
        document = tokenizer(document, **kwargs)
    count = Counter(document)

    if method in ('boolean', 'bool'):
        result = Counter({stem: 1 for stem in count})
    elif method in ('frequency', 'freq'):
        total = sum(count.values())
        result = Counter({stem: n / total for (stem, n) in count.items()})
    elif method in ('log-frequency', 'log-freq'):
        total = sum(count.values())
        result = Counter({stem: log(n / total) for (stem, n) in count.items()})
    elif method == 'count':
        result = count
    else:
        raise ValueError('invalid method: %r' % method)

    if func:
        for k, v in result.items():
            result[k] = func(v)
    if weights:
        result = apply_weights(result, weights)
    return result


def bag_of_documents(documents, method='weighted', weights=None, **kwargs):
    """
    Apply bag of words to a set of documents.

    Args:
        documents:
            A list of documents
        method:
            'weighted' (default):
                Inverse frequency weighting method. Each coordinate is
                proportional to the frequency of words weighted by a factor of
                log(N/doc_freq) where N is the number of documents and doc_freq
                is the number of documents that contains the word.
            + all methods accepted by :func:`bag_of_words`.
    """

    if weights and isinstance(weights, str):
        weights = compute_weights(documents, method=weights)

    if 'weighted' in method:
        if weights:
            raise ValueError('cannot specify weights with method %r' % method)
        method, _, _ = method.rpartition('-')
        method = method or 'frequency'
        weights = compute_weights(documents, method='log-df')

    if weights:
        kwargs['weights'] = weights
    return [bag_of_words(doc, method=method, **kwargs) for doc in documents]


def vectorize(bags, default=0.0, tokens=None):
    """
    Convert a list of bag of words to matrix form.

    Return:
        tokens:
            A list of tokens mapping to their respective indexes.
        default:
            Default value to assign to a token that does not exist on a
            document.
        matrix:
            A matrix representing the full bag of documents.
    """

    tokens = tokens or sorted_tokens_all(bags)
    data = [[bag.get(tk, default) for tk in tokens] for bag in bags]
    return np.array(data)


def unvectorize(matrix, tokens, single=False):
    """
    Revert the effect of the vectorize function.

    Args:
        matrix:
            A matrix of vectorized elements.
        tokens:
            A sorted list of tokens. Usually obtained by calling
            :func:`sorted_tokens_all(bag_of_documents)`.
        single (bool):
            True to unvectorize a single vector (default is False).

    Returns:
        A list of bags.
    """

    if single:
        matrix = [matrix]
    result = []
    for vec in matrix:
        bag = collections.Counter(dict(zip(tokens, vec)))
        result.append(bag)
    if single:
        return result[0]
    return result


def similarity_matrix(matrix, method='triangular', diag=None, norm=None):
    """
    Return the similarity matrix from a matrix.

    Args:
        matrix:
            A matrix created by vectorizing all elements.
        method:
            Method used to compute the similarity between two vectors. See
            the :func:`plagiarism.similarity` function.
        norm:
            Norm used by the similarity function.
        diag:
            The value of the diagonal, similarity of an element with itself.
    """

    size = len(matrix)
    if not isinstance(matrix, np.ndarray):
        matrix = vectorize(matrix)
    result = np.zeros([size, size], dtype=float)
    for i in range(size):
        vi = matrix[i]
        for j in range(i + 1, size):
            vj = matrix[j]
            value = similarity(vi, vj, method, norm=norm)
            result[i, j] = result[j, i] = value
    for i in range(size):
        if diag is None:
            u = matrix[i, i]
            result[i, i] = similarity(u, u, method, norm=norm)
        else:
            result[i, i] = diag
    return result


class SimilarPair(tuple):
    """
    Result of most_similar() function.
    """

    def __new__(cls, a, b, similarity=None, indexes=None):
        new = tuple.__new__(cls, (a, b))
        new.similarity = similarity
        new.indexes = indexes
        return new

    def __init__(self, a, b, similarity=None, indexes=None):
        super().__init__()


def most_similar(documents, similarity=None, n=None):
    """
    Retrieve the n most similar documents ordered by similarity.

    Args:
        documents:
            List of documents.
        similarity:
            Similarity matrix.
        n (int, optional):
            If given, corresponds to the maximum number of elements returned.

    Returns:
        A list of (doc[i], doc[j]) pairs. Each pair is an instance of a tuple
        subclass that also have the attributes .similarity (with the similarity
        value) and .indexes (with a tuple of (i, j) indexes for the pair).
    """

    result = []
    size = len(documents)
    for i in range(size):
        for j in range(i + 1, size):
            item = (similarity[i, j], (i, j))
            result.append(item)
    result.sort(reverse=True)
    result = [idx for sim, idx in result]
    if n:
        n = max(len(result), n)
        result = result[:n]

    docs = documents
    pair = SimilarPair
    return [pair(docs[i], docs[j], similarity[i, j], (i, j)) for i, j in result]


def common_tokens_all(documents, n=None, by_document=False):
    """
    Return a list of (token, relative frequency) pairs for the the n-th most
    common tokens.
    """

    counter = Counter()
    if by_document:
        size = len(documents)
        for doc in documents:
            for w in set(doc):
                counter[w] += 1
        common = counter.most_common(n)
        return [(word, n / size) for (word, n) in common]
    else:
        for doc in documents:
            for w in doc:
                counter[w] += 1
        total = sum(counter.values())
        common = counter.most_common(n)
        return [(word, count / total) for (word, count) in common]
