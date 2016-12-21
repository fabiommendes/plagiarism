from collections import Counter

import numpy as np

from plagiarism.math_utils import similarity
from plagiarism.tokenizers import stemmize
from plagiarism.utils import count_all, sorted_tokens_all


def apply_weights(counter, weights, default=1):
    """
    Apply weights on counter object.

    Args:
        counter:
            A map between tokens to frequencies.
        weights:
            A map of tokens to their respective weights.
        default:
            Default weight.
    """
    return Counter({stem: weights.get(stem, default) * freq
                    for (stem, freq) in counter.items()})


def bag_of_words(document, method='boolean', weights=None,
                 tokenizer=None, **kwargs):
    """
    Convert a text to a Counter object.

    Args:
        document:
            Can be a string of text or a list of stems. If data is a string, it
            will be converted to a list of stems using the given tokenizer
            function.
        method:
            'boolean' (default):
                Existing tokens receive a value of 1.
            'frequency':
                Weight corresponds to the relative frequency of each tokens
            'count':
                Weight corresponds to the number of times the word appears on
                text.
            'weighted':
                Inverse frequency weighting method.
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

    if method == 'boolean':
        return Counter({stem: 1 for stem in count})
    elif method == 'frequency':
        total = sum(count.values())
        return Counter({stem: n / total for (stem, n) in count.items()})
    elif method == 'count':
        return count
    elif method == 'weighted':
        counter = bag_of_words(document, 'frequency')
        return apply_weights(counter, weights)
    else:
        raise ValueError('invalid method: %r' % method)


def bag_of_documents(documents, method='weighted', **kwargs):
    """
    Apply bag of words to a set of documents.

    Args:
        documents:
            A list of documents
        method:
            'weighted' (default):
                Inverse frequency weighting method. Each coordinate is
                proportional to the frequency of words weighted by a factor of
                log(N/doc_frec) where N is the number of documents and doc_freq
                is the number of documents that contains the word.
            'boolean':
                Existing tokens receive a value of 1.
            'frequency':
                Weight corresponds to the relative frequency of each tokens
            'count':
                Weight corresponds to the number of times the word appears on
                text.
    """

    if method != 'weighted':
        return [bag_of_words(doc, method=method, **kwargs) for doc in documents]

    bag = bag_of_documents(documents, method='frequency', **kwargs)
    weights = count_all(documents, method='log-doc-freq')
    result = []
    for counter in bag:
        new = apply_weights(counter, weights)
        result.append(new)
    return result


def vectorize(bag_of_documents, default=0.0, tokens=None):
    """
    Convert bag of documents to matrix.

    Return:
        tokens:
            A list of tokens mapping to their respective indexes.
        default:
            Default value to assign to a token that does not exist on a
            document.
        matrix:
            A matrix representing the full bag of documents.
    """

    tokens = tokens or sorted_tokens_all(bag_of_documents)
    data = [[doc.get(tk, default) for tk in tokens] for doc in bag_of_documents]
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
        bag = dict(zip(tokens, vec))
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
    Return a list of (token, frequency) pairs for the the n-th most common
    tokens.
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
