from collections import Counter

import numpy as np

from plagiarism.math_utils import similarity
from plagiarism.tokenizers import stemmize
from plagiarism.utils import count_all, tokens_all


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
            Weighting factor used in as the values of the Counter object.

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


def vectorize(bag, default=0.0, tokens=None):
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

    tokens = tokens or tokens_all(bag)
    data = [[doc.get(tok, default) for tok in tokens] for doc in bag]
    return np.array(data)


def similarity_matrix(data, method='triangular', diag=1.0, norm=None):
    """
    Return the similarity matrix from a matrix.
    """

    size = len(data)
    if not isinstance(data, np.ndarray):
        data = vectorize(data)
    result = np.zeros([size, size], dtype=float) + diag
    for i in range(size):
        vi = data[i]
        for j in range(i + 1, size):
            vj = data[j]
            value = similarity(vi, vj, method, norm=norm)
            result[i, j] = result[j, i] = value
    return result


class SimilarPair(tuple):
    """
    Result of most_similar() function.
    """

    def __new__(cls, a, b, similarity=None, indexes=None):
        new = tuple.__new__(cls, (a, b))
        return new

    def __init__(self, a, b, similarity=None, indexes=None):
        self.similarity = similarity
        self.indexes = indexes


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
        A list of (doc[i], doc[j]) pairs.
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
