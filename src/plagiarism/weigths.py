import collections
from collections import Counter
from math import log


def apply_weights(bag_of_words, weights, default=1):
    """
    Apply weights on counter object.

    Args:
        bag_of_words:
            A map between tokens to values.
        weights:
            A map of tokens to their respective weights.
        default:
            Default weight.
    """
    return Counter({stem: weights.get(stem, default) * freq
                    for (stem, freq) in bag_of_words.items()})


def compute_weights(documents, method=None, **kwargs):
    """
    Compute the weights of each token according to the given method.

    Args:
        documents:
            A list of documents.
        method:
            Let df = "document frequency" be the number of documents that
            contain a token and tf = "token frequency" the number of times the
            token appears in all documents.

            'df':
                Weights are proportional to func(df, N_docs), where func must
                be  provided as a keyword argument.
            'tf':
                Weights are proportional to func(tf,  N_tk), where func must be
                provided as a keyword argument.
            'log-df':
                Weights each token as log(N_docs / df)
            'inv-df:
                Weights each token as N_docs / df
            'log-tf':
                Weights each token as log(N_tk / tf)
            'inv-tf:
                Weights each token as N_tk / tf

        **kwargs:
            Additional keyword parameters passed to the chosen method.

    Returns:
        A counter object.
    """

    try:
        method = METHODS[method]
    except KeyError:
        raise ValueError('invalid method: %r' % method)
    return method(documents, **kwargs)


def token_frequency(documents):
    """
    Return a counter with the number of times each token appears.
    """

    counter = collections.Counter()
    for word in tokens_set(documents):
        for doc in documents:
            counter[word] += doc.count(word)
    return counter


def document_frequency(documents):
    """
    Return a counter with the number of documents each token appears.
    """

    counter = collections.Counter()
    for word in tokens_set(documents):
        for doc in documents:
            counter[word] += word in doc
    return counter


def tokens_set(documents):
    """
    Return a set with all tokens found in all documents.
    """

    words = set()
    for doc in documents:
        words.update(doc)
    return words


def log_tf_weights(documents):
    return tf_weights(documents, func=lambda f, N: log(N / f))


def log_df_weights(documents):
    return df_weights(documents, func=lambda f, N: log(N / f))


def inv_tf_weights(documents):
    return tf_weights(documents, func=lambda f, N: N / f)


def inv_df_weights(documents):
    return df_weights(documents, func=lambda f, N: N / f)


def _weights(weights, N, func):
    for tk, f in weights.items():
        weights[tk] = func(f, N)
    return weights


def tf_weights(documents, func=None):
    _assure_func(func)
    weights = document_frequency(documents)
    N = sum(weights.values())
    return _weights(weights, N, func)


def df_weights(documents, func=None):
    _assure_func(func)
    weights = document_frequency(documents)
    N = len(documents)
    return _weights(weights, N, func)


def _assure_func(func):
    if func is None:
        raise TypeError('"func(freq, N)" keyword argument must be provided!')


METHODS = {
    None: log_df_weights,
    'log-df': log_df_weights,
    'inv-df': inv_df_weights,
    'log-tf': log_tf_weights,
    'inv-tf': inv_tf_weights,
    'df': df_weights,
    'tf': tf_weights,
}
