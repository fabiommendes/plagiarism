from functools import lru_cache

from stop_words import LANGUAGE_MAPPING

SUPPORTED_LANGUAGES = list(LANGUAGE_MAPPING) + list(LANGUAGE_MAPPING.values())

__all__ = [
    'get_stop_words', 'remove_stop_words', 'remove_stop_words_all',
    'remove_unique_tokens'
]


@lru_cache(50)
def get_stop_words(ref=None):
    """
    Return a list of stop split_to_words for the given database.
    """

    if ref is None:
        return get_stop_words('english')
    elif not isinstance(ref, str):
        return list(ref)

    if ref in ('python', 'c', 'c++', 'js'):
        return []
    elif ref in SUPPORTED_LANGUAGES:
        from stop_words import get_stop_words as getter
        return getter(ref)
    else:
        raise ValueError('could not understand %r' % ref)


def remove_stop_words(document, stop_words='english'):
    """
    Remove all stop words in the set from the given document.

    Args:
        document:
            a sequence of words/tokens.
        stop_words:
            a list of stop words or a name for the stop words set (e.g.:
            'portuguese')

    Returns:
        A list of words with the stop words removed.
    """

    stop_words = _stop_words_set(stop_words)
    return _remove_stop_words(document, stop_words)


def remove_stop_words_all(documents, stop_words='english'):
    """
    Apply :func:`remove_stop_words` in all documents in the list of documents.
    """

    stop_words = _stop_words_set(stop_words)
    return [_remove_stop_words(doc, stop_words) for doc in documents]


def _remove_stop_words(document, stop_words_set):
    return [tk for tk in document if tk not in stop_words_set]


def _stop_words_set(stop_words):
    if isinstance(stop_words, str):
        stop_words = get_stop_words(stop_words)
    return set(stop_words)


def remove_unique_tokens(documents):
    """
    Return a copy of documents with all tokens that appear only in a single
    document removed.
    """

    documents = [list(doc) for doc in documents]
    whitelist = set()
    blacklist = {}

    for doc in documents:
        tokens = set(doc)
        for tk in tokens:
            if tk in whitelist:
                continue
            if tk in blacklist:
                blacklist.pop(tk)
                whitelist.add(tk)
            blacklist[tk] = doc

    for tk, doc in blacklist.items():
        doc[:] = [x for x in doc if x != tk]

    return documents
