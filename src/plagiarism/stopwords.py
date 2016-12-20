from functools import lru_cache
from stop_words import LANGUAGE_MAPPING

SUPPORTED_LANGUAGES = list(LANGUAGE_MAPPING) + list(LANGUAGE_MAPPING.values())


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

