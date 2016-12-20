import collections

from plagiarism.utils import count_all


def remove_ngram(words, ngram):
    """
    Return list with all occurrences of n-gram removed.

    Examples:
        >>> remove_ngram(['foo', 'bar', 'ham', 'spam'], ('bar', 'ham'))
        ['foo', 'spam']
    """

    n = len(ngram)
    result = []
    iterations = len(words) - n
    idx = 0
    while idx <= iterations:
        for j, word in enumerate(ngram):
            if word != words[idx + j]:
                result.append(words[idx])
                break
        else:
            idx += n - 1
        idx += 1
    result.extend(words[idx:])
    return result


def optimal_bigrams(documents, iter=1, **kwargs):
    """
    Create a list of bi-grams for all documents and select which bi-grams are
    more appropriate to stay.

    Args:
        documents:
            A list of documents. Each document is a sequence of tokens.
        iter (int):
            Number of iterations. This function makes successive reductions
            until no change is made to the document list.
        min_freq (int, default=2):
            Minimum frequency required to form a bi-gram.
        accumulate (bool):
            If True, accumulate lower order n-grams.
        sep (str):
            String separator for joining to words in a bi-gram.
        join (callable):
            Function used to join a tuple of words into a bi-gram. If you
            want to preserve bi-gram as a tuple, use ``join=tuple``.
        predictable:
            If True (default), take precautions to make the optimal list to be
            predicable over different runs.
        allow_superposition:
            If True, allows superposition of bi-gram components.
    """

    if iter > 1:
        try:
            result = optimal_bigrams(documents, **kwargs)
        except StopIteration:
            return [list(doc) for doc in documents]
        else:
            return optimal_bigrams(result, iter - 1, **kwargs)

    # Parameters
    sep = kwargs.get('sep', ' ')
    join = kwargs.get('join', None)
    if join is None:
        def join(x):
            return sep.join(x)
    accumulate = kwargs.get('accumulate', False)
    predictable = kwargs.get('predictable', True)
    allow_superposition = kwargs.get('allow_superposition', False)

    # Select bi-grams
    bigrams = ngrams_all(documents, 2, join=tuple)
    bi_counter = count_all(bigrams)
    uni_counter = count_all(documents)
    if not uni_counter:
        raise ValueError('documents are empty: %r' % documents)

    # Get minimum acceptable frequency: we pick the frequency threshold that we
    # could expect that the most common word would appear in a pair by pure
    # random chance
    size = sum(uni_counter.values())
    freq = max(uni_counter.values()) / size
    min_freq = size * freq ** 2
    min_freq = max(min_freq, kwargs.get('min_freq', 2))
    data = {pair: n for pair, n in bi_counter.items() if n >= min_freq}
    bi_counter = collections.Counter()
    bi_counter.update(data)

    # Return early if no bi-grams are found
    if not bi_counter:
        if iter == 1:
            return [list(doc) for doc in documents]
        raise StopIteration

    # Iterate the list of most frequent pairs skipping possible collisions
    # Return early if bi_counter has no elements
    has_skip = True
    result = [list(doc) for doc in documents]
    while has_skip:
        candidates = []
        words_in_bigrams = set()
        has_skip = False
        most_common = bi_counter.most_common()
        if predictable:
            most_common.sort(key=lambda x: x[::-1])
        for (w1, w2), n in most_common:
            if w1 in words_in_bigrams or w2 in words_in_bigrams:
                has_skip = True
                if not allow_superposition:
                    continue
            words_in_bigrams.add(w1)
            words_in_bigrams.add(w2)
            candidates.append((w1, w2))
        candidates = set(candidates)

        # Remove used pairs
        for pair in candidates:
            del bi_counter[pair]

        # Recalculate list of bi-grams and uni-grams
        new_result = []
        for doc in result:
            new_doc = []
            extra = []
            last_idx = len(doc) - 1
            skip = False
            for i, word in enumerate(doc):
                if skip:
                    skip = False
                    continue
                elif i == last_idx:
                    new_doc.append(word)
                    break
                elif word in words_in_bigrams:
                    pair = (word, doc[i + 1])
                    if pair in candidates:
                        if accumulate:
                            extra.extend(pair)
                        new_doc.append(join(pair))
                        skip = True
                    else:
                        new_doc.append(word)
                else:
                    new_doc.append(word)
            new_doc.extend(extra)
            new_result.append(new_doc)
        result = new_result

    return result


def ngrams(document, n, sep=' ', join=None, accumulate=False):
    """
    Create list of n-grams from given sequence of words.

    Args:
        words:
            A document represented by a sequence of tokens.
        sep:
            String separator to join components of n-gram.
        join:
            A function that receives a tuple and return a n-gram object. If
            you want to represent n-grams by tuples, pass ``join=tuple``.
        accumulate:
            If True, all n-grams lists up to the given n.

    Example:
        >>> ngrams(['to', 'be', 'or', 'not', 'to', 'be'], 3, sep=' ')
        ['to be or', 'be or not', 'or not to', 'not to be']
    """

    if join is None:
        def join(x):
            values = map(str, x)
            return sep.join(values)
    if accumulate:
        return _ngrams_acc(document, n, sep, join)

    document = list(document)
    size = len(document)

    if n == size:
        return [join(document)]
    elif n > size:
        return []

    result = []
    for i in range(0, size - n + 1):
        result.append(join(document[i:i + n]))
    return result


def _ngrams_acc(words, n, sep=None, join=None):
    """
    Accumulate n-grams from 1 to n.
    """

    result = list(words)
    for i in range(2, n):
        result.extend(ngrams(words, i, sep, join))
    return result


def ngrams_all(documents, *args, **kwargs):
    """
    Like ngrams() function, but expects a list of documents.

    Accepts the same arguments as ngrams().
    """

    return [ngrams(doc, *args, **kwargs) for doc in documents]


