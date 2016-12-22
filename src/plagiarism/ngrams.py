import collections

from plagiarism.utils import count_all, count

__all__ = [
    'ngrams', 'ngrams_all', 'remove_ngram', 'remove_ngrams', 'merge_ngrams',
    'merge_ngrams_all', 'hierarchical_ngrams', 'load_ngrams', 'save_ngrams',
    'find_ngrams', 'find_bigrams',
]


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

    join_arg = join
    if join is None:
        def join(x):
            values = map(str, x)
            return sep.join(values)
    if accumulate:
        return _ngrams_acc(document, n, sep, join)
    if n == 2:
        if join_arg is tuple:
            return list(_bigrams(document))
        elif join_arg is None:
            return [x + sep + y for (x, y) in _bigrams(document)]
        else:
            return list(_bigrams(document, join))

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


# Faster implementation for bigrams
def _bigrams(document, join=None):
    it = iter(document)
    try:
        x = next(it)
    except StopIteration:
        return

    if join is None:
        for y in it:
            yield x, y
            x = y
    else:
        for y in it:
            yield join((x, y))
            x = y


def _ngrams_acc(document, n, sep=None, join=None):
    """
    Accumulate n-grams from 1 to n.
    """

    result = list(document)
    for i in range(2, n):
        result.extend(ngrams(document, i, sep, join))
    return result


def ngrams_all(documents, *args, **kwargs):
    """
    Like ngrams() function, but expects a list of documents.

    Accepts the same arguments as ngrams().
    """

    return [ngrams(doc, *args, **kwargs) for doc in documents]


def remove_ngram(document, ngram):
    """
    Return list with all occurrences of n-gram removed.

    Examples:
        >>> remove_ngram(['foo', 'bar', 'ham', 'spam'], ('bar', 'ham'))
        ['foo', 'spam']
    """

    n = len(ngram)
    result = []
    iterations = len(document) - n
    idx = 0
    while idx <= iterations:
        for j, word in enumerate(ngram):
            if word != document[idx + j]:
                result.append(document[idx])
                break
        else:
            idx += n - 1
        idx += 1
    result.extend(document[idx:])
    return result


def remove_ngrams(document, ngrams):
    """
    Like :func:`remove_ngram`, but remove all ngrams from list.
    """

    join = lambda x: x
    ngrams = hierarchical_ngrams(ngrams)
    return _merge_ngrams_worker(document, ngrams, join, True)


def merge_ngrams(document, ngrams, join=None):
    """
    Given a list of tokens and a list of valid n-grams, return a new version
    of the document in which every possible ngram in the list is merged.

    Args:
        document:
        ngrams:

    Returns:

    """

    join = join or (lambda x: ' '.join(x))
    ngrams = hierarchical_ngrams(ngrams)
    return _merge_ngrams_worker(document, ngrams, join)


def _merge_ngrams_worker(document, ngrams, join, remove=False):
    pending = collections.deque(document)
    done = []
    while pending:
        ngram = []
        node = ngrams
        while pending and pending[0] in node:
            word = pending.popleft()
            ngram.append(word)
            node = node[word]
        if ... not in node:
            pending.extendleft(reversed(ngram))
            ngram = []

        if ngram:
            if not remove:
                done.append(join(ngram))
        else:
            done.append(pending.popleft())
    return done


def merge_ngrams_all(documents, ngrams, join=None):
    """
    Apply :func:`merge_ngrams` to all documents in the list.
    """

    join = join or (lambda x: ' '.join(x))
    ngrams = hierarchical_ngrams(ngrams)
    return [_merge_ngrams_worker(doc, ngrams, join) for doc in documents]


def hierarchical_ngrams(ngrams):
    """
    Transform a list of ngrams in a three mapping each sequential word to the
    ngrams related to that word. Each n-gram is saved in the '...' node of
    the tree.

    This structure makes it easier to localize possible ngrams candidates from
    a sequence of words.

    Args:
        ngrams: list of ngram tuples

    Returns:
        An hierarchical tree made from dictionaries.

    Examples:
        Consider the list

        >>> ngrams = [('foo', 'bar'), ('foo', 'bar', 'baz'), ('ham', 'spam')]

        This function produces a tree equivalent to the following::

            {
                'foo': {
                    'bar': {
                        ...: ('foo', 'bar'),
                        'baz': {
                            ...: ('foo', 'bar', baz'),
                        },
                    }
                },
                'ham': {
                    'spam': {
                        ...: ('ham', 'spam'),
                    },
                },
            }

    """
    tree = {}
    for ngram in ngrams:
        node = tree
        for word in ngram:
            try:
                node = node[word]
            except KeyError:
                node[word] = new = {}
                node = new
        node[...] = ngram
    return tree


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


def find_bigrams(document, min_repetitions=2):
    """
    Finds all bi-grams

    Args:
        document: a list of words/tokens.

    Returns:
        A list of bigrams generated from document.
    """

    accepted_bigrams = []
    new_bigrams = True

    while new_bigrams:
        new_bigrams = []
        bigrams = ngrams(document, 2, join=tuple)
        w1_count = collections.Counter()
        w2_count = collections.Counter()
        for bigram in bigrams:
            w1, w2 = bigram
            w1_count[w1] += 1
            w2_count[w2] += 1

        used_first_words = set()
        used_second_words = set()

        data = (
            (N, w1_count[w1] - N, w2_count[w2] - N, w1, w2)
            for (w1, w2), N in count(bigrams).items() if N >= min_repetitions
        )

        def key(x):
            return -x[0], min(x[1], x[2]), max(x[1], x[2]), x[3], x[4]

        data = sorted(data, key=key)
        for N, neg_w1, neg_w2, w1, w2 in data:
            if w1 in used_second_words or w2 in used_first_words:
                continue

            if N > neg_w1 or N > neg_w2:
                new_bigrams.append((w1, w2))
                used_first_words.add(w1)
                used_second_words.add(w2)

        document = remove_ngrams(document, new_bigrams)
        accepted_bigrams.extend(new_bigrams)

    return accepted_bigrams


def find_ngrams(document, verbose=False, min_repetitions=2, max_iter=None):
    """
    Compute a list of useful ngrams for document.
    """

    bigrams = True
    iter = 0
    max_iter = float('inf') if max_iter is None else max_iter
    if verbose:
        print('finding bigrams in document with %s words' % len(document))

    while bigrams:
        iter += 1
        if iter > max_iter:
            break
        bigrams = find_bigrams(document, min_repetitions=min_repetitions)
        document = merge_ngrams(document, bigrams)
        if verbose:
            fmt = iter, len(bigrams), len(document)
            print('iter: %3d, %5d new bigrams, %7d tokens in document.' % fmt)
    ngrams = {tuple(x.split()) for x in document if ' ' in x}
    return ngrams


def save_ngrams(ngrams, file):
    """
    Save ngrams in the given file.

    Args:
        ngrams: list of ngrams
        file: file name
    """
    with open(file, 'w') as F:
        for ngram in ngrams:
            F.write(' '.join(ngram))
            F.write('\n')


def load_ngrams(file):
    """
    Load ngrams saved with save_ngrams() from the given file.
    """
    ngrams = []
    with open(file) as F:
        for line in F:
            ngrams.append(line.split())
    return ngrams
