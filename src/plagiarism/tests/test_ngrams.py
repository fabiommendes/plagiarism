from plagiarism.bag_of_words import count_all
from plagiarism.ngrams import optimal_bigrams, ngrams, remove_ngram, \
    merge_ngrams, remove_ngrams


def test_ngram_simple():
    assert ngrams(['foo', 'bar', 'baz'], 2, sep=' ') == \
           ['foo bar', 'bar baz']
    assert ngrams(['foo', 'bar', 'baz'], 2, join=tuple) == \
           [('foo', 'bar'), ('bar', 'baz')]
    assert ngrams(['foo', 'bar', 'baz'], 2, join=lambda x: ''.join(x)) == \
           [('foobar'), ('barbaz')]


def test_ngrams_acc():
    assert ngrams(['a', 'b', 'c'], 4, accumulate=True) == \
           ['a', 'b', 'c', 'a b', 'b c', 'a b c']


def test_fibo_unigrams(fibo_doc_tokens):
    docs = fibo_doc_tokens
    counter = count_all(docs)
    common = counter.most_common(10)
    common = sorted([x for x, y in common if y == common[0][1]])
    assert common == ', = x'.split()


def test_fibo_bigrams(fibo_doc_tokens):
    docs = fibo_doc_tokens
    counter = count_all([ngrams(doc, 2, sep=' ') for doc in docs])
    common = counter.most_common(10)
    common = sorted([x for x, y in common if y == common[0][1]])
    assert common == ['( n', ') :', 'n )']


def test_fibo_reduced_bigrams(fibo_doc_tokens):
    docs = fibo_doc_tokens
    bi_docs = optimal_bigrams(docs, 10, min_freq=2, sep=' ')

    for bi_doc, doc in zip(bi_docs, docs):
        assert ' '.join(bi_doc) == ' '.join(doc)

    assert 'def fibo ( n ) :' in count_all(bi_docs)


def test_remove_ngram():
    assert remove_ngram(['foo', 'bar', 'ham', 'spam'], ('bar', 'ham')) \
           == ['foo', 'spam']


def test_remove_ngrams():
    assert remove_ngrams(['foo', 'bar', 'ham', 'spam', 'eggs'],
                         [('bar', 'ham'), ('spam', 'eggs'), ('ham', 'spam')]) \
           == ['foo']


def test_merge():
    ngrams = [('foo', 'bar'), ('foo', 'bar', 'baz'), ('ham', 'spam')]
    phrase = 'from foo bar baz got foo bar eating ham spam eggs'
    words = phrase.split()
    assert merge_ngrams(words, ngrams) == [
        'from', 'foo bar baz', 'got', 'foo bar', 'eating', 'ham spam', 'eggs'
    ]


def test_merge_ngrams():
    words = list('abcde')
    ngrams = [('b', 'c', 'e'), ('c', 'd')]
    assert merge_ngrams(words, ngrams) == ['a', 'b', 'c d', 'e']

    words = list('abcde')
    ngrams = [('b', 'c', 'e'), ('b', 'c'), ('c', 'd')]
    assert merge_ngrams(words, ngrams) == ['a', 'b c', 'd', 'e']