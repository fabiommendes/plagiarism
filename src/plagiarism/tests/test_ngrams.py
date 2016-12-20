from plagiarism.bag_of_words import count_all
from plagiarism.ngrams import optimal_bigrams, ngrams, remove_ngram


def test_ngram_simple():
    assert ngrams(['foo', 'bar', 'baz'], 2, sep=' ') == \
           ['foo bar', 'bar baz']
    assert ngrams(['foo', 'bar', 'baz'], 2, join=tuple) == \
           [('foo', 'bar'), ('bar', 'baz')]
    assert ngrams(['foo', 'bar', 'baz'], 2, join=lambda x: ''.join(x)) == \
           [('foobar'), ('barbaz')]


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
    assert remove_ngram(['foo', 'bar', 'ham', 'spam'], ('bar', 'ham'))\
           == ['foo', 'spam']
