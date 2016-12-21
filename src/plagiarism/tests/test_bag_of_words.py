from plagiarism.bag_of_words import bag_of_words, bag_of_documents, vectorize, \
    similarity_matrix, most_similar, common_tokens_all
from plagiarism.ngrams import optimal_bigrams
from plagiarism.text import text_diff
from plagiarism.tokenizers import split_python_tokens, tokenize_all


def test_bag_of_words(fibo):
    bag = bag_of_words(fibo, tokenizer=split_python_tokens, method='frequency')
    common = bag.most_common(3)
    assert sorted([x for x, y in common]) == [',', 'x', 'y']


def test_bag_of_documents(fibo_docs):
    fibo_toks = tokenize_all(fibo_docs, tokenizer=split_python_tokens)
    docs = fibo_toks
    docs = optimal_bigrams(docs, 1, accumulate=True)
    bag = bag_of_documents(docs, method='weighted')
    M = vectorize(bag)
    M = similarity_matrix(M)
    sim = most_similar(fibo_docs, M, 5)
    a, b = sim[0]
    print(a)
    print(b)
    assert a == fibo_docs[0]
    assert b == fibo_docs[4]


def test_common_tokens(fibo_docs):
    toks = tokenize_all(fibo_docs)
    common = common_tokens_all(toks, 4)
    common = [x for x, n in common]
    assert sorted(common) == ['1', 'n', 'x', 'y']

    common = common_tokens_all(toks, 10, by_document=True)
    common = sorted([x for x, n in common if n == 1.0])
    assert common == ['1', 'def', 'fibo', 'n', 'return']