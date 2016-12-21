from plagiarism.tokenizers import split_python_tokens, \
    split_programming_tokens, tokenize, stemmize


def test_python_tokens(fibo):
    assert ' '.join(split_python_tokens(fibo)) == 'def fibo ( n ) : ' \
                                                  'x , y = 1 , 1 ' \
                                                  'for _ in range ( n ) : ' \
                                                  'x , y = y , x + y ' \
                                                  'return x'


def test_generic_programming_tokens(fibo):
    tokens = split_programming_tokens(fibo)
    assert tokens == [
        'def', 'fibo', '(', 'n', ')', ':',
        'x', ',', 'y', '=', '1', ',', '1',
        'for', '_', 'in', 'range', '(', 'n', ')', ':',
        'x', ',', 'y', '=', 'y', ',', 'x', '+', 'y',
        'return', 'x'
    ]


def test_stemmizer():
    stems = stemmize('carro andou para rua', language='portuguese')
    assert stems == ['carr', 'andou', 'rua']


def test_split_words():
    words = tokenize('foo bar. foo, bar!', tokenizer='split-words',
                     keep_stops=True)
    assert words == ['foo', 'bar', '.', 'foo', 'bar', '.']

