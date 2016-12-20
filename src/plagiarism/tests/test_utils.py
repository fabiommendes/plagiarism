from plagiarism.utils import count_all, timeit


def test_count_all(tokens):
    assert count_all([tokens, tokens], 'doc-freq') == \
           count_all([tokens], 'doc-freq')

    assert count_all([tokens]) == {k: 1 for k in tokens}
    assert count_all([tokens, tokens], 'doc') == {k: 2 for k in tokens}


def test_timeit():
    with timeit() as dt:
        pass

    assert dt.value > 0.0

    with timeit() as dt:
        pass
    intermediate = dt.value
    with dt.update():
        pass
    assert dt.value > intermediate

