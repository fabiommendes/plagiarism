from plagiarism.stopwords import get_stop_words, remove_stop_words


def test_stop_words():
    assert 'the' in get_stop_words()
    assert 'em' in get_stop_words('portuguese')
    assert 'def' not in get_stop_words('python')


def test_remove_stop_words():
    words = 'fui para casa'.split()
    assert remove_stop_words(words, 'portuguese') == ['casa']