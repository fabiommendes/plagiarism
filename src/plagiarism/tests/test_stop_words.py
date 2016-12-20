from plagiarism.stopwords import get_stop_words


def test_stop_words():
    assert 'the' in get_stop_words()
    assert 'em' in get_stop_words('portuguese')
    assert 'def' not in get_stop_words('python')
