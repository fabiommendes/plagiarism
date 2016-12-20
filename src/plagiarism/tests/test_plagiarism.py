import pytest
import plagiarism


def test_some_basic_functionality_about():
    assert 2 == 1 + 1

    return

    text1 = 'hello, john! my name is paul.'
    text2 = 'hello, paul! my name is john.'
    text3 = 'hello, george! my name is foo.'

    job = Job([('txt1', text1), ('txt2', text2), ('txt3', text3)], CodeDocument)
    print(job.count_optional())
    print(job.select_metrics())
    print(job.elements())
    print(job.ndarray())
    print(job.kmeans(2))