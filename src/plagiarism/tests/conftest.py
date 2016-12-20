import os

import pytest

from plagiarism.tokenizers import split_programming_tokens

BASEPATH = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def fibo(fibo_docs):
    return fibo_docs[0]


@pytest.fixture(scope='session')
def fibo_docs(fibo_path):
    files = []
    for file in sorted(os.listdir(fibo_path)):
        with open(os.path.join(fibo_path, file)) as F:
            files.append(F.read())
    return tuple(files)


@pytest.fixture(scope='session')
def fibo_doc_tokens(fibo_docs):
    documents = []
    for doc in fibo_docs:
        documents.append(tuple(split_programming_tokens(doc)))
    return tuple(documents)


@pytest.fixture(scope='session')
def fibo_path():
    return os.path.join(BASEPATH, 'db-fibo')


@pytest.fixture(scope='session')
def tokens():
    return ['foo', 'bar', 'foobar', 'func', 'x', '>=', '1']
