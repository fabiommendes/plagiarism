from plagiarism.weigths import tokens_set

__all__ = [
    'print_summary',
]


def print_summary(documents, title=True):
    """
    Prints a summary with information about documents.
    """

    tokens = tokens_set(documents)
    size = sum(len(doc) for doc in documents)

    if title:
        print('Document summary:')
    print('    %s documents' % len(documents))
    print('    %s unique tokens' % len(tokens))
    print('    %s tokens' % size)
    print()
