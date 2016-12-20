"""
A prepared set of tasks.
"""
import collections
import os
import sys

from plagiarism.bag_of_words import vectorize, similarity_matrix, \
    bag_of_documents, most_similar
from plagiarism.input import ask, yn_input, do_print, no_print, clear
from plagiarism.ngrams import optimal_bigrams
from plagiarism.text import text_diff, two_column
from plagiarism.tokenizers import tokenize_all
from plagiarism.utils import tokens_all, timeit

suspect_result = collections.namedtuple('SuspectResult',
    field_names=['documents', 'similar_pairs', 'similarity_matrix', 'tokens']
)


def documents_map(documents=None):
    """
    Return a dictionary mapping documents to their respective content.
    """

    # Use cwd as path if documents is not given.
    if documents is None:
        documents = os.getcwd()

    if isinstance(documents, collections.Mapping):
        return collections.OrderedDict(documents)
    if isinstance(documents, str):
        path = os.path.abspath(documents)
        files = [os.path.join(path, f) for f in os.listdir(path)]
        files = [f for f in files if os.path.exists(f) and os.path.isfile(f)]
        documents = collections.OrderedDict()
        for f in files:
            with open(f) as F:
                documents[os.path.split(f)[1]] = F.read().strip()
        return documents
    raise NotImplementedError


def find_suspects(documents=None, tokenizer='code', verbose=False,
                  accumulate=False):
    """
    Find documents with the highest suspicion of plagiarism.

    Args:
        documents:
            Dictionary mapping document name to its content.

    Returns:
        documents:
            Ordered dictionary mapping document names to documents.
        similar_pairs:
            List of pairs in decreasing order of similarity.
        similarity_matrix:
            Symmetric matrix with similarity values for the (i, j) documents.
        tokens:
            List of tokens used in the final comparison.
    """

    info = do_print if verbose else no_print

    # Find documents
    with timeit() as dt:
        documents = documents_map(documents)
        document_list = list(documents.values())
        info('Processing %s documents (%es).' % (len(documents), dt))

    # Tokenize documents
    with timeit() as dt:
        tokenized = tokenize_all(document_list, tokenizer=tokenizer)
        tokens = tokens_all(tokenized)
        info('%s tokens found.' % len(tokens))

    # Creating ngrams
    with dt.update():
        tokenized = optimal_bigrams(tokenized, 1,
                                    accumulate=accumulate,
                                    allow_superposition=False)
        tokens = tokens_all(tokenized)
        info('Using %s unique n-grams. (%es)' % (len(tokens), dt))

    # Bag of words
    with timeit() as dt:
        bag = bag_of_documents(tokenized, method='weighted')
        info('Computed bag of words for all documents. (%es)' % dt)

    # Computing similarity matrix
    with timeit() as dt:
        data = vectorize(bag)
        matrix = similarity_matrix(data, method='triangular', norm='l1')
        similar = most_similar(document_list, matrix)
        values = [x.similarity for x in similar]
        fmt = (min(values), max(values), dt)
        info('Similarity in the %s-%s range. (%es)' % fmt)

    return suspect_result(similar_pairs=similar,
                          similarity_matrix=matrix,
                          tokens=tokens,
                          documents=documents)


def interactive_find_suspects(documents=None, accumulate=None, show='diff'):
    """
    Interactive version of find_suspects.
    """

    clear()
    suspects = find_suspects(
        documents,
        verbose=True,
        accumulate=ask(accumulate, yn_input, 'Keep unigrams? [Y/n] '),
    )
    doc_names = list(suspects.documents.keys())

    if not yn_input('See documents? [Y/n] '):
        return
    clear()

    for pair in suspects.similar_pairs:
        i, j = pair.indexes
        name1, name2 = doc_names[i], doc_names[j]
        do_print('# %s vs. %s\n' % (name1, name2))
        do_print('Similarity: %.1f%%\n' % (100 * pair.similarity))
        doc1, doc2 = pair
        if show == 'diff':
            do_print(text_diff(doc1, doc2))
        elif show == 'twocolumn':
            do_print(two_column(doc1, doc2))
        do_print()
        if not yn_input('Next? [Y/n] '):
            break
        clear()

if __name__ == '__main__':
    interactive_find_suspects(sys.argv[-1], accumulate=True)
