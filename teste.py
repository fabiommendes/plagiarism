from pprint import pprint

from matplotlib import pyplot as plt

from plagiarism import sorted_tokens_all
from plagiarism.bag_of_words import bag_of_documents, \
    vectorize, similarity_matrix, most_similar, unvectorize
from plagiarism.clusterization import kmeans
from plagiarism.datasets import speeches
from plagiarism.ngrams import load_ngrams, merge_ngrams_all
from plagiarism.stopwords import remove_stop_words
from plagiarism.tokenizers import stemmize

stop_words_extra = '''é sr srs sra sras president deput tod quer brasil
    brasileir anos polít aqui faz país outr vam v exa orador'''.split()

L = sorted({x for x in speeches() if x})
L = L[:100]

docs = [stemmize(doc, 'portuguese') for doc in L]
docs = [remove_stop_words(doc, stop_words_extra) for doc in docs]
ngrams = load_ngrams('ngrams.stems')

print('merging ngrams')
docs = merge_ngrams_all(docs, ngrams)

print('bag of words')
bow = bag_of_documents(docs, 'boolean-weighted')

print('vectorizing')
matrix = vectorize(bow)

# print('similarity matrix')
# sim_matrix = similarity_matrix(matrix, method='', norm='L1')
# plt.hist(sim_matrix.flatten(), 50)
# plt.show()
# print(sim_matrix)
#
# print('finding most similar elements')
# sim = most_similar(docs, sim_matrix)
# i, j = sim[0].indexes
# print(L[i])
# print('*' * 80)
# print(L[j])

centroids, labels = kmeans(matrix, 20)
tokens = sorted_tokens_all(bow)
vecs = unvectorize(centroids, tokens)
for vec in vecs:
    pprint(vec.most_common(10))