from pprint import pprint

from matplotlib import pyplot as plt

from plagiarism.bag_of_words import bag_of_documents, \
    vectorize, similarity_matrix, most_similar, unvectorize
from plagiarism.clusterization import kmeans
from plagiarism.datasets import speeches
from plagiarism.ngrams import load_ngrams, merge_ngrams_all
from plagiarism.tokenizers import stemmize

L = sorted({x for x in speeches() if x})
L = L[:100]
docs = [stemmize(doc, 'portuguese') for doc in L]
ngrams = load_ngrams('ngrams.stems')
print('merging ngrams')
docs = merge_ngrams_all(docs, ngrams)

print('bag of words')
bow = bag_of_documents(docs, 'weighted')

print('vectorizing')
matrix = vectorize(bow)

print('similarity matrix')
sim_matrix = similarity_matrix(matrix, method='diff', norm='L1')
#plt.hist(sim_matrix.flatten(), 50)
#plt.show()
print(sim_matrix)

print('finding most similar elements')
sim = most_similar(docs, sim_matrix)
i, j = sim[0].indexes
print(L[i])
print('*' * 80)
print(L[j])

centroids, labels = kmeans(matrix, 20)
unvectorize(centroids)