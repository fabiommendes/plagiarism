from math import sqrt, log
from pprint import pprint

from matplotlib import pyplot as plt

from plagiarism import *
from plagiarism.clusterization import kmeans, dbscan
from plagiarism.reduction import plot_reduced
from plagiarism.datasets import speeches

stop_words_extra = '''é sr srs sra sras president deput tod quer brasil
    brasileir anos polít aqui faz país outr vam v exa orador vot cas revisã
    projet porqu ordem pod obrig aqua part trabalh hoj dess dest aqu parlament
    import precis govern estad lei bloc entã sim não diz vai portant aprov
    comissão

    emenda acord direit dia contr

    pt psdb pmdb psb pdt pp

    df rj sp mg
    '''.split()

L = sorted({x for x in speeches() if x})
L = L[:200]

docs = [tokenize(doc, 'stems', language='portuguese') for doc in L]
docs = [remove_stop_words(doc, stop_words_extra) for doc in docs]
docs = [[w for w in doc if not w.isdigit()] for doc in docs]
ngrams = load_ngrams('ngrams.stems')
print_summary(docs, False)

print('merging ngrams')
docs = merge_ngrams_all(docs, ngrams)
print_summary(docs, False)

print('removing unique words')
docs = remove_unique_tokens(docs)
print_summary(docs, False)

print('bag of words')
#weights = compute_weights(docs, method='df', func=lambda f, N: log(N / f))
weights = None
bow = bag_of_documents(docs, 'freq', func=sqrt, weights=weights)

print('vectorizing')
matrix = vectorize(bow)

# print('similarity matrix')
# sim_matrix = similarity_matrix(matrix, method='diff', norm='L2')
# plt.hist(sim_matrix.flatten(), 50)
# plt.show()
# print(sim_matrix)

print('reducing dimensionality')
plot_reduced(matrix, True, whiten=True)

print('computing dbscan')
samples, labels = dbscan(matrix, eps=3, k=10, metric='l2', normalize=False)
pprint(labels)
pprint(samples)
pprint(count(labels).most_common())


# print('finding most similar elements')
# sim = most_similar(docs, sim_matrix)
# i, j = sim[0].indexes
# print(L[i])
# print('*' * 80)
# print(L[j])

# centroids, labels = kmeans(matrix, 10, whiten=False)
# tokens = sorted_tokens_all(bow)
# vecs = unvectorize(centroids, tokens)
# for i, vec in enumerate(vecs):
#     print('-' * 80)
#     print(list(labels).count(i))
#     pprint(vec.most_common(10))