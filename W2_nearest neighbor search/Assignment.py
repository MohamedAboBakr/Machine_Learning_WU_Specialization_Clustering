import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
import pandas as pd
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances


# Load Wikipedia dataset
wiki = pd.read_csv('people_wiki.csv')

# Extract word count vectors


def load_sparse_csr(filename):
  loader = np.load(filename)
  data = loader['data']
  indices = loader['indices']
  indptr = loader['indptr']
  shape = loader['shape']
  return csr_matrix((data, indices, indptr), shape)


word_count = load_sparse_csr('people_wiki_word_count.npz')
with open('people_wiki_map_index_to_word.json') as people_wiki_map_index_to_word:
  map_index_to_word = json.load(people_wiki_map_index_to_word)


# Find nearest neighbors using word count vectors
model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)
print(distances)
print(indices)
neighbors = pd.DataFrame(data={'distance': distances.flatten()}, index=indices.flatten())
neighbors_names_distances = wiki.join(neighbors).sort_values('distance')[['name', 'distance']][:10]
print(neighbors_names_distances)

####################################################################################################################


def unpack_dict(matrix, map_index_to_word):
  table = sorted(map_index_to_word, key=map_index_to_word.get)
  data = matrix.data
  indices = matrix.indices
  indptr = matrix.indptr
  num_doc = matrix.shape[0]
  return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i + 1]]],
                                data[indptr[i]:indptr[i + 1]].tolist())}
          for i in xrange(num_doc)]


def top_words(name):
  row = wiki[wiki['name'] == name]
  dic = row['word_count'].iloc[0]
  word_count_ = pd.DataFrame(dic.items(), columns=['word', 'count'])
  word_count_table = word_count_.sort_values(['count'], ascending=False)
  return word_count_table


def has_top_words(word_count_vector):
  # extract the keys of word_count_vector and convert it to a set
  unique_words = set(word_count_vector.keys())
  # return True if common_words is a subset of unique_words
  # return False otherwise
  return set_most_5_common_words.issubset(unique_words)


wiki['word_count'] = unpack_dict(word_count, map_index_to_word)

top_words_obama = top_words('Barack Obama')
top_words_Barrio = top_words('Francisco Barrio')
combined_words = top_words_obama.set_index('word').join(top_words_Barrio.set_index('word'), lsuffix='_obama', rsuffix='_barrio')
combined_words = combined_words.rename(index=str, columns={'count_obama': 'Obama', 'count_barrio': 'Barrio'})
combined_words = combined_words.dropna()
most_5_common_words = combined_words['Obama'][:5].keys().tolist()
set_most_5_common_words = set(most_5_common_words)
print(set_most_5_common_words)

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
print(sum(wiki['has_top_words'] == True))

obama_index = wiki.index[wiki['name'] == 'Barack Obama'].tolist()
obama_words = word_count[obama_index]

Bush_index = wiki.index[wiki['name'] == 'George W. Bush'].tolist()
Bush_words = word_count[Bush_index]

Biden_index = wiki.index[wiki['name'] == 'Joe Biden'].tolist()
Biden_words = word_count[Biden_index]


dist1 = euclidean_distances(obama_words, Bush_words)
dist2 = euclidean_distances(obama_words, Biden_words)
dist3 = euclidean_distances(Biden_words, Bush_words)
print(dist1, dist2, dist3)


top_words_Bush = top_words('George W. Bush')
combined_words = top_words_obama.set_index('word').join(top_words_Bush.set_index('word'), lsuffix='_obama', rsuffix='_bush')
combined_words = combined_words.rename(index=str, columns={'count_obama': 'Obama', 'count_bush': 'Bush'})
combined_words = combined_words.dropna()
combined_words['Bush'] = combined_words['Bush'].apply(lambda x: int(x))
print(combined_words.head(10))

####################################################################################################################


# Extract the TF-IDF vectors
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)

model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)
neighbors = pd.DataFrame(data={'distance': distances.flatten()}, index=indices.flatten())
neighbors_names_distances = wiki.join(neighbors).sort_values('distance')[['name', 'distance']][:10]
print(neighbors_names_distances)


def top_words_tf_idf(name):
  row = wiki[wiki['name'] == name]
  dic = row['tf_idf'].iloc[0]
  word_count_ = pd.DataFrame(dic.items(), columns=['word', 'weight'])
  word_count_table = word_count_.sort_values(['weight'], ascending=False)
  return word_count_table


obama_tf_idf = top_words_tf_idf('Barack Obama')
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')

combined_words = obama_tf_idf.set_index('word').join(schiliro_tf_idf.set_index('word'), lsuffix='_obama', rsuffix='_schiliro')
combined_words = combined_words.rename(index=str, columns={'weight_obama': 'Obama', 'weight_schiliro': 'Schiliro'})
combined_words = combined_words.dropna()
most_5_common_words = combined_words['Obama'][:5].keys().tolist()
set_most_5_common_words = set(most_5_common_words)
print(set_most_5_common_words)

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
print(sum(wiki['has_top_words'] == True))


obama_index = wiki.index[wiki['name'] == 'Barack Obama'].tolist()
obama_words = tf_idf[obama_index]
Biden_index = wiki.index[wiki['name'] == 'Joe Biden'].tolist()
Biden_words = tf_idf[Biden_index]
dist1 = euclidean_distances(obama_words, Biden_words)
print(dist1)


####################################################################################################################

# Comptue length of all documents
def compute_length(row):
  return len(row['text'].split(' '))


wiki['length'] = wiki.apply(compute_length, axis=1)

# Compute 100 nearest neighbors and display their lengths
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame(data={'distance': distances.flatten()}, index=indices.flatten())
nearest_neighbors_euclidean = wiki.join(neighbors).sort_values('distance')[['name', 'length', 'distance']]
print(nearest_neighbors_euclidean.head())

'''
# plotting

plt.figure(figsize=(10.5, 4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)

plt.hist(nearest_neighbors_euclidean['length'][:100], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Barack Obama']].tolist()[0], color='k', linestyle='--', linewidth=4,
            label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Joe Biden']].tolist()[0], color='g', linestyle='--', linewidth=4,
            label='Length of Joe Biden', zorder=1)

plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size': 15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()
'''

#  we turn to cosine distances
model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model2_tf_idf.fit(tf_idf)
distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame(data={'distance': distances.flatten()}, index=indices.flatten())
nearest_neighbors_cosine = wiki.join(neighbors)[['name', 'length', 'distance']].sort_values('distance')
print(nearest_neighbors_cosine)

# plotting all
'''
plt.figure(figsize=(10.5, 4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)

plt.hist(nearest_neighbors_euclidean['length'][:100], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'][:100], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Barack Obama']].tolist()[0], color='k', linestyle='--', linewidth=4,
            label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki.index[wiki['name'] == 'Joe Biden']].tolist()[0], color='g', linestyle='--', linewidth=4,
            label='Length of Joe Biden', zorder=1)

plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size': 15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()
'''


# Problem with cosine distances: tweets vs. long articles
tweet = {'act': 3.4597778278724887,
         'control': 3.721765211295327,
         'democratic': 3.1026721743330414,
         'governments': 4.167571323949673,
         'in': 0.0009654063501214492,
         'law': 2.4538226269605703,
         'popular': 2.764478952022998,
         'response': 4.261461747058352,
         'to': 0.04694493768179923}

word_indices = []
for word in tweet.keys():
  if word in map_index_to_word.keys():
    word_indices.append(map_index_to_word[word])

tweet_tf_idf = csr_matrix((list(tweet.values()), ([0] * len(word_indices), word_indices)),
                          shape=(1, tf_idf.shape[1]))

obama_tf_idf = tf_idf[35817]
print (cosine_distances(obama_tf_idf, tweet_tf_idf))
distances, indices = model2_tf_idf.kneighbors(obama_tf_idf, n_neighbors=10)
print (distances)
