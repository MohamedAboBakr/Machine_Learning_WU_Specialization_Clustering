# Implementing Locality Sensitive Hashing from scratch

import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
import pandas as pd
import json
from sklearn.metrics.pairwise import pairwise_distances        # pairwise distances
from copy import copy                                          # deep copies
from itertools import combinations
import time


def norm(x):
  sum_sq = x.dot(x.T)
  norm = np.sqrt(sum_sq)
  return(norm)


def unpack_dict(matrix, map_index_to_word):
  table = sorted(map_index_to_word, key=map_index_to_word.get)
  data = matrix.data
  indices = matrix.indices
  indptr = matrix.indptr
  num_doc = matrix.shape[0]
  return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i + 1]]],
                                data[indptr[i]:indptr[i + 1]].tolist())}
          for i in xrange(num_doc)]


def load_sparse_csr(filename):
  loader = np.load(filename)
  data = loader['data']
  indices = loader['indices']
  indptr = loader['indptr']
  shape = loader['shape']
  return csr_matrix((data, indices, indptr), shape)


def generate_random_vectors(num_vector, dim):
  return np.random.randn(dim, num_vector)


def cosine_distance(x, y):
  xy = x.dot(y.T)
  dist = xy / (norm(x) * norm(y))
  return 1 - dist[0, 0]


# Load Wikipedia dataset
wiki = pd.read_csv('people_wiki.csv')
with open('people_wiki_map_index_to_word.json') as people_wiki_map_index_to_word:
  map_index_to_word = json.load(people_wiki_map_index_to_word)

tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)

doc = tf_idf[0, :]  # vector of tf-idf values for document 0
np.random.seed(0)
random_vectors = generate_random_vectors(num_vector=16, dim=547979)
print(doc.dot(random_vectors[:, 0]) >= 0)
bits = (doc.dot(random_vectors) >= 0)
pow_2 = (1 << np.arange(15, -1, -1))
val = bits.dot(pow_2)
print(val)


#####################################################################################################################


# Train an LSH model
# we will build a popular variant of LSH known as random binary projection

def train_lsh(data, num_vector=16, seed=None):

  dim = data.shape[1]
  if seed is not None:
    np.random.seed(seed)
  random_vectors = generate_random_vectors(num_vector, dim)

  powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

  table = {}

  # Partition data points into bins
  bin_index_bits = (data.dot(random_vectors) >= 0)

  # Encode bin index bits into integers
  bin_indices = bin_index_bits.dot(powers_of_two)

  # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
  for data_index, bin_index in enumerate(bin_indices):
    if bin_index not in table:
      # If no list yet exists for this bin, assign the bin an empty list.
      table[bin_index] = []
    # Fetch the list of document ids associated with the bin and add the document id to the end.
    table[bin_index].append(data_index)

  model = {'data': data,
           'bin_index_bits': bin_index_bits,
           'bin_indices': bin_indices,
           'table': table,
           'random_vectors': random_vectors,
           'num_vector': num_vector}

  return model


# checkpoint
model = train_lsh(tf_idf, num_vector=16, seed=143)
table = model['table']

if 0 in table and table[0] == [39583] and \
   143 in table and table[143] == [19693, 28277, 29776, 30399]:
  print 'Passed!'
else:
  print 'Check your code.'

#####################################################################################################################

# Inspect bins

print (wiki[wiki['name'] == 'Barack Obama'])
id = 35817
for bin, data in table.items():
  if id in data:
    print bin

print (wiki[wiki['name'] == 'Joe Biden'])
id = 24478
for bin, data in table.items():
  if id in data:
    print bin

print(sum(model['bin_index_bits'][35817] == model['bin_index_bits'][24478]))

print(wiki[wiki['name'] == 'Wynn Normington Hugh-Jones'])
print(np.array(model['bin_index_bits'][22745], dtype=int))
print(model['bin_index_bits'][35817] == model['bin_index_bits'][22745])

# the documents in the same bin as Barack Obama
same_bin = list(model['table'][model['bin_indices'][35817]])
same_bin.remove(35817)
print(same_bin)
docs = wiki[wiki.index.isin(same_bin)]
print(docs)


# In a high-dimensional space such as text features, we often get unlucky with our selection of only a few random vectors such that dissimilar data points go into the same bin while similar data points fall into different bins
obama_tf_idf = tf_idf[35817, :]
biden_tf_idf = tf_idf[24478, :]
print(cosine_distance(obama_tf_idf, biden_tf_idf))

for doc in same_bin:
  tf_idf_id = tf_idf[doc, :]
  print(cosine_distance(obama_tf_idf, tf_idf_id))

#####################################################################################################################


# Query the LSH model

def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
  """
  For a given query vector and trained LSH model, return all candidate neighbors for
  the query among all bins within the given search radius.

  Example usage
  -------------
  >>> model = train_lsh(corpus, num_vector=16, seed=143)
  >>> q = model['bin_index_bits'][0]  # vector for the first document

  >>> candidates = search_nearby_bins(q, model['table'])
  """
  num_vector = len(query_bin_bits)
  powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

  # Allow the user to provide an initial set of candidates.
  candidate_set = copy(initial_candidates)

  for different_bits in combinations(range(num_vector), search_radius):
    # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
    # Hint: you can iterate over a tuple like a list
    alternate_bits = copy(query_bin_bits)
    for i in different_bits:
      alternate_bits[i] = 1 - alternate_bits[i]

    # Convert the new bit vector to an integer index
    nearby_bin = alternate_bits.dot(powers_of_two)

    # Fetch the list of documents belonging to the bin indexed by the new bit vector.
    # Then add those documents to candidate_set
    # Make sure that the bin exists in the table!
    # Hint: update() method for sets lets you add an entire list to the set
    if nearby_bin in table:
      candidate_set |= set(table[nearby_bin])

  return candidate_set


'''
# checkpoint
obama_bin_index = model['bin_index_bits'][35817]  # bin index of Barack Obama
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0)
if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
  print 'Passed test'
else:
  print 'Check your code'
print 'List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261'
print candidate_set


candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1, initial_candidates=candidate_set)
if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                         23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                         19699, 2804, 20347]):
  print 'Passed test'
else:
  print 'Check your code'
print(candidate_set)
'''


def query(vec, model, k, max_search_radius):

  data = model['data']
  table = model['table']
  random_vectors = model['random_vectors']

  # Compute bin index for the query vector, in bit representation.
  bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()

  # Search nearby bins and collect candidates
  candidate_set = set()
  for search_radius in xrange(max_search_radius + 1):
    candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)

  # Sort candidates by their true distances from the query
  nearest_neighbors = pd.DataFrame(index=candidate_set)
  candidates = data[np.array(list(candidate_set)), :]
  nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()

  return nearest_neighbors.sort_values(['distance'], ascending=True)[:k], len(candidate_set)


# Experimenting with your LSH implementation

num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in xrange(17):
  start = time.time()
  # Perform LSH query using Barack Obama, with max_search_radius
  result, num_candidates = query(tf_idf[35817, :], model, k=10,
                                 max_search_radius=max_search_radius)
  end = time.time()
  query_time = end - start  # Measure time

  print('Radius:', max_search_radius)
  # Display 10 nearest neighbors, along with document ID and name
  print(result.join(wiki).sort_values('distance')['name'])

  # Collect statistics on 10 nearest neighbors
  average_distance_from_query = result['distance'][1:].mean()
  max_distance_from_query = result['distance'][1:].max()
  min_distance_from_query = result['distance'][1:].min()

  num_candidates_history.append(num_candidates)
  query_time_history.append(query_time)
  average_distance_from_query_history.append(average_distance_from_query)
  max_distance_from_query_history.append(max_distance_from_query)
  min_distance_from_query_history.append(min_distance_from_query)

'''
plt.figure(figsize=(7, 4.5))
plt.plot(num_candidates_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('# of documents searched')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(query_time_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()
'''

#####################################################################################################################

# Quality metrics for neighbors

def brute_force_query(vec, data, k):
  num_data_points = data.shape[0]
  # Compute distances for ALL data points in training set
  nearest_neighbors = pd.DataFrame(index=range(num_data_points))
  nearest_neighbors['distance'] = pairwise_distances(data, vec, metric='cosine').flatten()
  return nearest_neighbors.sort_values(['distance'], ascending=True)[:k]


max_radius = 17
precision = {i: [] for i in xrange(max_radius)}
average_distance = {i: [] for i in xrange(max_radius)}
query_time = {i: [] for i in xrange(max_radius)}

np.random.seed(0)
num_queries = 10

for i, ix in enumerate(np.random.choice(tf_idf.shape[0], num_queries, replace=False)):
  print('%s / %s' % (i, num_queries))
  ground_truth = set(brute_force_query(tf_idf[ix, :], tf_idf, k=25).index.values)
  # Get the set of 25 true nearest neighbors

  for r in xrange(1, max_radius):
    start = time.time()
    result, num_candidates = query(tf_idf[ix, :], model, k=10, max_search_radius=r)
    end = time.time()

    query_time[r].append(end - start)
    # precision = (# of neighbors both in result and ground_truth)/10.0
    precision[r].append(len(set(result.index.values) & ground_truth) / 10.0)
    average_distance[r].append(result['distance'][1:].mean())

'''
plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 17), [np.mean(average_distance[i]) for i in xrange(1, 17)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 17), [np.mean(precision[i]) for i in xrange(1, 17)], linewidth=4, label='Precison@10')
plt.xlabel('Search radius')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 17), [np.mean(query_time[i]) for i in xrange(1, 17)], linewidth=4, label='Query time')
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()
'''

#####################################################################################################################


# Effect of number of random vectors

precision = {i:[] for i in xrange(5,20)}
average_distance  = {i:[] for i in xrange(5,20)}
query_time = {i:[] for i in xrange(5,20)}
num_candidates_history = {i:[] for i in xrange(5,20)}
ground_truth = {}

np.random.seed(0)
num_queries = 10
docs = np.random.choice(tf_idf.shape[0], num_queries, replace=False)

for i, ix in enumerate(docs):
    ground_truth[ix] = set(brute_force_query(tf_idf[ix,:], tf_idf, k=25).index.values)
    # Get the set of 25 true nearest neighbors

for num_vector in xrange(5,20):
    print('num_vector = %s' % (num_vector))
    model = train_lsh(tf_idf, num_vector, seed=143)
    
    for i, ix in enumerate(docs):
        start = time.time()
        result, num_candidates = query(tf_idf[ix,:], model, k=10, max_search_radius=3)
        end = time.time()
        query_time[num_vector].append(end-start)
        precision[num_vector].append(len(set(result.index.values) & ground_truth[ix])/10.0)
        average_distance[num_vector].append(result['distance'][1:].mean())
        num_candidates_history[num_vector].append(num_candidates)


'''
plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(average_distance[i]) for i in xrange(5,20)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('# of random vectors')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(precision[i]) for i in xrange(5,20)], linewidth=4, label='Precison@10')
plt.xlabel('# of random vectors')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(query_time[i]) for i in xrange(5,20)], linewidth=4, label='Query time (seconds)')
plt.xlabel('# of random vectors')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(num_candidates_history[i]) for i in xrange(5,20)], linewidth=4,
         label='# of documents searched')
plt.xlabel('# of random vectors')
plt.ylabel('# of documents searched')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()
'''
