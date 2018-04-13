import matplotlib.pyplot as plt                                # plotting
import numpy as np                                             # dense matrices
import pandas as pd
import time
import json
import sys
import os
from scipy.sparse import csr_matrix                            # sparse matrices
from sklearn.preprocessing import normalize                    # normalizing vectors
from sklearn.metrics import pairwise_distances


def load_sparse_csr(filename):
  loader = np.load(filename)
  data = loader['data']
  indices = loader['indices']
  indptr = loader['indptr']
  shape = loader['shape']
  return csr_matrix((data, indices, indptr), shape)


wiki = pd.read_csv('people_wiki.csv')
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
tf_idf = normalize(tf_idf)

with open('people_wiki_map_index_to_word.json') as people_wiki_map_index_to_word:
  map_index_to_word = json.load(people_wiki_map_index_to_word)
table = sorted(map_index_to_word, key=map_index_to_word.get)
print(wiki.shape[0])
print(wiki.iloc[37198])
########################################################################################################################


# Implement k-means
def get_initial_centroids(data, k, seed=None):
  if seed is not None:
    np.random.seed(seed)
  n = data.shape[0]
  rand_indices = np.random.randint(0, n, k)
  centroids = data[rand_indices, :].toarray()
  return centroids


def assign_clusters(data, centroids):
  distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
  cluster_assignment = np.argmin(distances_from_centroids, axis=1)
  return cluster_assignment


def revise_centroids(data, k, cluster_assignment):
  new_centroids = []
  for i in xrange(k):
    # Select all data points that belong to cluster i. Fill in the blank (RHS only)
    member_data_points = data[cluster_assignment == i]
    # Compute the mean of the data points. Fill in the blank (RHS only)
    centroid = member_data_points.mean(axis=0)
    # Convert numpy.matrix type to numpy.ndarray type
    centroid = centroid.A1
    new_centroids.append(centroid)
  new_centroids = np.array(new_centroids)
  return new_centroids


# Assessing convergence
def compute_heterogeneity(data, k, centroids, cluster_assignment):
  heterogeneity = 0.0
  for i in xrange(k):
    members_data_points = data[cluster_assignment == i, :]
    if members_data_points.shape[0] > 0:
      distances = pairwise_distances(members_data_points, [centroids[i]], metric='euclidean')
      sq_distances = distances**2
      heterogeneity += np.sum(sq_distances)
  return heterogeneity


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
  centroids = initial_centroids[:]
  prev_cluster_assignment = None
  for itr in xrange(maxiter):
    if verbose:
      print(itr)
    cluster_assignment = assign_clusters(data, centroids)
    centroids = revise_centroids(data, k, cluster_assignment)
    # Check for convergence: if none of the assignments changed, stop
    if prev_cluster_assignment is not None and (prev_cluster_assignment == cluster_assignment).all():
      break
    if prev_cluster_assignment is not None:
      num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
      if verbose:
        print('    elements changed their cluster assignment: ', num_changed)
    if record_heterogeneity is not None:
      score = compute_heterogeneity(data, k, centroids, cluster_assignment)
      record_heterogeneity.append(score)
    prev_cluster_assignment = cluster_assignment[:]
  return centroids, cluster_assignment


# Plotting convergence metric
def plot_heterogeneity(heterogeneity, k):
  plt.figure(figsize=(7, 4))
  plt.plot(heterogeneity, linewidth=4)
  plt.xlabel('# Iterations')
  plt.ylabel('Heterogeneity')
  plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
  plt.rcParams.update({'font.size': 16})
  plt.tight_layout()
  plt.show()



k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity, verbose=True)
plot_heterogeneity(heterogeneity, k)

# Which of the cluster contains the greatest number of data points in the end?
print(np.bincount(cluster_assignment))

# Beware of local minima
k = 10
heterogeneity = {}

start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
  initial_centroids = get_initial_centroids(tf_idf, k, seed)
  centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                         record_heterogeneity=None, verbose=False)
  # To save time, compute heterogeneity only once in the end
  heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
  print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
  sys.stdout.flush()
end = time.time()
print(end - start)



def smart_initialize(data, k, seed=None):
  '''Use k-means++ to initialize a good set of centroids'''
  if seed is not None:  # useful for obtaining consistent results
    np.random.seed(seed)
  centroids = np.zeros((k, data.shape[1]))

  # Randomly choose the first centroid.
  # Since we have no prior knowledge, choose uniformly at random
  idx = np.random.randint(data.shape[0])
  centroids[0] = data[idx, :].toarray()
  # Compute distances from the first centroid chosen to all the other data points
  squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2

  for i in xrange(1, k):
    # Choose the next centroid randomly, so that the probability for each data point to be chosen
    # is directly proportional to its squared distance from the nearest centroid.
    # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
    idx = np.random.choice(data.shape[0], 1, p=squared_distances / sum(squared_distances))
    centroids[i] = data[idx, :].toarray()
    # Now compute distances from the centroids to all data points
    squared_distances = np.min(pairwise_distances(data, centroids[0:i + 1], metric='euclidean')**2, axis=1)

  return centroids



k = 10
heterogeneity_smart = {}
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
  initial_centroids = smart_initialize(tf_idf, k, seed)
  centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                         record_heterogeneity=None, verbose=False)
  # To save time, compute heterogeneity only once in the end
  heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
  print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
  sys.stdout.flush()
end = time.time()
print(end - start)


# plotting
plt.figure(figsize=(8, 5))
plt.boxplot([heterogeneity.values(), heterogeneity_smart.values()], vert=False)
plt.yticks([1, 2], ['k-means', 'k-means++'])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()



def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
  heterogeneity = {}

  min_heterogeneity_achieved = float('inf')
  best_seed = None
  final_centroids = None
  final_cluster_assignment = None

  for i in xrange(num_runs):

        # Use UTC time if no seeds are provided
    if seed_list is not None:
      seed = seed_list[i]
      np.random.seed(seed)
    else:
      seed = int(time.time())
      np.random.seed(seed)

    # Use k-means++ initialization
    # YOUR CODE HERE
    initial_centroids = smart_initialize(data, k, seed)

    # Run k-means
    # YOUR CODE HERE
    centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False)

    # To save time, compute heterogeneity only once in the end
    # YOUR CODE HERE
    heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)

    if verbose:
      print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
      sys.stdout.flush()

    # if current measurement of heterogeneity is lower than previously seen,
    # update the minimum record of heterogeneity.
    if heterogeneity[seed] < min_heterogeneity_achieved:
      min_heterogeneity_achieved = heterogeneity[seed]
      best_seed = seed
      final_centroids = centroids
      final_cluster_assignment = cluster_assignment

  # Return the centroids and cluster assignments that minimize heterogeneity.
  return final_centroids, final_cluster_assignment


########################################################################################################################


def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
  plt.figure(figsize=(7, 4))
  plt.plot(k_values, heterogeneity_values, linewidth=4)
  plt.xlabel('K')
  plt.ylabel('Heterogeneity')
  plt.title('K vs. Heterogeneity')
  plt.rcParams.update({'font.size': 16})
  plt.tight_layout()
  plt.show()


filename = 'kmeans-arrays.npz'

heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
  arrays = np.load(filename)
  centroids = {}
  cluster_assignment = {}
  for k in k_list:
    print k
    sys.stdout.flush()
    centroids[k] = arrays['centroids_{0:d}'.format(k)]
    cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]
    score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
    heterogeneity_values.append(score)
  # plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
  print('File not found. Skipping.')

########################################################################################################################

# Visualize clusters of documents


def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, map_index_to_word, display_content=True):
  '''wiki: original dataframe
     tf_idf: data matrix, sparse matrix format
     map_index_to_word: SFrame specifying the mapping betweeen words and column indices
     display_content: if True, display 8 nearest neighbors of each centroid'''

  print('==========================================================')
  # Visualize each cluster c
  for c in xrange(k):
        # Cluster heading
    print('Cluster {0:d}    '.format(c)),
    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroids[c].argsort()[::-1]
    for i in xrange(5):  # Print each word along with the TF-IDF weight
      print('{0:s}:{1:.3f}'.format(map_index_to_word[idx[i]], centroids[c, idx[i]])),
    print('')

    if display_content:
      # Compute distances from the centroid to all data points in the cluster,
      # and compute nearest neighbors of the centroids within the cluster.
      distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
      distances[cluster_assignment != c] = float('inf')  # remove non-members from consideration
      nearest_neighbors = distances.argsort()
      # For 8 nearest neighbors, print the title as well as first 180 characters of text.
      # Wrap the text at 80-character mark.
      for i in xrange(8):
        text = ' '.join(wiki.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki.iloc[nearest_neighbors[i]]['name'],
                                                             distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
    print('==========================================================')


visualize_document_clusters(wiki, tf_idf, centroids[2], cluster_assignment[2], 2, table)
k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, table)
print(np.bincount(cluster_assignment[10]))
visualize_document_clusters(wiki, tf_idf, centroids[25], cluster_assignment[25], 25,
                            table, display_content=False)
