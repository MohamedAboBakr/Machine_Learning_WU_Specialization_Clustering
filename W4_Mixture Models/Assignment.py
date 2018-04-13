import pandas as pd
import numpy as np                                             # dense matrices
import matplotlib.pyplot as plt                                # plotting
from scipy.stats import multivariate_normal                    # multivariate Gaussian distribution
import copy
# image handling library
from PIL import Image
from io import BytesIO
import sframe
import matplotlib.mlab as mlab


##########################################################################################################################
# Log likelihood

def log_sum_exp(Z):
  """ Compute log(\sum_i exp(Z_i)) for some array Z."""
  return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


def loglikelihood(data, weights, means, covs):
  """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
  num_clusters = len(means)
  num_dim = len(data[0])

  ll = 0
  for d in data:

    Z = np.zeros(num_clusters)
    for k in range(num_clusters):

      # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
      delta = np.array(d) - means[k]
      exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))

      # Compute loglikelihood contribution for this data point and this cluster
      Z[k] += np.log(weights[k])
      Z[k] -= 1 / 2. * (num_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)

    # Increment loglikelihood contribution of this data point across all clusters
    ll += log_sum_exp(Z)

  return ll

##########################################################################################################################


# E-step: assign cluster responsibilities, given current parameters


def compute_responsibilities(data, weights, means, covariances):
  '''E-step: compute responsibilities, given the current parameters'''
  num_data = len(data)
  num_clusters = len(means)
  resp = np.zeros((num_data, num_clusters))

  # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
  # Hint: To compute likelihood of seeing data point i given cluster k, use multivariate_normal.pdf.
  for i in range(num_data):
    for k in range(num_clusters):
      # YOUR CODE HERE
      resp[i, k] = weights[k] * multivariate_normal.pdf(data[i], mean=means[k], cov=covariances[k])

  # Add up responsibilities over each data point and normalize
  row_sums = resp.sum(axis=1)[:, np.newaxis]
  resp = resp / row_sums

  return resp


'''
# Checkpoint
resp = compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])

if resp.shape==(2,2) and np.allclose(resp, np.array([[0.10512733, 0.89487267], [0.46468164, 0.53531836]])):
    print 'Checkpoint passed!'
else:
    print 'Check your code again.'

'''
##########################################################################################################################


# M-step: Update parameters, given current cluster responsibilities

def compute_soft_counts(resp):
    # Compute the total responsibility assigned to each cluster, which will be useful when
    # implementing M-steps below. In the lectures this is called N^{soft}
  counts = np.sum(resp, axis=0)
  return counts


# Updating weights

def compute_weights(counts):
  num_clusters = len(counts)
  weights = [0.] * num_clusters
  for k in range(num_clusters):
    # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
    # HINT: compute # of data points by summing soft counts.
    # YOUR CODE HERE
    weights[k] = counts[k] / np.sum(counts)
  return weights


'''
# checkPoint
resp = compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.],[0,0]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
weights = compute_weights(counts)

print counts
print weights

if np.allclose(weights, [0.27904865942515705, 0.720951340574843]):
    print 'Checkpoint passed!'
else:
    print 'Check your code again.'
'''

# Updating means


def compute_means(data, resp, counts):
  num_clusters = len(counts)
  num_data = len(data)
  means = [np.zeros(len(data[0]))] * num_clusters

  for k in range(num_clusters):
    # Update means for cluster k using the M-step update rule for the mean variables.
    # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
    weighted_sum = 0.
    for i in range(num_data):
      # YOUR CODE HERE
      weighted_sum += data[i] * resp[i][k]
    # YOUR CODE HERE
    means[k] = weighted_sum / counts[k]

  return means


'''
# Checkpoint

data_tmp = np.array([[1., 2.], [-1., -2.]])
resp = compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0., 0.]), np.array([1., 1.])],
                                covariances=[np.array([[1.5, 0.], [0., 2.5]]), np.array([[1., 1.], [1., 2.]])])
counts = compute_soft_counts(resp)
means = compute_means(data_tmp, resp, counts)

if np.allclose(means, np.array([[-0.6310085, -1.262017], [0.25140299, 0.50280599]])):
  print 'Checkpoint passed!'
else:
  print 'Check your code again.'
'''

# Updating covariances


def compute_covariances(data, resp, counts, means):
  num_clusters = len(counts)
  num_dim = len(data[0])
  num_data = len(data)
  covariances = [np.zeros((num_dim, num_dim))] * num_clusters

  for k in range(num_clusters):
    # Update covariances for cluster k using the M-step update rule for covariance variables.
    # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
    weighted_sum = np.zeros((num_dim, num_dim))
    for i in range(num_data):
      # YOUR CODE HERE (Hint: Use np.outer on the data[i] and this cluster's mean)
      weighted_sum += resp[i][k] * np.outer(data[i] - means[k], data[i] - means[k])
    # YOUR CODE HERE
    covariances[k] = weighted_sum / counts[k]

  return covariances


'''
# checkpoint
data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
means = compute_means(data_tmp, resp, counts)
covariances = compute_covariances(data_tmp, resp, counts, means)

if np.allclose(covariances[0], np.array([[0.60182827, 1.20365655], [1.20365655, 2.4073131]])) and     np.allclose(covariances[1], np.array([[ 0.93679654, 1.87359307], [1.87359307, 3.74718614]])):
    print 'Checkpoint passed!'
else:
  print 'Check your code again.'
'''

##########################################################################################################################

# The EM algorithm


def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):

  # Make copies of initial parameters, which we will update during each iteration
  means = init_means[:]
  covariances = init_covariances[:]
  weights = init_weights[:]

  # Infer dimensions of dataset and the number of clusters
  num_data = len(data)
  num_dim = len(data[0])
  num_clusters = len(means)

  # Initialize some useful variables
  resp = np.zeros((num_data, num_clusters))
  ll = loglikelihood(data, weights, means, covariances)
  ll_trace = [ll]

  for it in range(maxiter):
    if it % 5 == 0:
      print("Iteration %s" % it)

    # E-step: compute responsibilities
    resp = compute_responsibilities(data, weights, means, covariances)

    # M-step
    # Compute the total responsibility assigned to each cluster, which will be useful when
    # implementing M-steps below. In the lectures this is called N^{soft}
    counts = compute_soft_counts(resp)

    # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
    # YOUR CODE HERE
    weights = compute_weights(counts)

    # Update means for cluster k using the M-step update rule for the mean variables.
    # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
    # YOUR CODE HERE
    means = compute_means(data, resp, counts)

    # Update covariances for cluster k using the M-step update rule for covariance variables.
    # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
    # YOUR CODE HERE
    covariances = compute_covariances(data, resp, counts, means)

    # Compute the loglikelihood at this iteration
    # YOUR CODE HERE
    ll_latest = loglikelihood(data, weights, means, covariances)
    ll_trace.append(ll_latest)

    # Check for convergence in log-likelihood and store
    if (ll_latest - ll) < thresh and ll_latest > -np.inf:
      break
    ll = ll_latest

  if it % 5 != 0:
    print("Iteration %s" % it)

  out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

  return out


##########################################################################################################################
# Testing the implementation on the simulated data

def generate_MoG_data(num_data, means, covariances, weights):
  """ Creates a list of data points """
  num_clusters = len(weights)
  data = []
  for i in range(num_data):
    #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
    k = np.random.choice(len(weights), 1, p=weights)[0]

    # Use np.random.multivariate_normal to create data from this cluster
    x = np.random.multivariate_normal(means[k], covariances[k])

    data.append(x)
  return data



# Model parameters
init_means = [
    [5, 0],  # mean of cluster 1
    [1, 1],  # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]],  # covariance of cluster 1
    [[.92, .38], [.38, .91]],  # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
init_weights = [1 / 4., 1 / 2., 1 / 4.]  # weights of each cluster


# Generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)

'''
# plotting
plt.figure()
d = np.vstack(data)
plt.plot(d[:, 0], d[:, 1], 'ko')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()
'''

np.random.seed(4)

# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1 / 3.] * 3

# Run EM
# results = EM(data, initial_means, initial_covs, initial_weights)

##########################################################################################################################


def plot_contours(data, means, covs, title):
  plt.figure()
  plt.plot([x[0] for x in data], [y[1] for y in data], 'ko')  # data

  delta = 0.025
  k = len(means)
  x = np.arange(-2.0, 7.0, delta)
  y = np.arange(-2.0, 7.0, delta)
  X, Y = np.meshgrid(x, y)
  col = ['green', 'red', 'indigo']
  for i in range(k):
    mean = means[i]
    cov = covs[i]
    sigmax = np.sqrt(cov[0][0])
    sigmay = np.sqrt(cov[1][1])
    sigmaxy = cov[0][1] / (sigmax * sigmay)
    Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
    plt.contour(X, Y, Z, colors=col[i])
    plt.title(title)
  plt.rcParams.update({'font.size': 16})
  plt.tight_layout()
  plt.show()


# Parameters after initialization
plot_contours(data, initial_means, initial_covs, 'Initial clusters')

# Parameters after 12 iterations
results = EM(data, initial_means, initial_covs, initial_weights, maxiter=12)  # YOUR CODE HERE
plot_contours(data, results['means'], results['covs'], 'Clusters after 12 iterations')

# Parameters after running EM to convergence
results = EM(data, initial_means, initial_covs, initial_weights)
plot_contours(data, results['means'], results['covs'], 'Final clusters')


# Plot the loglikelihood that is observed at each iteration
loglikelihoods = results['loglik']

plt.plot(range(len(loglikelihoods)), loglikelihoods, linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()
