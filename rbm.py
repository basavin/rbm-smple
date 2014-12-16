#!/s/python-3.4.1/bin/python3.4

"""

In this file, I provide an implementation of a Restricted
Botlzmann machine with real-valued input units (of range [0,1]).

The notation and the training procedure followes the Hinton's paper
"A Practical Guide to Training Restricted Boltzmann Machines", 2010.

"""

from loader import load_mnist, load_ads, save_mnist_image
from time import time
import numpy as np
import os


def rbm(dataset, num_hidden, learn_rate, epochs, batchsize):
   """

   Train a Restricted Boltzmann Machine where input can be
   either binary or real-valued in the interval [0,1].
   For training, we split the dataset into several smaller 
   mini-batches. We also use momentum when updating weights.

   @dataset - a numpy array of dims [num_examples] x [num_features]

   @returns w: edge weights   of dims [num_visible] x [num_hidden]
            a: visible biases of dims [num_visible] x 1
            b: hidden biases  of dims [num_hidden]  x 1

   """
   
   num_visible = dataset.shape[1]
   num_examples = dataset.shape[0]
   
   print("Training RBM with", num_visible, "visible units,", 
          num_hidden, "hidden units,", num_examples, "examples and", 
          epochs, "epochs...")

   start_time = time()

   batches = num_examples // batchsize
   
   w = 0.1 * np.random.randn(num_visible, num_hidden)
   a = np.zeros((1, num_visible))
   b = -4.0 * np.ones((1, num_hidden))

   w_inc = np.zeros((num_visible, num_hidden))
   a_inc = np.zeros((1, num_visible))
   b_inc = np.zeros((1, num_hidden))

   for epoch in range(epochs):
      error = 0
      for batch in range(batches):
         #### --- Positive phase of contrastive divergence --- ####

         # get next batch of data
         v0 = dataset[int(batch*batchsize):int((batch+1)*batchsize)]

         # in this matrix, m[i,j] is prob h[j] = 1 given example v[i]
         # dims: [num_ex] x [num_hidden]
         prob_h0 = logistic(v0, w, b)
    
         # sample the states of hidden units based on prob_h0
         h0 = prob_h0 > np.random.rand(batchsize, num_hidden)

         # positive phase products
         vh0 = np.dot(v0.T, prob_h0)

         # activation values needed to update biases
         poshidact = np.sum(prob_h0, axis=0)
         posvisact = np.sum(v0, axis=0)

         #### --- Negative phase of contrastive divergence --- ####

         # reconstruct the data by sampling the visible states from hidden states
         v1 = logistic(h0, w.T, a)

         # sample hidden states from visible states
         prob_h1 = logistic(v1, w, b)

         #negative phase products
         vh1 = np.dot(v1.T, prob_h1)

         # activation values needed to update biases
         neghidact = np.sum(prob_h1, axis=0)
         negvisact = np.sum(v1, axis=0)

         #### --- Updating the weights --- ####

         # set momentum as per Hinton's practical guide to training RBMs
         m = 0.5 if epoch > 5 else 0.9

         # update the weights
         w_inc = w_inc * m + (learn_rate/batchsize) * (vh0 - vh1)
         a_inc = a_inc * m + (learn_rate/batchsize) * (posvisact - negvisact)
         b_inc = b_inc * m + (learn_rate/batchsize) * (poshidact - neghidact)

         a += a_inc
         b += b_inc
         w += w_inc

         error += np.sum((v0 - v1) ** 2)

      print("Epoch %s completed. Reconstruction error is %0.2f. Time elapsed (sec): %0.2f" 
            % (epoch + 1, error, time() - start_time))

   print ("Training completed.\nTotal training time (sec): %0.2f \n" % (time() - start_time))
   return w, a, b


def logistic(x,w,b):
   xw = np.dot(x, w)
   replicated_b = np.tile(b, (x.shape[0], 1))

   return 1.0 / (1 + np.exp(- xw - b))

      
def reconstruct(v0, w, a, b):
   """

   Reconstruct the input vector v0 by sampling hidden states
   and then sampling visible states from the hidden states.
   This is useful for measuring the reconstruction error, which
   can be viewed as a goodness of learned RBM.

   @v0 - original data
   @w - weights, a - visible biases, b - hidden biases

   @returns - v1 - reconstructed state of visible units 

   """

   num_hidden = w.shape[1]
   prob_h0 = logistic(v0, w, b)
   h0 = prob_h0 > np.random.rand(1, num_hidden)

   return logistic(h0, w.T, a)

def sample_hidden(v0,w,b):
   """

   Sample a vector of hidden units given the visible vector
   and the trained weights. This function is used to convert
   the initial representation into the representation given
   by hidden units.

   @v0 - data (can be a single vector or a matrix of n vectors)
   @w - weights, b - hidden biases

   @returns - h0 - sampled states of hidden units

   """
   num_hidden = w.shape[1]
   return logistic(v0, w, b)


def save_weights(w, a, b, directory, n_examples, num_hidden):
   if not os.path.exists(directory):
      os.makedirs(directory)

   w_name = directory + os.sep + "w_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(w_name, w)
   a_name = directory + os.sep + "a_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(a_name, a)
   b_name = directory + os.sep + "b_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(b_name, b)




def test_ads():
   data, labels = load_ads()
   print(data[0])
   num_hidden = 1000
   n_examples = 3279
   w, a, b  = rbm(data, num_hidden, learn_rate = 0.001, epochs = 10, batchsize = 100)
   print("Saving weights...")
   save_weights(w, a, b, "Output", n_examples, num_hidden)

def test_mnist():
   """

   As a test case, load train set images, then train RBM,
   then use several test set images to measure the
   reconstruction error on the unseen data points.
   Fnally, save the reconstructed images and learned weights 
   in the "Output" folder.

   """

   # load data
   n_examples = 10000
   images, labels = load_mnist(n_examples, training = True)

   # train one layer of RBM
   num_hidden = 500
   w, a, b  = rbm(images, num_hidden, learn_rate = 0.1, epochs = 50, batchsize = 100)

   # save all weights
   print("Saving weights...")
   save_weights(w, a, b, "Output", n_examples, num_hidden)
   
   # try to reconstruct some test set images
   print("Generating and saving the reconstructed images...")
   images, labels = load_mnist(10, training = False)
   for i in range(10):
      data = images[i]
      save_mnist_image(data, "Output", str(i) + "original.png")
      data = reconstruct(data, w, a, b)
      save_mnist_image(data, "Output", str(i) + "reconstructed.png")
   print("Done!")


if __name__ == '__main__':
   test_mnist()

