#!/s/python-3.4.1/bin/python3.4

import struct
from pylab import *
from array import array
import numpy as np
import os


def load_mnist(n_examples=0, training = True):
   """ 

   Load MNIST images and labels into memory

   @training - set this to True if you want to load the train set
   @n_examples - set this to zero if you want to load all data points

   @return images: np-array of dims [n_examples] x [n_rows * n_cols]
           labels: np-array of dims [n_examples] x 1

   Assumes that this script is in the same folder as files
   - train-images-idx3-ubyte, train-labels-idx1-ubyte, 
   - t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte

   """

   if training:
      values = 'train-images-idx3-ubyte'
      labels = 'train-labels-idx1-ubyte'
   else:
      values = 't10k-images-idx3-ubyte'
      labels = 't10k-labels-idx1-ubyte'
   
   print("Trying to open datafiles...")
   with open(values, "rb") as f:
      magic_number,n_images,n_rows,n_columns = struct.unpack('>iiii', f.read(16))

      if (n_examples == 0 or n_examples > n_images):
         n_examples = n_images # load all examples

      print("Loading", n_examples, "examples from", values, "file...") 

      raw = array("B", f.read(int(n_rows * n_columns * n_examples)))
      images = np.zeros((n_examples, int(n_rows * n_columns)), dtype=np.uint8)

      for i in range(n_examples):
         start = int(i * n_rows * n_columns)
         end = int((i+1) * n_rows * n_columns)
         images[i] = np.array(raw[start : end])
      
      images = np.true_divide(images, 255) # all features between 0 and 1

   with open(labels, "rb") as f:
      magic_number,n_labels = struct.unpack('>ii', f.read(8))

      print("Loading", n_examples, "labels from", labels, "file...") 

      raw = array("B", f.read(int(n_examples))) 
      labels = np.array(raw, dtype=np.uint8)

   print("Loading completed.\n")
   return images, labels

      
def filter_by_digit_mnist(digit, images, labels):
   """

   Fitler the array of images and return only those that
   have a label specified by parameter @digit

   """

   indexes = [i for i in range(len(labels)) if labels[i] == digit]
   filtered = images[indexes]
   return filtered


def save_mnist_image(image, directory, filename):
   if not os.path.exists(directory):
      os.makedirs(directory)
   imshow(image.reshape((28,28)), cmap=cm.gray)
   axis('off')
   savefig(directory + os.sep + filename)


def load_ads():
   # 3279 examples and 1558 features for each example
   data = np.zeros((3279, 1558))
   labels = []
   i = 0
   for line in open("ad.arff", "r"):
      line = line.strip()
      if len(line) == 0: continue
      if line.startswith("@"): continue

      tokens = line.split(",")
      
      for j in range(len(tokens)-1):
         data[i][j] = float(tokens[j])

      labels.append(tokens[-1])
      i+=1

   # make sure that all values are between 0 and 1
   col1_max = data[:,0].max()
   col2_max = data[:,1].max()
   data[:,0] = np.true_divide(data[:,0], col1_max)
   data[:,1] = np.true_divide(data[:,1], col2_max)

   return data, labels

if __name__ == '__main__':
   """

   As a test case, load first 10 train images and
   leave only those that have a digit 3,
   and save them as .png files in the working dir

   """

   images, labels = load_mnist(training=True, n_examples=10)
   filtered = filter_by_digit_mnist(3, images, labels)

   for i in range(len(filtered)):
      save_mnist_image(filtered[i], ".", "train" + str(i) + "filtered.png")


   



