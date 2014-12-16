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

   Assumes that this script has access to data/ folder with files
   - train-images-idx3-ubyte, train-labels-idx1-ubyte, 
   - t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte

   """

   if training:
      values = 'data/train-images-idx3-ubyte'
      labels = 'data/train-labels-idx1-ubyte'
   else:
      values = 'data/t10k-images-idx3-ubyte'
      labels = 'data/t10k-labels-idx1-ubyte'
   
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

   print("Loading data completed.\n")
   return images, labels

      
def filter_by_digit_mnist(digit, images, labels):
   """

   Note: This is an example function of what you can do 
   with the dataset of MNIST digits loaded in memory.

   Filter the array of images and return only those images 
   that have a label specified by parameter @digit

   """

   indexes = [i for i in range(len(labels)) if labels[i] == digit]
   return images[indexes]


def save_mnist_image(image, directory, filename):
   """

   Using numpy/matplotlib library, save the mnist
   train examples as actual images

   @directory - save image in this directory
   @filename - using this string as a filename

   """
   if not os.path.exists(directory):
      os.makedirs(directory)
   imshow(image.reshape((28,28)), cmap=cm.gray)
   axis('off')
   savefig(directory + os.sep + filename)



if __name__ == '__main__':
   """

   Example usage: 

   Load first 10 train images and leave only those 
   that display a digit 3, and save them as .png files 
   in the working directory.

   """

   images, labels = load_mnist(n_examples=10, training=True)
   filtered = filter_by_digit_mnist(3, images, labels)

   print("Saving images...")
   for i in range(len(filtered)):
      save_mnist_image(filtered[i], ".", "train" + str(i) + "filtered.png")
   print("Done!")


   



