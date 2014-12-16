"""

Convert my data (represented as numpy matrix [num_ex] x [features])
into the arff format. Class will be the last attribute.

This is useful when we want to feed the new representation of
the data produced by an RBM into another machine learning toolkit
like Weka.

"""

from loader import load_mnist, load_ads
from rbm import sample_hidden
import numpy as np


# TODO: right now it only works for MNIST data set due to hard-coded class values
def convert_to_arff(images, labels, filename, class_string):
   f = open(filename,'w')
   f.write('@RELATION DATA=t10k-images.idx3-ubyte-LABELS=t10k-labels.idx1-ubyte\n\n')
   for i in range(len(images[0])):
      f.write('@ATTRIBUTE pixel' + str(i+1) + '\treal\n')
   f.write('@ATTRIBUTE class\t' + class_string + '\n\n')
   f.write('@DATA\n')
   for j in range(len(images)):
      if j % 10000 == 0 and j != 0:
         print('Saved', j, 'converted examples so far...')
      image = images[j]
      output = ''
      for i in range(len(image)):
         if image[i] == 0.0:
            #compress for space
            output += str(0)
         else:
            output += str(image[i])
         output += ','
      f.write(output + str(labels[j]) + '\n')
   print("Done!")
   f.close()



def test_ads():
   # load original dataset
   data, labels = load_ads()

   # convert input data into the RBM's internal representation
   print("Converting the original data into new representation...")
   w = np.load('Output/w_v3279_h1000.npy')
   b = np.load('Output/b_v3279_h1000.npy')
   h = sample_hidden(data,w,b)
   print("Saving the converted dataset...")
   convert_to_arff(h, labels, 'converted.arff', '{ad, noad}')


def test_mnist():
   # load original dataset
   images, labels = load_mnist(n_examples=0, training = False)

   # convert input data into the RBM's internal representation
   print("Converting the original data into new representation...")
   w = np.load('Output/w_v60000_h500.npy')
   b = np.load('Output/b_v60000_h500.npy')
   h = sample_hidden(images,w,b)
   print("Done!")

   # save the converted data as arff file
   print("Saving the converted dataset...")
   convert_to_arff(h, labels, 'converted.arff', '{0,1,2,3,4,5,6,7,8,9}')



if __name__ == '__main__':
   test_ads()
    
