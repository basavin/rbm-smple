## Overview

This is a simple implementation of RBM learning algorithm written in `Python` which uses `numpy` package to speed up matrix calculations.

## Directions

To run RBM learning algorithm on MNIST dataset with default parameters, type

`python rbm.py`

Otherwise you can specify all parameters yourself

`python rbm.py <num_examples> <num_hidden_units> <num_epochs> <learn_rate>`

This will train an RBM on training examples and then try to reconstruct the test images given the learned weights. Both original test images and the reconstructed test images will then be saved in `Output/` folder.

## Dependencies

Please note this project requires Python 3.x to run. It also requires the `numpy` package.
