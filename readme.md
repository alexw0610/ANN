#Artifical Neural Network

This repository contains a no dependency java implementation of an artifical neural net. The forward-pass and backpropagation are implemented using 2d and 3d arrays instead of the popular alternative of a matrix implementation.
This choice was made to give the user an alternative in trying to understand backpropagation without the essentials being hidden behind a matrix implementation. 
Note that this implementation is not meant to compete with the performance of state of the art neural networks but rather serves as a showcase of a neural net for people that try to understand how neural networks fundamentaly work.

The implementation contains methods for persisting and loading the trained weights to disk. The training is done by supplying two 2d arrays of data containing matching input and solution data for the network to train with. 

The current implementation uses a sigmoid activation function but changing the relevant method to use any other desired activation function should be no problem for the user.

The branch "master" contains a psvm with an XOR example.
The branch "mnistDigit" contains a working example of a neural network that classifies the handwritten digits [0-9] from the mnist database [1].



[1] http://yann.lecun.com/exdb/mnist/