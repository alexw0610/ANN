# Artificial Neural Network

![Alt text](screenshot.png?raw=true "Screenshot of a visual representation of a ANN.")

This repository contains a no dependency java implementation of an artifical neural net (ANN). The forward-pass and backpropagation are implemented using 2d and 3d arrays instead of a matrix implementation.
This choice was made to give the user an alternative in trying to understand backpropagation without the essentials being hidden behind a matrix implementation. 
Note that this implementation is not designed for performance but rather serves as a showcase of a ANN for people that try to understand how simple neural networks fundamentally work.

The implementation contains methods for persisting and loading the trained weights to disk. The training is done by supplying two 2d arrays of data containing matching input and solution data for the network to train with. The ANN constructor allows for variable amounts and sizes of layers.

The current implementation uses a sigmoid activation function but changing the relevant method to use any other desired activation function should be no problem for the user.

The branch "master" contains a neural net setup with an XOR example.
The branch "mnistDigit" contains a working example of a neural network that classifies the handwritten digits [0-9] from the mnist database [1].
The branch "visual" includes helper classes to utilize OpenGL to display the current state of any given ANN object.


[1] http://yann.lecun.com/exdb/mnist/
