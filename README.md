# ArtificialNeuralNetwork

This project is a sandbox based on this article : https://rubikscode.net/2022/07/04/implementing-simple-neural-network-in-c/
One good TL;DR of ANN can be found here : https://www.linkedin.com/pulse/functions-weights-bias-artificial-neural-networks-doug-rose/

# Model

````
[Neuron] ------Synapse ------- [Neuron]
````

## Synapse
A neuron is connected to another through a synapse.
On the target neuron's perspective, the incoming synapse has a weight.

## Neuron

A neuron has 0..n parent neurons.
A neuron has 0..n descendant neurons.

## Layers

Each series of neurons on the same level is called a layer.
Layers may have different sizes, and sizes do not need to increase nor decrease.

````
Layer 0		Layer 1		...		Layer j  
-------------------------------------------  
n(0,0)		n(1,0)		...		n(j,0)	
n(0,1)		n(1,1)		...		n(j,1)  
n(0,2)		n(1,2)		...		...  
...			...			...  
...						...  
						...  
````

## Neural Network

Each neuron of the last layer is connected to each layer of the previous layer, and so on.

# Functions

## Input function

A neuron receives an input from each synapse connected to it.
Each synapse has a weight (a coefficient).

The input function of a neuron is the weighted sum of all the outputs of the parent neurons :

For the neuron k on the layer n, having j synapses connected to it, the input function is :

````
a(k,n) = w0 * a(0, n-1) + w1 * a(1, n-1) + ... + wj * a(j, n-1)
````

## Activation function

The neuron sends the result of the input function to an activation function.
The role of this function is to activate or not the output, given the value of the input.

It can be seen as a gate with a threshold (step activation function), or a projection of the real numbers to the interval [0,1] (sigmoid)...

# C# implementation

## Model

The model is inspired by the article linked in the abstract of this document. It aims to represent the ANN in the object oriented programmation's perspective, and as such, will not
be using matricial calculus, even though matrix-based implementations would be way much faster and efficient than the one in this project.  

## Limits / missing

One common implementation mentions 'bias' (plural : biases) as a part of the neuron modelling. The bias is a number variable held by the neuron,
and the bias is added to the result of the input function.

This notion is missing from the article used as a source for this implementation, but might be easily added.