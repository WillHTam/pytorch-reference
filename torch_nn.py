"""
torch.nn has predefined building blocks for making an nn

All modules are callable, which means that the instance of any class can act as a function when applied to its arguments.

Linear class implements a feed-forward layer with optional bias
"""
import torch
import torch.nn as nn

print('Linear Model')
# Here a randomly initialized feed-forward layer with 2 inputs and five outputs appplied to a float tensor
l = nn.Linear(2, 5)
v = torch.FloatTensor([1, 2])
print(l(v), '\n')

print('Sequential Model')
# Three-layer NN with softmax on output applied on dimension 1 (dimension 0 is batch samples)
# and ReLU nonlinearities and dropout
s = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1)
)
print(s)
# send a 'minibatch' through
print(s(torch.FloatTensor([[1,2]])))


