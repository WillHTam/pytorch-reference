"""
Loss functions are implemented as an nn.Module subclass
Usually accpets two arguments: output(prediction) and desired output(labels)

Most common are 
    nn.MSELoss for mse
    nn.BCELoss for binary cross-entropy loss 
        This is for single probability value ie usually Sigmoid layer output
    nn.BCEWithLogits assumes raw scores and applies Sigmoid itself
        Above two used for binary classification
    nn.CrossEntropyLoss and nn.NLLLoss - "maximum likelihood" criteria 
        for multi-class classification problems. Fromer expects raw scores
        and then applies LogSoftmax by itself, second expects log probabilities
        as the input

Optimizers - takes gradients of model parameters and change these parametesr
    and change these parameters with the goal of decreasing loss.  

    Package is `torch.optim`  On contruction, need to pass iterable
    of 'Variables' which will be modified during the optimization process.
    Usually pass the result of params() call from nn.Module

    Line 2&3: making the data samples and target labels into Torch tesnors
    Line 4: Passing data samples to the network
    Line 5: Feed output and target labels to loss function

    Line 6: Since each tensor in this computation graph remembers its parent, calculate gradients
    for the whole network by calling backward() on a loss function result. 
    The result of this call will be the unrolling of the graph of the performed computations and the 
    calculating of gradients for every leaf tensor with require_grad=True. 
    Usually, such tensors are our model's parameters, such as weights and biases of feed-forward networks, 
    and convolution filters.

    Line 7: After loss.backward() is finished, gradients are acculmulated.  step() is the
    optimizer, taking all gradients from the params and applying them

    Line 8: Zero the gradients of parameters
"""

for batch_samples, batch_labels in iterate_batches(data, batch_size=32): #1
    batch_samples_t = torch.tensor(batch_samples) #2
    batch_labels_t = torch.tensor(batch_labels) #3 
    out_t = net(batch_samples_t) #4 
    loss_t = loss_function(out_t, batch_labels_t) #4
    loss_t.backward() #5 
    optimizer.step() #6
    optimizer.zero_grad() #7
