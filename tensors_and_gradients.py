"""
PyTorch tensors have built-in gradient calculation and tracking machinery, so all you need
to do is to convert the data into tensors and perform computations using the tensor's methods 
and functions provided by torch.

Every tensor has several attributes related to gradients
* grad
    Property which holds a tensor of the same shape containing computed gradients.

* is_leaf
    True, if this tensor was constructed by the user and False, if the object is a result of function transformation

* requires_grad
    True, if this tensor requires gradients to be calculated.
    This property is inherited from leaf tensors, which get this value from the the tensor construction step
    By default this is False
"""
import torch

v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])

v_sum = v1 + v2
v_res = (v_sum * 2).sum()

print('v_res', v_res)

print('isleaf v1',v1.is_leaf,'v2', v2.is_leaf)

print('is leaf v_sum', v_sum.is_leaf,'v_res', v_res.is_leaf)

print('v1 requires_grad', v1.requires_grad)

print('v2 requires_grad', v2.requires_grad)

# Now let's tell PyTorch to calculate the gradients of our graph

print('\n.backward() called here\n')
v_res.backward()

print('v1.grad', v1.grad)

print('v2.grad', v2.grad)
