import torch
import numpy as np

a = torch.FloatTensor(3, 2)
print('a1\n')
print(a) #3x2 tensor with random(?) numbers

# to clear the tensor's content 
a.zero_() # operations with an underscore afterword are inplace
print('a2\n')
print(a)

# another way to create a tensor is to provide an iteratble
b = torch.FloatTensor([[1,2,3], [3,2,1]])
print('b\n')
print(b)

# np is familiar
n = np.zeros(shape=(3, 2))
print(type(n))

n = torch.tensor(n)
print(n)
print(type(n))

# above creates a double precision tensor which is not required
# instead specify 32 or 16 bit
n = np.zeros(shape=(3,2), dtype=np.float32)
torch.tensor(n)

# or
n = np.zeros(shape=(3,2))
torch.tensor(n, dtype=torch.float32)

# scalar tensors
c = torch.tensor([1,2,3])
print(c)

s = c.sum()
print(s)

print(s.item())

torch.tensor(1)
