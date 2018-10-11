import torch
import torch.nn as nn

class OurModule(nn.Module):
    """
    Subclassing nn.Module allows for extended customization and stacking of NN layers.
    Using the parameters() method shows a module's params
    Zero gradients with zero_grads()
    Establishes the convention of module application to data.
        Every module needs to performs its data transformation in the forward() method
        by overriding it.
    
    Allows nesting of submodels into higher-level models

    This class inherits nn.Module, with three params:
        Input size, output size, and optional dropout probability
    After initializing, call nn.Sequential and assign it to pipe
        This automatically registesr this module
    """
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax()
        )
    
    def forward(self, x):
        """
        This function manually registers submodules.

        Here, we override the forward function with our implementation of data 
            transformation. As our module is a very simple wrapper around other layers, 
            we just need to ask them to transform the data. 

        Note that to apply a module to the data, you need to call the module as 
            callable (that is, pretend that the module instance is a function and 
            call it with the arguments) and not use the forward() function of the 
            nn.Module class. This is because nn.Module overrides the __call__() method, 
            which is being used when we treat an instance as callable. 
            This method does some nn.Module magic stuff and calls your forward() method. 
            If you call forward() directly, you'll intervene with the nn.Module duty, 
            which can give you wrong results.
        """
        return self.pipe(x)

# Below create module and provide it with args
# Create tensor, wrapped into variabe and ask module to transform it 
# nn.Module overrides __str__() and __repr__() to have a nice print structure
if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    v = torch.FloatTensor([[2,3]])
    out = net(v)
    print(net)
    print(out)

