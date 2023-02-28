from engine import Value
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        act = sum([wi * xi for (wi, xi) in zip(self.w, x)], self.b)
        out = act.tanh()
        return out
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out
    
class MLP:
    def __init__(self, nin, nouts):
        dims = [nin] + nouts
        self.layers = [Layer(dims[i], dims[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        x = [val.data for val in x]
        return x[0] if len(x) == 1 else x