import math 

class Value:
    def __init__(self, data, _op='', _label='', _children=()):
        self.data = data
        self.grad = 0.0
        
        self.label = _label
        
        self._op = _op
        self._prev = set(_children)
        self._getstr = lambda: None
        self._backward = lambda: None        
    
    def backward(self):
        self.grad = 1.0
        self._recback()
        
    def zero_grad(self):
        self.grad = 0.0
        for p in self._prev:
            p.zero_grad()
    
    def _recback(self):
        self._backward()
        for p in self._prev:
            p._recback()
    
    def tanh(self):
        num = math.exp(self.data) - math.exp(-self.data)
        den = math.exp(self.data) + math.exp(-self.data)
        out = Value(num / den, _op='tanh', _children=(self,))
        
        def _backward():
            self.grad += 1 - (num/den)**2
        out._backward = _backward
        
        return out
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, _op='+', _children=(self, other))
        
        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, _op='*', _children=(self, other))
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)) #allow only floats and ints
        out = Value(self.data ** other, _op='^', _children=(self,))
        def _backward():
            self.grad += other * self.data**(other-1)
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __radd__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return (self**-1) * other
    
    def __repr__(self):
        return f'Value(data={self.data})'