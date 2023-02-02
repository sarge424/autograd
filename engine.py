class Value:
    def __init__(self, data, _op='', _children=()):
        self.data = data
        self.grad = 0.0
        
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

        
    def __repr__(self):
        return f'({self._op}){self.data}|{self.grad}'
    
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