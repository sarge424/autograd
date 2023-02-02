class Value:
    def __init__(self, data, _op='', _children=()):
        self.data = data
        self.grad = 0.0
        
        self._op = _op
        self._prev = set(_children)
        self.backward = lambda: None

        
    def __repr__(self):
        return f'({self._op}){self.data}|{self.grad}'
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, _op='+', _children=(self, other))
        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out.backward = _backward
        
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, _op='*', _children=(self, other))
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out.backward = _backward
        
        return out
    
    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other * self