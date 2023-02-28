from engine import Value

x1 = Value(2.0, _label='x1')
x2 = Value(0.0, _label='x2')

w1 = Value(-3.0, _label='w1')
w2 = Value(1.0, _label='w2')

b = Value(6.8813735870195432, _label='b')

p = x1 * w1
q = x2 * w2

n = p + q + b
o = Value.tanh(n)

o.backward()
o.backward()
o.backward()

print('w1', w1, 'grad', w1.grad)
print('x1', x1, 'grad', x1.grad)
print()
print('w2', w2, 'grad', w2.grad)
print('x2', x2, 'grad', x2.grad)
print()
print('p', p.data, 'grad', p.grad)
print('q', q.data, 'grad', q.grad)
print()
print('o', o, 'data', o.data, 'grad', o.grad)