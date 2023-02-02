from engine import Value

x = Value(10, _label='x')
y = Value(20, _label='y')

z = 2*(x**2) + 4*(x*y) + 3

print(x)
print(y)
print(z)
print()
 
z.grad = 1.0
z._backward()
print(x)
print(y)
print(z)
