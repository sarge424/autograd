from engine import Value

x = Value(4)
y = Value(3)

z = x**2

print(x)
print(y)
print(z)
print()
 
z.grad = 1.0
z._backward()
print(x)
print(y)
print(z)
