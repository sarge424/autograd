from engine import Value

x = Value(4)
y = Value(3)

z = 2 * x

print(x)
print(y)
print(z)

z.grad = 1.0
z.backward()

print(x)
print(y)
print(z)