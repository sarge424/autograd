from engine import Value
import nn

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

m = nn.MLP(3, [4, 4, 1])
ypred = [m(x) for x in xs]

print(ypred)