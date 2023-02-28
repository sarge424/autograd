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

for i in range(1, 501):
    #forward pass
    ypred = [m(x) for x in xs]
    loss = sum([(yp - y)**2 for y, yp in zip(ys, ypred)])
    
    #backward pass
    for p in m.parameters():
        p.grad = 0.0
    loss.backward()

    #gradient descent
    for p in m.parameters():
        p.data += -0.01 * p.grad

    print(i,":", loss.data)
    
print([m(x).data for x in xs])