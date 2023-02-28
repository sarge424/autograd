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

i = 0
while True:
    i+=1
    ypred = [m(x) for x in xs]

    loss = sum([(yp - y)**2 for y, yp in zip(ys, ypred)])
    #for p in m.parameters():
        #print("{:.4f} {:.4f}".format(p.data, p.grad))
        
    #print()
    for p in m.parameters():
        p.grad = 0.0

    loss.backward()

    for p in m.parameters():
        p.data += -0.01 * p.grad

    print(i,":", loss.data)
    q = input()