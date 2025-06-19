import axon

with axon.Context() as ctx:
    t1 = axon.tensor([[1, 2, 3], 
                      [5, 4, 6]], requires_grad=True)
    print(t1)

