import axon

def test_tensor():
    t1 = axon.tensor([[1, 2, 3], 
                      [5, 4, 6]], requires_grad=True)

    t2 = axon.tensor([1, 2, 3], requires_grad=True)

    assert t1.requires_grad
    assert t2.requires_grad

    print(t1)
    print(t2)

