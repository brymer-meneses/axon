import axon

def test_tensor_add() -> None:

    t1 = axon.tensor([1, 2, 3], requires_grad=True)
    t2 = axon.tensor([1, 2, 3], requires_grad=True)

    # t3 = t1 + t2

    print(t1)
    print(t2)

