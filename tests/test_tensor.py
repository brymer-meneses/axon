import axon

def test_compilation():

    @axon.compile
    def test_func(a, b):
        l = a * b
        l.backward()

    a = axon.tensor([1, 2, 3], requires_grad=True)
    b = axon.tensor([1, 2, 3], requires_grad=True)
    r = test_func(a, b)
    print(test_func)
