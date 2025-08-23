import axon


def test_compilation():

    @axon.compile
    def test_func(a, b):
        return a + b

    r = test_func(1, 2)
