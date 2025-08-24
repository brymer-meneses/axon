import axon


def test_compilation():

    @axon.compile
    def test_func(a, b):
        l = a * b
        l.backward()

    a = axon.tensor([1, 2, 3], requires_grad=True)
    b = axon.tensor([1, 2, 3], requires_grad=True)
    r = test_func(a, b)
    expected_ir = \
"""module {
  func.func @graph(%arg0: !axon.tensor_ref<3xf32, requires_grad>, %arg1: !axon.tensor_ref<3xf32, requires_grad>) {
    %0 = axon.get_data %arg0 : !axon.tensor_ref<3xf32, requires_grad> -> tensor<3xf32>
    %1 = axon.get_data %arg1 : !axon.tensor_ref<3xf32, requires_grad> -> tensor<3xf32>
    %2 = axon.mul %0, %1 : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
    %3 = axon.fill_like %2, 1.000000e+00 : tensor<3xf32> -> tensor<3xf32>
    %4 = axon.mul %3, %1 : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
    %5 = axon.mul %3, %0 : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
    axon.accumulate_grad %arg0, %4 : !axon.tensor_ref<3xf32, requires_grad>, tensor<3xf32>
    axon.accumulate_grad %arg1, %5 : !axon.tensor_ref<3xf32, requires_grad>, tensor<3xf32>
    return
  }
}
"""

    assert test_func.dump_ir() == expected_ir
