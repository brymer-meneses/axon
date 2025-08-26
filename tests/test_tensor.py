import axon
import inspect

def check_ir(func):
    expected_ir = inspect.cleandoc(func.__doc__)
    assert func.dump_ir().strip() == expected_ir.strip()

@axon.jit()
def tensor_mul(a, b):
    """
    module {
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
    l = a * b
    l.backward()

@axon.jit()
def tensor_add(a, b):
    """
    module {
      func.func @graph(%arg0: !axon.tensor_ref<3xf32, requires_grad>, %arg1: !axon.tensor_ref<3xf32, requires_grad>) {
        %0 = axon.get_data %arg0 : !axon.tensor_ref<3xf32, requires_grad> -> tensor<3xf32>
        %1 = axon.get_data %arg1 : !axon.tensor_ref<3xf32, requires_grad> -> tensor<3xf32>
        %2 = axon.add %0, %1 : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
        %3 = axon.fill_like %2, 1.000000e+00 : tensor<3xf32> -> tensor<3xf32>
        axon.accumulate_grad %arg0, %3 : !axon.tensor_ref<3xf32, requires_grad>, tensor<3xf32>
        axon.accumulate_grad %arg1, %3 : !axon.tensor_ref<3xf32, requires_grad>, tensor<3xf32>
        return
      }
    }
    """
    l = a + b
    l.backward()

def test_jit():
    a = axon.tensor([1, 2, 3], requires_grad=True)
    b = axon.tensor([1, 2, 3], requires_grad=True)

    tensor_mul(a, b)
    tensor_add(a, b)

    check_ir(tensor_mul)
    check_ir(tensor_add)
