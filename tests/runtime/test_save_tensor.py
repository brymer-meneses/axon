from axon import Tensor


def test_save_tensor_materializes_saved_value_and_stays_lazy_otherwise():
    a = Tensor.randn((10, 10))
    b = a * a
    b.save()

    # Compute on the same trace so materializing the returned tensor should
    # also materialize the explicitly saved tensor.
    c = b**2.0
    d = c.mean()
    e = d.log()

    assert not e.is_evaluated
    assert not b.is_evaluated
    assert not d.is_evaluated

    e.evaluate()

    assert e.is_evaluated
    assert b.is_evaluated
    assert not d.is_evaluated

    # Re-evaluating an already materialized tensor is a no-op.
    e.evaluate()
    assert e.is_evaluated


def test_save_tensor_does_not_materialize_independent_branches():
    a = Tensor.randn((10, 10))
    b = a * a
    b.save()

    # Force a separate trace that doesn't touch b again; b should remain lazy.
    detached = (a**2.0).mean()
    detached.evaluate()

    assert detached.is_evaluated
    assert not b.is_evaluated


def test_save_tensor_can_be_reused_in_new_traces_without_leaking_state():
    a = Tensor.randn((10, 10))
    b = a * a
    b.save()

    # First branch: materialize b via a downstream computation.
    first_branch = (b**2.0).mean().log()
    first_branch.evaluate()

    assert b.is_evaluated
    assert first_branch.is_evaluated

    # Second branch should succeed and leave intermediates lazy.
    second_mean = (b + b).mean()
    second_branch = second_mean.log()
    assert not second_mean.is_evaluated
    assert not second_branch.is_evaluated

    second_branch.evaluate()

    assert b.is_evaluated
    assert not second_mean.is_evaluated
    assert second_branch.is_evaluated
