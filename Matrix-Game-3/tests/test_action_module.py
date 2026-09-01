import torch

from wan.modules.action_module import WanRMSNorm


def test_wan_rms_norm_applies_learned_weight():
    norm = WanRMSNorm(4, eps=1e-6)
    inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    unit_weight_output = norm(inputs).detach()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 1.25, 1.5, 2.0]))

    output = norm(inputs)
    expected = torch.nn.functional.rms_norm(inputs.float(), (norm.dim,), eps=norm.eps)
    expected = expected.to(inputs.dtype) * norm.weight

    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, unit_weight_output)

    output.sum().backward()
    assert norm.weight.grad is not None
    assert torch.count_nonzero(norm.weight.grad) == norm.weight.numel()
