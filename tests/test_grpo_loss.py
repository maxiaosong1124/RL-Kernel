# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.grpo_loss import NativeGRPOLossOp
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp
from rl_engine.kernels.ops.triton.triton_grpo_loss import TritonGRPOLossOp
from rl_engine.testing import (
    compute_policy_ratio,
    compute_reference_kl,
    make_synthetic_rl_kernel_batch,
    masked_mean,
    selected_logprobs_reference,
)

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton GRPO loss requires a CUDA device and Triton.",
)

_NUM_PROMPTS = 3
_SPP = 4
_COMP_LEN = 6
_VOCAB = 64


# Shared helpers
def _batch(seed=0, *, device="cpu", valid_density=0.9):
    return make_synthetic_rl_kernel_batch(
        num_prompts=_NUM_PROMPTS,
        samples_per_prompt=_SPP,
        prompt_len=0,
        completion_len=_COMP_LEN,
        vocab_size=_VOCAB,
        valid_density=valid_density,
        device=device,
        seed=seed,
    )


def _logits_like(batch, seed, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch.batch_size, batch.completion_len, _VOCAB, generator=gen, device=device)


def _current_logps(batch, seed, *, device="cpu"):
    """A realistic per-token current-policy logp derived from synthetic logits."""
    logits = _logits_like(batch, seed, device=device)
    return selected_logprobs_reference(logits, batch.token_ids)


def _reference_group_advantages(rewards, samples_per_prompt, eps=1e-6):
    """Mirror of examples.grpo_single_gpu.make_group_advantages normalization."""
    grouped = rewards.view(-1, samples_per_prompt)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return ((grouped - group_mean) / group_std).reshape(-1)


def _reference_loss(current_logps, old_logps, ref_logps, advantages, mask, clip_eps, beta):
    """Mirror of examples.grpo_single_gpu.grpo_loss using the testing helpers."""
    ratio = compute_policy_ratio(current_logps, old_logps, mask)
    unclipped = ratio * advantages.float()
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.float()
    policy_loss_terms = -torch.minimum(unclipped, clipped)
    kl_terms = compute_reference_kl(current_logps, ref_logps, mask)
    policy_loss = masked_mean(policy_loss_terms, mask)
    kl = masked_mean(kl_terms, mask)
    return policy_loss + beta * kl, policy_loss, kl


# pure-PyTorch reference op
def test_group_advantages_matches_reference_uniform():
    op = NativeGRPOLossOp()
    rewards = _batch(seed=1).rewards
    got = op.group_advantages(rewards, samples_per_prompt=_SPP)
    expected = _reference_group_advantages(rewards, samples_per_prompt=_SPP)
    assert torch.allclose(got, expected, atol=1e-6)


def test_boundaries_agree_with_uniform():
    op = NativeGRPOLossOp()
    rewards = _batch(seed=2).rewards
    base = op.group_advantages(rewards, samples_per_prompt=_SPP)
    bounds = list(range(0, _NUM_PROMPTS * _SPP + 1, _SPP))
    by_bounds = op.group_advantages(rewards, group_boundaries=bounds)
    assert torch.allclose(base, by_bounds, atol=1e-6)


def test_variable_group_boundaries():
    op = NativeGRPOLossOp()
    rewards = torch.tensor([1.0, 3.0, 10.0, 20.0, 30.0])
    got = op.group_advantages(rewards, group_boundaries=[0, 2, 5])
    # Group 0: [1, 3] -> mean 2, std 1 -> [-1, 1]
    g0 = torch.tensor([-1.0, 1.0])
    # Group 1: [10, 20, 30] -> mean 20, std sqrt(200/3)
    g1 = (torch.tensor([10.0, 20.0, 30.0]) - 20.0) / (200.0 / 3.0) ** 0.5
    expected = torch.cat([g0, g1])
    assert torch.allclose(got, expected, atol=1e-5)


def test_forward_loss_matches_reference():
    op = NativeGRPOLossOp()
    batch = _batch(seed=0)
    current = _current_logps(batch, seed=100)
    clip_eps, beta = 0.2, 0.01

    loss, policy_loss, kl = op.forward(
        current,
        batch.old_logps,
        batch.ref_logps,
        batch.rewards,
        batch.completion_mask,
        clip_eps=clip_eps,
        beta=beta,
        samples_per_prompt=_SPP,
    )

    sample_adv = _reference_group_advantages(batch.rewards, samples_per_prompt=_SPP)
    adv_tokens = (
        sample_adv[:, None]
        .expand_as(batch.completion_mask)
        .clone()
        .masked_fill(~batch.completion_mask, 0.0)
    )
    exp_loss, exp_policy, exp_kl = _reference_loss(
        current, batch.old_logps, batch.ref_logps, adv_tokens, batch.completion_mask, clip_eps, beta
    )

    assert torch.allclose(loss, exp_loss, atol=1e-6)
    assert torch.allclose(policy_loss, exp_policy, atol=1e-6)
    assert torch.allclose(kl, exp_kl, atol=1e-6)


def test_gradient_flows_to_policy_logits():
    op = NativeGRPOLossOp()
    batch = _batch(seed=4)
    current = _current_logps(batch, seed=104).clone().requires_grad_(True)

    loss, _, _ = op.forward(
        current,
        batch.old_logps,
        batch.ref_logps,
        batch.rewards,
        batch.completion_mask,
        clip_eps=0.2,
        beta=0.01,
        samples_per_prompt=_SPP,
    )
    loss.backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    # Masked-out tokens must receive zero gradient.
    assert torch.all(current.grad[~batch.completion_mask] == 0.0)


def test_requires_exactly_one_group_spec():
    op = NativeGRPOLossOp()
    rewards = _batch(seed=6).rewards
    with pytest.raises(ValueError):
        op.group_advantages(rewards)
    with pytest.raises(ValueError):
        op.group_advantages(rewards, samples_per_prompt=_SPP, group_boundaries=[0, 4, 8, 12])


# Triton fused op (validated against the native reference)
@requires_triton_cuda
def test_triton_forward_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    batch = _batch(seed=0, device="cuda")
    current = _current_logps(batch, seed=100, device="cuda")
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    args = (current, batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask)
    n_loss, n_policy, n_kl = native.forward(*args, **kwargs)
    t_loss, t_policy, t_kl = fused.forward(*args, **kwargs)

    assert torch.allclose(t_loss, n_loss, atol=1e-4, rtol=1e-4)
    assert torch.allclose(t_policy, n_policy, atol=1e-4, rtol=1e-4)
    assert torch.allclose(t_kl, n_kl, atol=1e-4, rtol=1e-4)


@requires_triton_cuda
def test_triton_backward_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    batch = _batch(seed=7, device="cuda")
    current = _current_logps(batch, seed=107, device="cuda")
    rest = (batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    cur_n = current.clone().requires_grad_(True)
    loss_n, _, _ = native.forward(cur_n, *rest, **kwargs)
    loss_n.backward()

    cur_t = current.clone().requires_grad_(True)
    loss_t, _, _ = fused.forward(cur_t, *rest, **kwargs)
    loss_t.backward()

    assert cur_t.grad is not None
    assert torch.allclose(cur_t.grad, cur_n.grad, atol=1e-4, rtol=1e-4)
    assert torch.all(cur_t.grad[~batch.completion_mask] == 0.0)


@requires_triton_cuda
def test_triton_backward_with_grad_scaling():
    """A non-unit upstream gradient must scale the policy-logp gradient linearly."""
    fused = TritonGRPOLossOp()
    batch = _batch(seed=3, device="cuda")
    current = _current_logps(batch, seed=103, device="cuda")
    rest = (batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    cur1 = current.clone().requires_grad_(True)
    loss1, _, _ = fused.forward(cur1, *rest, **kwargs)
    loss1.backward()

    cur2 = current.clone().requires_grad_(True)
    loss2, _, _ = fused.forward(cur2, *rest, **kwargs)
    (3.0 * loss2).backward()

    assert torch.allclose(cur2.grad, 3.0 * cur1.grad, atol=1e-4, rtol=1e-4)


@requires_triton_cuda
def test_triton_group_advantages_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    rewards = _batch(seed=5, device="cuda").rewards
    got = fused.group_advantages(rewards, samples_per_prompt=_SPP)
    expected = native.group_advantages(rewards, samples_per_prompt=_SPP)
    assert torch.allclose(got, expected, atol=1e-5)
    # Variable group sizes via boundaries.
    got_b = fused.group_advantages(rewards, group_boundaries=[0, 5, 12])
    exp_b = native.group_advantages(rewards, group_boundaries=[0, 5, 12])
    assert torch.allclose(got_b, exp_b, atol=1e-5)


@requires_triton_cuda
def test_triton_apply_with_per_sequence_advantages_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    batch = _batch(seed=11, device="cuda")
    current = _current_logps(batch, seed=111, device="cuda")

    sample_adv = native.group_advantages(batch.rewards, samples_per_prompt=_SPP)  # per-sequence
    adv_tokens = native.expand_advantages(sample_adv, batch.completion_mask)  # per-token for native

    n_loss, _, _ = native.apply(
        current,
        batch.old_logps,
        batch.ref_logps,
        adv_tokens,
        batch.completion_mask,
        clip_eps=0.2,
        beta=0.05,
    )
    t_loss, _, _ = fused.apply(
        current,
        batch.old_logps,
        batch.ref_logps,
        sample_adv,
        batch.completion_mask,
        clip_eps=0.2,
        beta=0.05,
    )
    assert torch.allclose(t_loss, n_loss, atol=1e-4, rtol=1e-4)


# Pipeline composition: logp op -> grpo_loss op
def test_native_logp_composes_with_native_grpo():
    logp_op = NativeLogpOp()
    grpo = NativeGRPOLossOp()
    batch = _batch(seed=21)
    logits = _logits_like(batch, seed=121)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)
    rest = (batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask)

    current = logp_op.apply_fp32(logits, batch.token_ids)
    loss, _, _ = grpo.forward(current, *rest, **kwargs)

    oracle = selected_logprobs_reference(logits, batch.token_ids)
    exp, _, _ = grpo.forward(oracle, *rest, **kwargs)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, exp, atol=1e-6)


def test_native_logp_grpo_pipeline_is_differentiable_to_logits():
    """NativeLogpOp uses log_softmax/gather, so grads flow logits -> loss."""
    logp_op = NativeLogpOp()
    grpo = NativeGRPOLossOp()
    batch = _batch(seed=22)
    logits = _logits_like(batch, seed=122).clone().requires_grad_(True)

    current = logp_op.apply_fp32(logits, batch.token_ids)
    loss, _, _ = grpo.forward(
        current,
        batch.old_logps,
        batch.ref_logps,
        batch.rewards,
        batch.completion_mask,
        clip_eps=0.2,
        beta=0.05,
        samples_per_prompt=_SPP,
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@requires_triton_cuda
def test_dispatched_logp_composes_with_triton_grpo():
    """Real pipeline: dispatched CUDA fused logp -> Triton GRPO loss."""
    from rl_engine.kernels.registry import kernel_registry

    logp_op = kernel_registry.get_op("logp")
    grpo = TritonGRPOLossOp()
    batch = _batch(seed=23, device="cuda")
    logits = _logits_like(batch, seed=123, device="cuda")
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)
    rest = (batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask)

    current = logp_op.apply_fp32(logits, batch.token_ids)
    loss, _, _ = grpo.forward(current, *rest, **kwargs)

    oracle = selected_logprobs_reference(logits, batch.token_ids)
    exp, _, _ = NativeGRPOLossOp().forward(oracle, *rest, **kwargs)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, exp, atol=1e-3, rtol=1e-3)


# Loss step: masked-token invariance and a gradient step
def _perturb_inactive(batch, current):
    """Set garbage at masked positions; the loss must ignore them."""
    inactive = ~batch.completion_mask
    cur = current.clone()
    old = batch.old_logps.clone()
    ref = batch.ref_logps.clone()
    cur[inactive] = 1000.0
    old[inactive] = -1000.0
    ref[inactive] = 500.0
    return cur, old, ref


def test_masked_tokens_do_not_affect_native_loss():
    op = NativeGRPOLossOp()
    batch = _batch(seed=8, valid_density=0.75)
    current = _current_logps(batch, seed=108)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    base, _, _ = op.forward(
        current, batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask, **kwargs
    )
    cur_p, old_p, ref_p = _perturb_inactive(batch, current)
    pert, _, _ = op.forward(cur_p, old_p, ref_p, batch.rewards, batch.completion_mask, **kwargs)
    assert torch.allclose(base, pert)


@requires_triton_cuda
def test_masked_tokens_do_not_affect_triton_loss():
    fused = TritonGRPOLossOp()
    batch = _batch(seed=8, device="cuda", valid_density=0.75)
    current = _current_logps(batch, seed=108, device="cuda")
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    base, _, _ = fused.forward(
        current, batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask, **kwargs
    )
    cur_p, old_p, ref_p = _perturb_inactive(batch, current)
    pert, _, _ = fused.forward(cur_p, old_p, ref_p, batch.rewards, batch.completion_mask, **kwargs)
    assert torch.allclose(base, pert, atol=1e-5)


def _descend(op, batch, seed, *, device="cpu", steps=5, lr=0.05):
    rest = (batch.old_logps, batch.ref_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)
    initial = op.forward(_current_logps(batch, seed, device=device), *rest, **kwargs)[0]
    params = _current_logps(batch, seed, device=device).clone().requires_grad_(True)
    for _ in range(steps):
        loss, _, _ = op.forward(params, *rest, **kwargs)
        (grad,) = torch.autograd.grad(loss, params)
        params = (params - lr * grad).detach().requires_grad_(True)
    final = op.forward(params, *rest, **kwargs)[0]
    return initial, final


def test_grpo_gradient_step_reduces_loss():
    """Full loss step: forward -> backward -> SGD on the policy logps lowers the loss."""
    op = NativeGRPOLossOp()
    initial, final = _descend(op, _batch(seed=9), seed=109)
    assert final.item() < initial.item()


@requires_triton_cuda
def test_triton_grpo_gradient_step_reduces_loss():
    fused = TritonGRPOLossOp()
    initial, final = _descend(fused, _batch(seed=9, device="cuda"), seed=109, device="cuda")
    assert final.item() < initial.item()


# Registry dispatch (device-dependent backend selection)
def test_registry_dispatches_grpo_loss():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("grpo_loss")
    assert hasattr(op, "forward") and hasattr(op, "group_advantages")
    if _HAS_TRITON and torch.cuda.is_available():
        assert isinstance(op, TritonGRPOLossOp)
    else:
        assert isinstance(op, NativeGRPOLossOp)
