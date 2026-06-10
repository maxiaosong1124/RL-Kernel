# GRPO Loss

GRPO Loss computes the Group Relative Policy Optimization objective for RL post-training:
it normalizes raw sequence rewards within each generation group into advantages, then
evaluates the clipped surrogate objective plus a reference-KL penalty over the active
completion tokens. It targets the GRPO training step, where a naive PyTorch implementation
allocates several broadcasted `[batch, completion_len]` intermediates and a per-token
advantage tensor.

The operator consumes per-token **log-probs**, so it composes directly with the [Fused LogP](fused-logp.md) operator:

```
logits --[logp op]--> logps --[grpo_loss op]--> loss
```

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

grpo_loss = kernel_registry.get_op("grpo_loss")

loss, policy_loss, kl = grpo_loss(
    current_logps,        # [B, T] current policy logps (differentiable)
    old_logps,            # [B, T] inference engine log-probs
    ref_logps,            # [B, T] reference model log-probs
    rewards,              # [B]
    completion_mask,      # [B, T]
    clip_eps=0.2,
    beta=0.04,
    samples_per_prompt=8,  # uniform groups; or pass group_boundaries=[...]
)

loss.backward()           # gradient flows into current_logps
```

Note: `B = num_prompts * samples_per_prompt`. The [B, T] tensors are made contiguous and flattened to 1-D [N = B*T] before the kernel launch.

### Group specification

Provide **exactly one** of:
- `samples_per_prompt: int` — uniform groups (every prompt has the same number of samples).
- `group_boundaries` — CSR-style offsets of length `num_groups + 1` (e.g. `[0, 8, 16, 24]`)
  for variable-sized groups.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `TritonGRPOLossOp` | Triton JIT kernels | Fused forward + analytic backward. |
| PyTorch fallback | `NativeGRPOLossOp` | None | Reference path; CPU and Triton-less GPUs. |

The Triton op fuses three kernels: `_group_norm_kernel` (per-group reward mean/std in
registers), and token-parallel `_grpo_fwd_kernel` / `_grpo_bwd_kernel`. Each token gathers
its advantage on the fly (`seq_id = token_index // completion_len`), so the broadcasted
`[B, T]` advantage tensor is never materialized.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `current_logps` | `[B, T]` | float (fp32 recommended) | Differentiable input |
| `old_logps` | `[B, T]` | float | Constant (no grad). |
| `ref_logps` | `[B, T]` | float | Constant (no grad). |
| `rewards` | `[B]` | float | One scalar per sequence. |
| `completion_mask` | `[B, T]` | bool / {0,1} | 2-D; `True` marks active tokens. |
| `loss` (output) | scalar | float32 | `policy_loss + beta * kl`. |
| `policy_loss`, `kl` (output) | scalar | float32 | Detached reporting values. |

## Accuracy

Reference semantics (matching `examples/grpo_single_gpu.py`):

```python
# advantages: group-normalized rewards (population std, unbiased=False)
grouped = rewards.view(-1, samples_per_prompt)
adv = (grouped - grouped.mean(1, keepdim=True)) / grouped.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
adv = adv.reshape(-1)[:, None].expand_as(completion_mask)

ratio = torch.exp(current_logps - old_logps)
policy = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
diff = ref_logps - current_logps
kl = torch.exp(diff) - diff - 1.0                      # k3 estimator
loss = masked_mean(policy, completion_mask) + beta * masked_mean(kl, completion_mask)
```

The Triton op matches the native reference (forward and backward) to `atol=1e-4` in fp32.
Composing with the dispatched CUDA fused logp matches a torch oracle to `atol=1e-3`.

## Performance Notes

```bash
python benchmarks/benchmark_grpo_loss.py
python benchmarks/benchmark_grpo_loss.py --configs "64,8,512;256,16,1024"
```

Indicative results (RTX PRO 6000, SM120, fp32):

| shape (prompts × samples × len) | tokens | forward speedup | fwd+bwd speedup | peak VRAM (native → Triton) |
| --- | --- | --- | --- | --- |
| 64 × 8 × 512 | 0.26M | 4.7× | 3.0× | 10 MB → 1 MB |
| 128 × 8 × 1024 | 1.05M | 3.2× | 2.7× | 40 MB → 4 MB |
| 256 × 16 × 1024 | 4.19M | 3.0× | 3.0× | 160 MB → 16 MB |

The ~10× VRAM reduction comes from not materializing the broadcasted advantage and
per-token surrogate/KL intermediates.

## Tests

```bash
python -m pytest tests/test_grpo_loss.py -v
```

Covers the native reference, Triton forward/backward vs native, the
`logp → grpo_loss` pipeline, masked-token invariance, an SGD loss step, and registry
dispatch. Triton tests skip without CUDA + Triton.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py`
- `rl_engine/kernels/ops/triton/triton_grpo_loss.py`
- `rl_engine/kernels/registry.py`
- `tests/test_grpo_loss.py`
- `benchmarks/benchmark_grpo_loss.py`
