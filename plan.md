# Plan: Minimal Reproducible Single-GPU GRPO Example

## Issue

Add a self-contained `examples/grpo_single_gpu.py` script that can run end-to-end on a single A100 with a toy dataset and demonstrate the RL-Kernel GRPO training path without requiring a multi-GPU setup.

## Goals

- Provide a runnable example under `examples/` because the repository currently has no examples.
- Avoid external datasets, model downloads, vLLM services, DeepSpeed, Ray, or multi-process launchers.
- Exercise the same core pieces used by the existing tests: synthetic RL-shaped batches, selected-token logprobs, GRPO/PPO-style ratio and KL helpers, masked reductions, and kernel backend dispatch.
- Keep a CPU smoke-test path so the example can be validated in CI or on development machines without CUDA.

## Implementation Plan

1. Create `examples/grpo_single_gpu.py`.
   - Build a deterministic synthetic GRPO batch with `make_synthetic_rl_kernel_batch`.
   - Use a tiny embedding-plus-linear policy as the trainable model.
   - Compute group-normalized rewards and token-level advantages.
   - Initialize old policy logprobs and reference logprobs from the initial policy.
   - Resolve the RL-Kernel logp backend on CUDA through `kernel_registry`; use `NativeLogpOp` for explicit CPU runs.
   - Run a few GRPO optimization steps with clipped policy loss and reference KL.
   - Print concise per-step metrics and final completion status.

2. Add `tests/test_grpo_single_gpu_example.py`.
   - Run the example as a subprocess from the repository root.
   - Force `--device cpu` and tiny tensor sizes for a fast, deterministic smoke test.
   - Assert the process exits successfully and reports completion.

3. Validate locally.
   - Run the new CPU smoke test.
   - Run the example directly with tiny CPU settings.
   - Check edited files for linter diagnostics.

## Acceptance Criteria

- `python examples/grpo_single_gpu.py --device auto` runs on a single GPU when CUDA is available.
- `python examples/grpo_single_gpu.py --device cpu --steps 2 ...` runs without CUDA.
- The script completes at least one optimizer step and verifies finite loss values.
- The smoke test passes with `pytest tests/test_grpo_single_gpu_example.py`.

## Validation Performed

- Built the CUDA extension with `MAX_JOBS=2 python setup.py build_ext --inplace`.
- Ran the strict single-GPU path on A100:
  `python examples/grpo_single_gpu.py --device cuda --require-fused-logp --steps 2 --num-prompts 1 --samples-per-prompt 2 --prompt-len 2 --completion-len 3 --vocab-size 16 --hidden-dim 8`.
- Confirmed CUDA dispatch selected `FusedLogpGenericOp` and completed two GRPO optimization steps.
- Ran `python -m pytest tests/test_grpo_single_gpu_example.py` and confirmed both smoke tests passed.
