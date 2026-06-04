# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_grpo_single_gpu_example_cpu_smoke():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in [str(REPO_ROOT), env.get("PYTHONPATH", "")] if path
    )
    result = subprocess.run(
        [
            sys.executable,
            "examples/grpo_single_gpu.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--num-prompts",
            "1",
            "--samples-per-prompt",
            "2",
            "--prompt-len",
            "2",
            "--completion-len",
            "3",
            "--vocab-size",
            "16",
            "--hidden-dim",
            "8",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "completed grpo_single_gpu" in result.stdout
    assert "device=cpu" in result.stdout


def test_grpo_single_gpu_example_require_fused_rejects_cpu_fallback():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in [str(REPO_ROOT), env.get("PYTHONPATH", "")] if path
    )
    result = subprocess.run(
        [
            sys.executable,
            "examples/grpo_single_gpu.py",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--num-prompts",
            "1",
            "--samples-per-prompt",
            "2",
            "--completion-len",
            "2",
            "--vocab-size",
            "8",
            "--require-fused-logp",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--require-fused-logp was set" in result.stderr
