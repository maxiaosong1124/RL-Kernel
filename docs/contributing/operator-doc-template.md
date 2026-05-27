# Operator Doc Template

Copy this template when adding a new operator page.

## Summary

Describe what the operator does, which RL workload it targets, and why it exists.

## Entry Point

```python
# Show the public Python API used by callers.
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA |  |  |  |
| ROCm |  |  |  |
| PyTorch fallback |  |  |  |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
|  |  |  |  |

## Dispatch Behavior

Explain how the registry chooses this operator and what happens when a backend is
unavailable.

## Accuracy

Document the PyTorch reference implementation, tolerance, and any dtype-specific behavior.

## Performance Notes

List the benchmark command and the workload sizes used to evaluate the operator.

## Tests

```bash
python -m pytest path/to/test.py -v
```

## Known Limitations

Track unsupported shapes, dtypes, devices, or architecture constraints.
