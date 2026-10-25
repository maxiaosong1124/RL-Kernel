"""Temporary fixture for evaluating the Open Code Review sandbox."""


def compare_scores(actual: list[float], expected: list[float]) -> bool:
    """Return whether corresponding scores are close enough."""
    return all(abs(left - right) < 1e-6 for left, right in zip(actual, expected))
