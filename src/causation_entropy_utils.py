from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


def _as_1d_array(x, *, name: str = "array") -> np.ndarray:
    """Return x as a flattened 1D NumPy float array."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_same_length(*arrays: np.ndarray) -> None:
    """Ensure all arrays have the same number of observations."""
    lengths = {arr.size for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("All input arrays must have the same number of observations.")


def _bin_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    """Construct bin edges spanning the data range."""
    if nbins < 1:
        raise ValueError("nbins must be a positive integer.")

    xmin = np.min(x)
    xmax = np.max(x)

    if xmin == xmax:
        eps = 0.5 if xmin == 0 else 0.5 * abs(xmin)
        return np.linspace(xmin - eps, xmax + eps, nbins + 1)

    return np.linspace(xmin, xmax, nbins + 1)


def empirical_prob(x, nbins: int = 10) -> np.ndarray:
    """Estimate a 1D empirical probability mass function by histogram binning."""
    x = _as_1d_array(x, name="x")
    counts, _ = np.histogram(x, bins=_bin_edges(x, nbins))
    return counts.astype(float) / x.size


def joint_prob(x, y, nbins: int = 10) -> np.ndarray:
    """Estimate a 2D empirical joint probability mass function."""
    x = _as_1d_array(x, name="x")
    y = _as_1d_array(y, name="y")
    _validate_same_length(x, y)

    counts, _, _ = np.histogram2d(
        x,
        y,
        bins=[_bin_edges(x, nbins), _bin_edges(y, nbins)],
    )
    return counts.astype(float) / x.size


def entropy(x, nbins: int = 10, base: float = 2) -> float:
    """Estimate the entropy of a scalar variable from a histogram."""
    probs = empirical_prob(x, nbins=nbins)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs) / np.log(base)))


def conditional_entropy(x, *y_vars, nbins: int = 10, base: float = 2) -> float:
    x = _as_1d_array(x, name="x")
    if len(y_vars) == 0:
        return entropy(x, nbins=nbins, base=base)

    y_arrays = [_as_1d_array(y, name=f"y_vars[{i}]") for i, y in enumerate(y_vars)]
    _validate_same_length(x, *y_arrays)

    sample = np.column_stack([x, *y_arrays])
    bins = [_bin_edges(sample[:, i], nbins) for i in range(sample.shape[1])]
    joint_counts, _ = np.histogramdd(sample, bins=bins)

    joint_probs = joint_counts / x.size
    cond_probs = joint_probs.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(
            joint_probs > 0,
            joint_probs * (np.log(joint_probs / cond_probs) / np.log(base)),
            0.0,
        )

    return float(-np.sum(term))


def scipy_conditional_entropy(x, y, nbins: int = 10, base: float = 2) -> float:
    """Reference implementation of H(X | Y) using H(X,Y) - H(Y)."""
    joint = joint_prob(x, y, nbins=nbins).ravel()
    joint = joint[joint > 0]

    py = empirical_prob(y, nbins=nbins)
    py = py[py > 0]

    return float(scipy_entropy(joint, base=base) - scipy_entropy(py, base=base))

def mutual_information(x, y, nbins=10, base=2):
    return entropy(x, nbins=nbins, base=base) - conditional_entropy(x, y, nbins=nbins, base=base)

def conditional_mutual_information(x, y, *z, nbins=10, base=2):
    return conditional_entropy(x, *z, nbins=nbins, base=base) - conditional_entropy(x, y, *z, nbins=nbins, base=base)

def causation_entropy(target, source, conditioning_set=(), nbins=10, base=2):
    return conditional_mutual_information(target, source, *conditioning_set, nbins=nbins, base=base)