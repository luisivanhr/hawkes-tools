"""Standalone random-sampling helpers."""

from __future__ import annotations

import time

import numpy as np

__all__ = [
    "test_uniform",
    "test_gaussian",
    "test_poisson",
    "test_exponential",
    "test_uniform_int",
    "test_discrete",
    "test_uniform_threaded",
]


def test_uniform(*args):
    """Draw uniform samples.

    Supported call forms are ``test_uniform(size[, seed])`` and
    ``test_uniform(low, high, size[, seed])``.
    """

    if len(args) in (1, 2):
        low, high, size, seed = 0.0, 1.0, args[0], args[1] if len(args) == 2 else None
    elif len(args) in (3, 4):
        low, high, size, seed = args[0], args[1], args[2], args[3] if len(args) == 4 else None
    else:
        raise TypeError("test_uniform expects size[, seed] or low, high, size[, seed]")
    low = float(low)
    high = float(high)
    if high <= low:
        raise ValueError("high must be greater than low")
    return _rng(seed).uniform(low, high, size=_validate_size(size))


def test_gaussian(*args):
    """Draw Gaussian samples.

    Supported call forms are ``test_gaussian(size[, seed])`` and
    ``test_gaussian(mean, std, size[, seed])``.
    """

    if len(args) in (1, 2):
        mean, std, size, seed = 0.0, 1.0, args[0], args[1] if len(args) == 2 else None
    elif len(args) in (3, 4):
        mean, std, size, seed = args[0], args[1], args[2], args[3] if len(args) == 4 else None
    else:
        raise TypeError("test_gaussian expects size[, seed] or mean, std, size[, seed]")
    std = float(std)
    if std <= 0:
        raise ValueError("std must be positive")
    return _rng(seed).normal(float(mean), std, size=_validate_size(size))


def test_exponential(intensity, size, seed=None):
    """Draw exponential samples with rate ``intensity``."""

    intensity = float(intensity)
    if intensity <= 0:
        raise ValueError("intensity must be positive")
    return _rng(seed).exponential(1.0 / intensity, size=_validate_size(size))


def test_poisson(rate, size, seed=None):
    """Draw Poisson samples with mean ``rate``."""

    rate = float(rate)
    if rate < 0:
        raise ValueError("rate must be non-negative")
    return _rng(seed).poisson(rate, size=_validate_size(size)).astype(float)


def test_uniform_int(low, high, size, seed=None):
    """Draw integer samples uniformly from ``[low, high)``."""

    low = int(low)
    high = int(high)
    if high <= low:
        raise ValueError("high must be greater than low")
    return _rng(seed).integers(low, high, size=_validate_size(size)).astype(float)


def test_discrete(probabilities, size, seed=None):
    """Draw categorical samples according to ``probabilities``."""

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty 1d array")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("probabilities must have positive sum")
    probs = probs / total
    return _rng(seed).choice(probs.size, size=_validate_size(size), p=probs).astype(float)


def test_uniform_threaded(size, wait_time=0):
    """Draw unseeded uniform samples after an optional delay.

    Every call creates an independent generator seeded from operating-system
    entropy.
    """

    wait_time = float(wait_time)
    if wait_time > 0:
        time.sleep(wait_time)
    return test_uniform(size)


def _rng(seed):
    return np.random.default_rng(None if seed is None else int(seed))


def _validate_size(size):
    size = int(size)
    if size < 0:
        raise ValueError("size must be non-negative")
    return size
