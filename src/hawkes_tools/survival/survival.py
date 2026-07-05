"""Survival analysis helper functions."""

from __future__ import annotations

import numpy as np


def kaplan_meier(timestamps, event_observed):
    """Compute the Kaplan-Meier survival function estimate."""

    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)
    else:
        timestamps = np.asarray(timestamps)
    if isinstance(event_observed, list):
        event_observed = np.array(event_observed)
    else:
        event_observed = np.asarray(event_observed)

    timestamps_observed = timestamps[event_observed == 1]
    unique_timestamps_observed = np.concatenate((np.zeros(1), np.unique(timestamps_observed)))
    return np.cumprod(
        np.fromiter(
            (
                1.0 - np.sum(t == timestamps_observed) / np.sum(t <= timestamps)
                for t in unique_timestamps_observed
            ),
            dtype="float",
            count=unique_timestamps_observed.size,
        )
    )


def nelson_aalen(timestamps, event_observed):
    """Compute the Nelson-Aalen cumulative hazard estimate."""

    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)
    else:
        timestamps = np.asarray(timestamps)
    if isinstance(event_observed, list):
        event_observed = np.array(event_observed)
    else:
        event_observed = np.asarray(event_observed)

    timestamps_observed = timestamps[event_observed == 1]
    unique_timestamps_observed = np.concatenate((np.zeros(1), np.unique(timestamps_observed)))
    return np.cumsum(
        np.fromiter(
            (
                np.sum(t == timestamps_observed) / np.sum(t <= timestamps)
                for t in unique_timestamps_observed
            ),
            dtype="float",
            count=unique_timestamps_observed.size,
        )
    )
