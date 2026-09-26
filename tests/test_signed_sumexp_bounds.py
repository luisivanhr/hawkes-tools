import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.base import TimeFunction
from hawkes_tools.hawkes import HawkesKernelSumExp, SimuHawkes, SimuHawkesSumExpKernels
from hawkes_tools.hawkes import numeric


class SignedSumExpBoundsTest(unittest.TestCase):
    @staticmethod
    def _simulation(specialized, adjacency, decays, baseline, **kwargs):
        if specialized:
            return SimuHawkesSumExpKernels(
                adjacency=adjacency, decays=decays, baseline=baseline,
                verbose=False, **kwargs,
            )
        kernels = [
            [HawkesKernelSumExp(weights, decays) for weights in row]
            for row in adjacency
        ]
        return SimuHawkes(kernels=kernels, baseline=baseline, verbose=False, **kwargs)

    def test_cancelling_components_bound_the_later_rise(self):
        # h(s) = .5 (exp(-s) - exp(-10 s)) is nonnegative and has norm .45.
        # At the event h(0) = 0, but h(.25) is approximately .34836.
        for specialized in (False, True):
            for compiled in (False, True):
                for baseline in ([0.1], [TimeFunction(0.1)]):
                    with self.subTest(
                        specialized=specialized, compiled=compiled,
                        time_function=isinstance(baseline[0], TimeFunction),
                    ), patch.object(numeric, "NUMBA_AVAILABLE", compiled):
                        simulation = self._simulation(
                            specialized, [[[0.5, -0.05]]], [1.0, 10.0], baseline,
                        )
                        simulation.set_timestamps([np.array([1.0])], end_time=1.0)
                        bound = simulation._total_intensity_bound(1.0)
                        self.assertAlmostEqual(bound, 0.6)
                        self.assertAlmostEqual(
                            simulation._intensity_at(1.25)[0], 0.448357892223753,
                        )
                        future = np.array([
                            simulation._intensity_at(t)[0]
                            for t in np.linspace(1.0, 5.0, 101)
                        ])
                        self.assertTrue(np.all(future <= bound))

    def test_multivariate_envelope_and_history_boundary(self):
        decays = np.array([0.0, 1.0, 10.0])
        adjacency = np.array([
            [[9.0, 0.5, -0.05], [0.0, -0.2, 0.3], [0.0, 0.1, -0.4]],
            [[-3.0, -0.4, 0.2], [0.0, 0.2, -0.1], [0.0, -0.3, -0.2]],
            [[0.0, 0.3, -0.5], [0.0, -0.2, 0.1], [0.0, 0.4, 0.2]],
        ])
        timestamps = [np.array([0.1, 0.75, 1.5]), np.array([]), np.array([0.3, 0.75])]
        baseline = np.array([0.1, -0.2, 0.3])
        events, sizes = numeric.pack_realization(timestamps)
        time = 0.75
        for include_current in (False, True):
            history = [ts[ts <= time] if include_current else ts[ts < time] for ts in timestamps]
            expected_bound = sum(max(value, 0.0) for value in baseline)
            for i in range(3):
                for j in range(3):
                    for amplitude, decay in zip(adjacency[i, j], decays):
                        expected_bound += sum(
                            max(amplitude, 0.0) * decay * np.exp(-decay * (time - tj))
                            for tj in history[j]
                        )
            for bound_function in (
                numeric.sumexp_intensity_bound_reference, numeric.sumexp_intensity_bound,
            ):
                with self.subTest(include_current=include_current, function=bound_function.__name__):
                    bound = bound_function(
                        time, events, sizes, baseline, adjacency, decays, include_current,
                    )
                    self.assertAlmostEqual(bound, expected_bound)
                    # Freeze the available history: future recorded events must not
                    # contribute to either the envelope or this no-new-event path.
                    for future_time in time + np.array([0.0, 0.01, 0.1, 0.25, 1.0, 10.0]):
                        values = baseline.copy()
                        for i in range(3):
                            for j in range(3):
                                for amplitude, decay in zip(adjacency[i, j], decays):
                                    values[i] += sum(
                                        amplitude * decay * np.exp(-decay * (future_time - tj))
                                        for tj in history[j]
                                    )
                        self.assertLessEqual(np.maximum(values, 0.0).sum(), bound + 1e-14)

            # Generic kernels and the specialized simulation must obey the same
            # per-component bound, including empty nodes and event-time inclusion.
            for specialized in (False, True):
                with self.subTest(include_current=include_current, specialized=specialized):
                    simulation = self._simulation(specialized, adjacency, decays, baseline)
                    simulation.threshold_negative_intensity()
                    simulation.set_timestamps(timestamps, end_time=1.5)
                    self.assertAlmostEqual(
                        simulation._total_intensity_bound(time, include_current), expected_bound,
                    )

    def test_nonnegative_negative_and_empty_history_limits(self):
        decays = np.array([1.0, 10.0])
        for amplitudes in ([0.5, 0.05], [-0.5, -0.05], [0.0, 0.0]):
            for timestamps in (np.array([]), np.array([0.2, 0.8])):
                with self.subTest(amplitudes=amplitudes, timestamps=timestamps):
                    adjacency = np.array([[amplitudes]])
                    baseline = np.array([0.2])
                    events, sizes = numeric.pack_realization([timestamps])
                    expected = 0.2 + sum(
                        max(amplitude, 0.0) * decay * np.exp(-decay * (0.8 - tj))
                        for amplitude, decay in zip(amplitudes, decays)
                        for tj in timestamps
                    )
                    for function in (
                        numeric.sumexp_intensity_bound_reference, numeric.sumexp_intensity_bound,
                    ):
                        self.assertAlmostEqual(
                            function(0.8, events, sizes, baseline, adjacency, decays), expected,
                        )

    def test_simulation_continues_after_zero_current_intensity(self):
        # A seed event has zero immediate effect but can still have descendants.
        # Mock only the random draws, retaining the actual proposal scaling,
        # intensity evaluation, acceptance, and event insertion code.
        for specialized in (False, True):
            for compiled in (False, True):
                with self.subTest(specialized=specialized, compiled=compiled), patch.object(
                    numeric, "NUMBA_AVAILABLE", compiled,
                ):
                    simulation = self._simulation(
                        specialized, [[[0.5, -0.05]]], [1.0, 10.0], [0.0], max_jumps=2,
                    )
                    simulation.set_timestamps([np.array([1.0])], end_time=1.0)
                    simulation.end_time = 2.0
                    with patch.object(simulation, "_rng") as rng:
                        rng.exponential.side_effect = lambda scale: 0.125 * scale
                        rng.uniform.return_value = 0.0
                        with self.assertWarnsRegex(UserWarning, "Baselines have not been set"):
                            simulation.simulate()
                    np.testing.assert_allclose(simulation.timestamps[0], [1.0, 1.25])
                    rng.exponential.assert_called_once_with(2.0)

    def test_first_future_jump_probability_matches_integrated_intensity(self):
        # Until the first new event, the intensity is deterministic. Its
        # integrated hazard gives an independent distributional check of thinning.
        horizon = 2.0
        integrated = 0.1 * horizon + 0.5 * (1.0 - np.exp(-horizon)) - 0.05 * (
            1.0 - np.exp(-10.0 * horizon)
        )
        probability = -np.expm1(-integrated)
        n_simulations = 512
        tolerance = 6.0 * np.sqrt(probability * (1.0 - probability) / n_simulations)
        for specialized, compiled in ((False, True), (True, False), (True, True)):
            with self.subTest(specialized=specialized, compiled=compiled), patch.object(
                numeric, "NUMBA_AVAILABLE", compiled,
            ):
                jumped = 0
                for seed in range(n_simulations):
                    simulation = self._simulation(
                        specialized, [[[0.5, -0.05]]], [1.0, 10.0], [0.1],
                        max_jumps=2, seed=seed,
                    )
                    simulation.set_timestamps([np.array([1.0])], end_time=1.0)
                    simulation.end_time = 1.0 + horizon
                    simulation.simulate()
                    jumped += simulation.n_total_jumps == 2
                self.assertAlmostEqual(jumped / n_simulations, probability, delta=tolerance)


if __name__ == "__main__":
    unittest.main()
