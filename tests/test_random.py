import unittest

import numpy as np
from scipy import stats

from hawkes_tools.random import (
    test_discrete,
    test_exponential,
    test_gaussian,
    test_poisson,
    test_uniform,
    test_uniform_int,
    test_uniform_threaded,
)


class RandomHelperTest(unittest.TestCase):
    def test_seeded_samples_are_reproducible_and_seed_sensitive(self):
        generators = [
            (test_uniform, (8,)),
            (test_uniform, (-2.0, 5.0, 8)),
            (test_gaussian, (8,)),
            (test_gaussian, (-10.0, 0.5, 8)),
            (test_exponential, (1.576, 8)),
            (test_poisson, (5.0, 8)),
            (test_uniform_int, (-2, 100, 8)),
            (test_discrete, (np.array([0.1, 0.2, 0.7]), 8)),
        ]
        for func, args in generators:
            with self.subTest(func=func.__name__, args=args):
                first = func(*args, 12099)
                second = func(*args, 12099)
                other = func(*args, 12100)
                np.testing.assert_array_equal(first, second)
                self.assertGreater(np.max(np.abs(first - other)), 0.0)

    def test_distribution_sanity(self):
        size = 5000
        uniform = test_uniform(size, 123)
        uniform_stat, uniform_p = stats.kstest(uniform, "uniform")
        self.assertLess(uniform_stat, 0.05)
        self.assertGreater(uniform_p, 0.01)

        gaussian = test_gaussian(-1.0, 2.0, size, 123)
        gaussian_stat, gaussian_p = stats.kstest(gaussian, "norm", (-1.0, 2.0))
        self.assertLess(gaussian_stat, 0.05)
        self.assertGreater(gaussian_p, 0.01)

        exponential = test_exponential(1.5, size, 123)
        exponential_stat, exponential_p = stats.kstest(exponential, "expon", (0.0, 1.0 / 1.5))
        self.assertLess(exponential_stat, 0.05)
        self.assertGreater(exponential_p, 0.01)

        probs = np.array([0.1, 0.2, 0.3, 0.4])
        discrete = test_discrete(probs, size, 123)
        observed = np.array([np.sum(discrete == i) for i in range(probs.size)], dtype=float)
        _, discrete_p = stats.chisquare(observed, size * probs)
        self.assertGreater(discrete_p, 0.01)

    def test_integer_and_poisson_shapes_and_ranges(self):
        integers = test_uniform_int(-2, 5, 100, 123)
        self.assertEqual(integers.shape, (100,))
        self.assertGreaterEqual(integers.min(), -2)
        self.assertLess(integers.max(), 5)

        poisson = test_poisson(3.0, 100, 123)
        self.assertEqual(poisson.shape, (100,))
        self.assertTrue(np.all(poisson >= 0.0))
        self.assertTrue(np.all(np.equal(poisson, np.floor(poisson))))

        threaded_1 = test_uniform_threaded(20)
        threaded_2 = test_uniform_threaded(20)
        self.assertEqual(threaded_1.shape, (20,))
        self.assertGreater(np.max(np.abs(threaded_1 - threaded_2)), 0.0)

    def test_validation_errors_are_explicit(self):
        invalid_calls = [
            (test_uniform, (1.0, 1.0, 10), "high must be greater than low"),
            (test_gaussian, (0.0, 0.0, 10), "std must be positive"),
            (test_exponential, (0.0, 10), "intensity must be positive"),
            (test_poisson, (-1.0, 10), "rate must be non-negative"),
            (test_uniform_int, (5, 5, 10), "high must be greater than low"),
            (test_discrete, (np.array([0.0, 0.0]), 10), "positive sum"),
            (test_discrete, (np.array([0.1, -0.1]), 10), "non-negative"),
            (test_uniform, (-1,), "size must be non-negative"),
        ]
        for func, args, message in invalid_calls:
            with self.subTest(func=func.__name__, args=args):
                with self.assertRaisesRegex(ValueError, message):
                    func(*args)


if __name__ == "__main__":
    unittest.main()
