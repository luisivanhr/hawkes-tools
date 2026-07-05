import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.prox import (
    ProxBinarsity,
    ProxElasticNet,
    ProxEquality,
    ProxGroupL1,
    ProxL1,
    ProxL1w,
    ProxL2,
    ProxL2Sq,
    ProxMulti,
    ProxNuclear,
    ProxPositive,
    ProxSlope,
    ProxZero,
)


class ProxPublicBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.coeffs = np.array(
            [
                -0.86017247,
                -0.58127151,
                -0.6116414,
                0.23186939,
                -0.85916332,
                1.6783094,
                1.39635801,
                1.74346116,
                -0.27576309,
                -1.00620197,
            ],
            dtype=float,
        )

    def test_public_exports_cover_reference_prox_names(self):
        import hawkes_tools.prox as prox

        expected = {
            "ProxZero",
            "ProxPositive",
            "ProxL1",
            "ProxL1w",
            "ProxL2Sq",
            "ProxL2",
            "ProxTV",
            "ProxNuclear",
            "ProxSlope",
            "ProxElasticNet",
            "ProxMulti",
            "ProxEquality",
            "ProxBinarsity",
            "ProxGroupL1",
        }
        self.assertEqual(set(), expected - set(prox.__all__))

    def test_l1_l1w_l2_and_l2sq_match_source_formulas(self):
        coeffs = self.coeffs
        strength = 3e-2
        step = 1.7

        l1 = ProxL1(strength, range=(3, 8))
        expected = coeffs.copy()
        sub = coeffs[3:8]
        expected[3:8] = np.sign(sub) * np.maximum(np.abs(sub) - strength * step, 0.0)
        np.testing.assert_allclose(l1.call(coeffs, step=step), expected)
        self.assertAlmostEqual(l1.value(coeffs), strength * np.abs(coeffs[3:8]).sum())

        weights = np.arange(5, dtype=float)
        l1w = ProxL1w(strength, weights=weights, range=(3, 8))
        expected = coeffs.copy()
        expected[3:8] = np.sign(sub) * np.maximum(np.abs(sub) - strength * weights * step, 0.0)
        np.testing.assert_allclose(l1w.call(coeffs, step=step), expected)
        self.assertAlmostEqual(l1w.value(coeffs), strength * np.sum(weights * np.abs(coeffs[3:8])))

        l2sq = ProxL2Sq(strength, range=(3, 8))
        expected = coeffs.copy()
        expected[3:8] = coeffs[3:8] / (1.0 + strength * step)
        np.testing.assert_allclose(l2sq.call(coeffs, step=step), expected)
        self.assertAlmostEqual(l2sq.value(coeffs), 0.5 * strength * np.dot(coeffs[3:8], coeffs[3:8]))

        l2 = ProxL2(strength, range=(3, 8))
        threshold = step * strength * np.sqrt(5)
        norm = np.linalg.norm(coeffs[3:8])
        expected = coeffs.copy()
        expected[3:8] = coeffs[3:8] * max(1.0 - threshold / norm, 0.0)
        np.testing.assert_allclose(l2.call(coeffs, step=step), expected)
        self.assertAlmostEqual(l2.value(coeffs), strength * np.sqrt(5) * norm)

    def test_positive_elasticnet_equality_multi_and_nuclear_behaviors(self):
        coeffs = self.coeffs

        positive = ProxPositive((3, 8))
        expected = coeffs.copy()
        expected[3:8] = np.maximum(expected[3:8], 0.0)
        np.testing.assert_allclose(positive.call(coeffs), expected)

        strength = 0.03
        ratio = 0.3
        step = 1.7
        elastic = ProxElasticNet(strength, ratio=ratio)
        l1 = ProxL1(strength * ratio)
        l2sq = ProxL2Sq(strength * (1.0 - ratio))
        expected = l2sq.call(l1.call(coeffs, step=step), step=step)
        np.testing.assert_allclose(elastic.call(coeffs, step=step), expected)
        self.assertAlmostEqual(elastic.value(coeffs), l1.value(coeffs) + l2sq.value(coeffs))

        equality = ProxEquality(range=(0, 3))
        projected = equality.call(coeffs)
        np.testing.assert_allclose(projected[:3], np.mean(coeffs[:3]))
        self.assertEqual(equality.value(np.ones(4)), 0.0)
        self.assertEqual(equality.value(np.array([1.0, 2.0])), np.inf)

        multi = ProxMulti([ProxL1(strength, range=(0, 5)), ProxPositive(range=(5, 10))])
        sequential = ProxPositive(range=(5, 10)).call(ProxL1(strength, range=(0, 5)).call(coeffs))
        np.testing.assert_allclose(multi.call(coeffs), sequential)

        matrix = np.diag([3.0, 1.0]).reshape(-1)
        nuclear = ProxNuclear(strength=0.5, n_rows=2)
        np.testing.assert_allclose(nuclear.call(matrix, step=1.0).reshape(2, 2), np.diag([2.5, 0.5]))
        self.assertAlmostEqual(nuclear.value(matrix), 2.0)

    def test_group_l1_and_binarsity_validation_and_values(self):
        blocks_start = [0, 3, 8]
        blocks_length = [3, 5, 2]
        strength = 0.5
        step = 1.7
        coeffs = self.coeffs

        prox = ProxGroupL1(strength=strength, blocks_start=blocks_start, blocks_length=blocks_length)
        expected = coeffs.copy()
        expected_value = 0.0
        for start, length in zip(blocks_start, blocks_length):
            end = start + length
            norm = np.linalg.norm(coeffs[start:end])
            threshold = step * strength * np.sqrt(length)
            expected[start:end] *= max(1.0 - threshold / norm, 0.0)
            expected_value += strength * np.sqrt(length) * norm
        np.testing.assert_allclose(prox.call(coeffs, step=step), expected)
        self.assertAlmostEqual(prox.value(coeffs), expected_value)

        binarsity = ProxBinarsity(strength=strength, blocks_start=blocks_start, blocks_length=blocks_length)
        expected_value = 0.0
        for start, length in zip(blocks_start, blocks_length):
            expected_value += np.abs(coeffs[start + 1 : start + length] - coeffs[start : start + length - 1]).sum()
        self.assertAlmostEqual(binarsity.value(coeffs), strength * expected_value)

        for cls in (ProxGroupL1, ProxBinarsity):
            with self.assertRaisesRegex(ValueError, "same size"):
                cls(strength, blocks_start=[0, 3], blocks_length=[2])
            with self.assertRaisesRegex(ValueError, "not overlap"):
                cls(strength, blocks_start=[0, 3, 8], blocks_length=[4, 5, 1])
            with self.assertRaisesRegex(ValueError, "sorted"):
                cls(strength, blocks_start=[0, 8, 3], blocks_length=[1, 1, 1])
            with self.assertRaisesRegex(ValueError, "positive size"):
                cls(strength, blocks_start=[0, 3, 8], blocks_length=[2, 0, 2])
            with self.assertRaisesRegex(ValueError, "starting"):
                cls(strength, blocks_start=[0, -3, 8], blocks_length=[1, 1, 1])
            with self.assertRaisesRegex(ValueError, "selected range"):
                cls(strength, blocks_start=[0, 4, 8], blocks_length=[3, 3, 10], range=(0, 17))

    def test_parameter_validation_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "range"):
            ProxL1(range=(-1, 3))
        with self.assertRaisesRegex(ValueError, "range"):
            ProxL1(range=(3, 3))
        for cls in (ProxL1, ProxL2Sq, ProxL2, ProxElasticNet, ProxNuclear, ProxSlope):
            with self.assertRaisesRegex(ValueError, "strength"):
                cls(strength=-0.1)
            with self.assertRaisesRegex(ValueError, "strength"):
                cls(strength=np.inf)
        with self.assertRaisesRegex(ValueError, "ratio"):
            ProxElasticNet(0.1, ratio=1.5)
        with self.assertRaisesRegex(ValueError, "weights"):
            ProxL1w(0.1, weights=[1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "step"):
            ProxL1(0.1).call(self.coeffs, step=-1.0)
        with self.assertRaisesRegex(ValueError, "out"):
            ProxL1(0.1).call(self.coeffs, out=np.zeros(3))
        with self.assertRaisesRegex(ValueError, "n_rows"):
            ProxNuclear(0.1, n_rows=0)


if __name__ == "__main__":
    unittest.main()
