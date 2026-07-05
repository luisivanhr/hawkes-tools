import unittest

import numpy as np

from hawkes_tools.base_model import (
    GRAD,
    HESSIAN_NORM,
    LOSS,
    LOSS_AND_GRAD,
    PASS_OVER_DATA,
    ModelGeneralizedLinear,
    ModelLipschitz,
    ModelSecondOrder,
    ModelSelfConcordant,
)


class QuadraticModel(ModelSecondOrder):
    def _set_data(self, center):
        self.center = np.asarray(center, dtype=float)
        self.dtype = self.center.dtype

    def _get_n_coeffs(self):
        return self.center.size

    def _loss(self, coeffs):
        diff = coeffs - self.center
        return 0.5 * float(diff @ diff)

    def _grad(self, coeffs, out):
        out[:] = coeffs - self.center

    def _hessian_norm(self, coeffs, point):
        del point
        return float(coeffs @ coeffs)


class GLMStub(ModelGeneralizedLinear):
    def _loss(self, coeffs):
        return float(np.sum(coeffs))


class LipModel(ModelLipschitz):
    def _set_data(self, values):
        self.values = np.asarray(values, dtype=float)
        self.dtype = self.values.dtype

    def _get_n_coeffs(self):
        return self.values.size

    def _loss(self, coeffs):
        return float(np.sum(coeffs))

    def _get_lip_best(self):
        self.lip_calls = getattr(self, "lip_calls", 0) + 1
        return float(np.linalg.norm(self.values) ** 2)


class SelfConcordantModel(ModelSelfConcordant):
    def _set_data(self, value):
        self.value = float(value)
        self.dtype = np.dtype("float64")

    def _get_n_coeffs(self):
        return 1

    def _loss(self, coeffs):
        return float(coeffs[0] ** 2)

    def _get_sc_constant(self):
        return self.value


class BaseModelTest(unittest.TestCase):
    def test_first_and_second_order_counters_match_tick_protocol(self):
        model = QuadraticModel().fit(np.array([1.0, 2.0]))
        coeffs = np.array([2.0, 4.0])

        self.assertEqual(model.n_coeffs, 2)
        self.assertEqual(model.loss(coeffs), 2.5)
        np.testing.assert_allclose(model.grad(coeffs), np.array([1.0, 2.0]))
        loss, grad = model.loss_and_grad(coeffs)
        self.assertEqual(loss, 2.5)
        np.testing.assert_allclose(grad, np.array([1.0, 2.0]))
        self.assertEqual(model.hessian_norm(coeffs, coeffs), 20.0)

        self.assertEqual(model.n_calls_loss, 2)
        self.assertEqual(model.n_calls_grad, 2)
        self.assertEqual(model.n_calls_loss_and_grad, 1)
        self.assertEqual(model.n_calls_hessian_norm, 1)
        self.assertEqual(
            model.n_passes_over_data,
            model.pass_per_operation[LOSS]
            + model.pass_per_operation[GRAD]
            + model.pass_per_operation[LOSS_AND_GRAD]
            + model.pass_per_operation[HESSIAN_NORM],
        )
        self.assertEqual(PASS_OVER_DATA, "n_passes_over_data")

    def test_fit_guards_and_shape_validation(self):
        model = QuadraticModel()
        with self.assertRaisesRegex(ValueError, "fit"):
            _ = model.n_coeffs
        with self.assertRaisesRegex(ValueError, "fit"):
            model.loss(np.ones(2))

        model.fit(np.ones(2))
        with self.assertRaisesRegex(ValueError, "expects 2"):
            model.grad(np.ones(3))

    def test_model_labels_features_and_generalized_linear_coeff_count(self):
        features = np.arange(12.0).reshape(4, 3)
        labels = np.arange(4.0)

        model = GLMStub(fit_intercept=True).fit(features, labels)
        self.assertEqual(model.n_samples, 4)
        self.assertEqual(model.n_features, 3)
        self.assertEqual(model.n_coeffs, 4)
        self.assertEqual(model._epoch_size, 4)
        self.assertEqual(model._rand_max, 4)

        no_intercept = GLMStub(fit_intercept=False).fit(features, labels)
        self.assertEqual(no_intercept.n_coeffs, 3)

        with self.assertRaisesRegex(ValueError, "Features has 4 samples"):
            GLMStub().fit(features, labels[:3])

    def test_lipschitz_best_is_cached_and_requires_fit(self):
        model = LipModel()
        with self.assertRaisesRegex(ValueError, "fit"):
            model.get_lip_best()

        model.fit(np.array([3.0, 4.0]))
        self.assertEqual(model.get_lip_best(), 25.0)
        self.assertEqual(model.get_lip_best(), 25.0)
        self.assertEqual(model.lip_calls, 1)

    def test_self_concordant_constant_requires_fit(self):
        model = SelfConcordantModel()
        with self.assertRaisesRegex(ValueError, "fit"):
            _ = model._sc_constant
        model.fit(3.5)
        self.assertEqual(model._sc_constant, 3.5)

    def test_deep_import_wrappers_resolve_public_classes(self):
        from hawkes_tools.base_model.model_first_order import ModelFirstOrder
        from hawkes_tools.base_model.model_generalized_linear import (
            ModelGeneralizedLinear as ImportedGLM,
        )

        self.assertTrue(issubclass(QuadraticModel, ModelFirstOrder))
        self.assertIs(ImportedGLM, ModelGeneralizedLinear)


if __name__ == "__main__":
    unittest.main()
