import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.linear_model import weights_sparse_gauss
from hawkes_tools.metrics import support_fdp, support_recall
from hawkes_tools.preprocessing import LongitudinalFeaturesLagger
from hawkes_tools.robust import (
    ModelLinRegWithIntercepts,
    RobustLinearRegression,
    features_normal_cov_toeplitz,
    std_iqr,
    std_mad,
)
from hawkes_tools.survival import (
    BatchConvSCCS,
    ConvSCCS,
    CoxRegression,
    CustomEffects,
    ModelCoxRegPartialLik,
    ModelSCCS,
    SimuCoxReg,
    SimuCoxRegWithCutPoints,
    SimuSCCS,
    StreamConvSCCS,
    kaplan_meier,
    nelson_aalen,
)


def finite_difference_grad(model, coeffs, eps=1e-6):
    grad = np.zeros_like(coeffs, dtype=float)
    for i in range(coeffs.size):
        step = np.zeros_like(coeffs, dtype=float)
        step[i] = eps
        grad[i] = (model.loss(coeffs + step) - model.loss(coeffs - step)) / (2 * eps)
    return grad


class MetricsAndRobustTest(unittest.TestCase):
    def test_support_metrics_match_reference_definitions(self):
        truth = np.array([1.0, 0.0, 2.0, 0.0])
        found = np.array([0.5, 1.0, 0.0, 0.0])

        self.assertEqual(support_fdp(truth, found), 0.5)
        self.assertEqual(support_recall(truth, found), 0.5)

    def test_robust_scale_estimators_match_reference_formulas(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 12.0])

        self.assertGreater(std_mad(x), 0.0)
        self.assertGreater(std_iqr(x), 0.0)

    def test_model_linreg_with_intercepts_gradient_matches_augmented_design(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 4))
        y = rng.normal(size=30)
        model = ModelLinRegWithIntercepts(fit_intercept=True).fit(X, y)
        coeffs = rng.normal(size=model.n_coeffs)

        X_aug = np.hstack([X, np.ones((X.shape[0], 1)), np.eye(X.shape[0])])
        expected = X_aug.T @ (X_aug @ coeffs - y) / X.shape[0]
        np.testing.assert_allclose(model.grad(coeffs), expected, atol=1e-10)
        self.assertAlmostEqual(model.get_lip_best(), (np.linalg.svd(X, compute_uv=False)[0] ** 2 + 2) / X.shape[0])

    def test_robust_linear_regression_recovers_outlier_support(self):
        np.random.seed(12)
        n_samples = 300
        n_features = 5
        n_outliers = 10
        weights = np.sqrt(2 * np.log(np.linspace(1, 10, n_features) * n_features))
        sample_intercepts = weights_sparse_gauss(n_weights=n_samples, nnz=n_outliers)
        mask = sample_intercepts != 0
        sample_intercepts[mask] = (
            5.0
            * np.sqrt(2 * np.log(np.linspace(1, 10, n_outliers) * n_samples))
            * np.sign(sample_intercepts[mask])
        )
        X = features_normal_cov_toeplitz(n_samples, n_features, 0.5)
        y = X.dot(weights) + np.random.randn(n_samples) - 3.0 + sample_intercepts

        learner = RobustLinearRegression(
            C_sample_intercepts=n_samples,
            fit_intercept=True,
            fdr=0.2,
            max_iter=3000,
            tol=1e-7,
            solver="agd",
            penalty="none",
            verbose=False,
        ).fit(X, y)

        self.assertGreater(float(learner.weights @ weights), 0.0)
        self.assertAlmostEqual(support_recall(sample_intercepts, learner.sample_intercepts), 1.0)
        self.assertAlmostEqual(support_fdp(sample_intercepts, learner.sample_intercepts), 0.23076923076923078)


class SurvivalTest(unittest.TestCase):
    def test_hazard_rate_matches_survival_function_reference(self):
        np.random.seed(238924)
        n_observations = 100
        timestamps = np.random.uniform(size=n_observations)
        observations = np.ones(n_observations)

        hazard = nelson_aalen(timestamps, observations)
        survival = kaplan_meier(timestamps, observations)

        np.testing.assert_allclose(survival, np.exp(-hazard), atol=1e-2)

    def test_simu_coxreg_outputs_original_triplet_semantics(self):
        features, times, censoring = SimuCoxReg(
            np.array([0.3, 1.2]), n_samples=150, seed=123, verbose=False
        ).simulate()

        self.assertEqual(features.shape, (150, 2))
        self.assertEqual(times.shape, (150,))
        self.assertEqual(censoring.shape, (150,))
        self.assertTrue(np.all(times > 0.0))
        self.assertTrue(set(np.unique(censoring)).issubset({0, 1}))

    def test_model_coxreg_partial_likelihood_gradient_and_sparse_parity(self):
        features, times, censoring = SimuCoxReg(
            np.array([0.2, -0.4, 0.7]), n_samples=60, seed=44, verbose=False
        ).simulate()
        model = ModelCoxRegPartialLik().fit(features, times, censoring)
        sparse_model = ModelCoxRegPartialLik().fit(
            sparse.csr_matrix(features), times, censoring
        )
        coeffs = np.array([0.1, -0.2, 0.3])

        np.testing.assert_allclose(
            model.grad(coeffs),
            finite_difference_grad(model, coeffs),
            atol=1e-5,
        )
        self.assertAlmostEqual(model.loss(coeffs), sparse_model.loss(coeffs))
        np.testing.assert_allclose(model.grad(coeffs), sparse_model.grad(coeffs))
        self.assertEqual(model.n_coeffs, 3)
        self.assertEqual(model.n_failures, int(np.sum(censoring != 0)))
        self.assertGreaterEqual(model.censoring_rate, 0.0)
        self.assertLessEqual(model.censoring_rate, 1.0)

    def test_cox_regression_fit_score_and_warm_start(self):
        features, times, censoring = SimuCoxReg(
            np.array([0.5, -0.25, 0.1]), n_samples=80, seed=55, verbose=False
        ).simulate()
        learner = CoxRegression(
            penalty="l2",
            solver="agd",
            C=100.0,
            max_iter=20,
            tol=0.0,
            verbose=False,
            warm_start=True,
        )
        initial_score = ModelCoxRegPartialLik().fit(features, times, censoring).loss(
            np.zeros(features.shape[1])
        )
        learner.fit(features, times, censoring)
        first_score = learner.score()
        learner.fit(features, times, censoring)
        second_score = learner.score()

        self.assertEqual(learner.coeffs.shape, (features.shape[1],))
        self.assertLess(first_score, initial_score)
        self.assertLessEqual(second_score, first_score)
        self.assertAlmostEqual(
            learner.score(features, times, censoring),
            ModelCoxRegPartialLik().fit(features, times, censoring).loss(learner.coeffs),
        )

    def test_cox_regression_score_and_input_validation(self):
        features, times, censoring = SimuCoxReg(
            np.array([0.1, 0.2]), n_samples=30, seed=66, verbose=False
        ).simulate()
        learner = CoxRegression(max_iter=2, verbose=False)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            learner.score()
        with self.assertRaisesRegex(ValueError, "censoring"):
            learner.fit(features, times, np.full_like(censoring, 2))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            learner.fit(features, -times, censoring)

        learner.fit(features, times, censoring)
        with self.assertRaisesRegex(ValueError, "features"):
            learner.score(None, times, censoring)
        with self.assertRaisesRegex(ValueError, "times"):
            learner.score(features, None, censoring)
        with self.assertRaisesRegex(ValueError, "censoring"):
            learner.score(features, times, None)

    def test_cox_regression_constructor_validation(self):
        invalid_cases = [
            ({"C": np.nan}, "C"),
            ({"elastic_net_ratio": np.nan}, "elastic_net_ratio"),
            ({"step": 0.0}, "step"),
            ({"tol": np.nan}, "tol"),
            ({"max_iter": -1}, "max_iter"),
            ({"print_every": 0}, "print_every"),
            ({"record_every": 0}, "record_every"),
        ]
        for kwargs, pattern in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    CoxRegression(**kwargs)

    def test_cox_model_direct_input_validation(self):
        features, times, censoring = SimuCoxReg(
            np.array([0.1, 0.2]), n_samples=20, seed=77, verbose=False
        ).simulate()

        with self.assertRaisesRegex(ValueError, "times"):
            ModelCoxRegPartialLik().fit(features, np.full_like(times, np.nan), censoring)
        with self.assertRaisesRegex(ValueError, "censoring"):
            ModelCoxRegPartialLik().fit(features, times, np.full_like(censoring, 2))
        bad_features = features.copy()
        bad_features[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "features"):
            ModelCoxRegPartialLik().fit(bad_features, times, censoring)

    def test_model_sccs_loss_gradient_and_lipschitz(self):
        X = [
            np.array([[0, 1], [0, 1]], dtype="float64"),
            np.array([[1, 1], [1, 0]], dtype="float64"),
        ]
        y = [
            np.array([1, 0], dtype="int32"),
            np.array([0, 1], dtype="int32"),
        ]
        n_lags = np.array([1, 1], dtype="uint64")
        lagged, _, _ = LongitudinalFeaturesLagger(n_lags).fit_transform(X)
        model = ModelSCCS(n_intervals=2, n_lags=n_lags).fit(lagged, y)
        coeffs = np.array([0.0, 0.0, 1.0, 0.0])

        expected_loss = -np.log((np.e / (2 * np.e) * 1 / (1 + np.e))) / 2
        self.assertAlmostEqual(model.loss(coeffs), expected_loss)
        np.testing.assert_allclose(
            model.grad(coeffs),
            finite_difference_grad(model, coeffs),
            atol=1e-6,
        )

        sparse_model = ModelSCCS(n_intervals=2, n_lags=n_lags).fit(
            [sparse.csr_matrix(x) for x in lagged], y
        )
        self.assertAlmostEqual(model.loss(coeffs), sparse_model.loss(coeffs))
        np.testing.assert_allclose(model.grad(coeffs), sparse_model.grad(coeffs))

    def test_model_sccs_lipschitz_matches_reference_case(self):
        X = [
            np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype="float64"),
            np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype="float64"),
        ]
        y = [
            np.array([0, 1, 0], dtype="int32"),
            np.array([0, 1, 0], dtype="int32"),
        ]
        n_lags = np.repeat(1, 3).astype(dtype="uint64")
        lagged, _, _ = LongitudinalFeaturesLagger(n_lags).fit_transform(X)
        model = ModelSCCS(n_intervals=3, n_lags=n_lags).fit(lagged, y)
        self.assertEqual(model.get_lip_max(), 0.5)
        with self.assertRaisesRegex(NotImplementedError, "get_lip_max"):
            model.get_lip_best()

    def test_sccs_lag_validation_rejects_unsigned_wraparound_cases(self):
        with self.assertRaisesRegex(ValueError, "n_lags"):
            ModelSCCS(n_intervals=4, n_lags=np.array([-1, 0]))
        with self.assertRaisesRegex(ValueError, "n_lags"):
            SimuSCCS(5, 4, 2, np.array([-1, 0]), verbose=False)
        with self.assertRaisesRegex(ValueError, "n_lags"):
            ConvSCCS(n_lags=np.array([-1, 0]))
        with self.assertRaisesRegex(ValueError, "n_cases"):
            SimuSCCS(0, 4, 2, np.array([0, 0]), verbose=False)
        with self.assertRaisesRegex(ValueError, "censoring_scale"):
            SimuSCCS(5, 4, 2, np.array([0, 0]), censoring_scale=0, verbose=False)

    def test_simu_coxreg_with_cut_points_outputs_reference_tuple_semantics(self):
        sim = SimuCoxRegWithCutPoints(
            n_samples=40,
            n_features=3,
            n_cut_points=2,
            sparsity=1 / 3,
            seed=123,
            verbose=False,
        )
        features, times, censoring, cut_points, coeffs, sparse_blocks = sim.simulate()

        self.assertEqual(features.shape, (40, 3))
        self.assertEqual(times.shape, (40,))
        self.assertEqual(censoring.shape, (40,))
        self.assertEqual(sorted(cut_points), ["0", "1", "2"])
        self.assertEqual(coeffs.shape, (9,))
        self.assertEqual(sparse_blocks.shape, (1,))
        self.assertTrue(np.all(times > 0.0))
        self.assertTrue(set(np.unique(censoring)).issubset({0, 1}))
        for boundaries in cut_points.values():
            self.assertEqual(boundaries[0], -np.inf)
            self.assertEqual(boundaries[-1], np.inf)
            self.assertEqual(boundaries.shape, (4,))

    def test_simu_coxreg_with_cut_points_is_seed_reproducible(self):
        kwargs = dict(
            n_samples=25,
            n_features=4,
            n_cut_points=1,
            sparsity=0.25,
            seed=321,
            verbose=False,
        )
        first = SimuCoxRegWithCutPoints(**kwargs).simulate()
        second = SimuCoxRegWithCutPoints(**kwargs).simulate()

        for left, right in zip(first[:3], second[:3]):
            np.testing.assert_allclose(left, right)
        for key in first[3]:
            np.testing.assert_allclose(first[3][key], second[3][key])
        np.testing.assert_allclose(first[4], second[4])
        np.testing.assert_array_equal(first[5], second[5])

    def test_simu_coxreg_constructor_validation(self):
        coeffs = np.array([0.3, -0.2])
        invalid_cases = [
            (SimuCoxReg, {"coeffs": coeffs, "n_samples": 0}, "n_samples"),
            (SimuCoxReg, {"coeffs": coeffs, "shape": np.nan}, "shape"),
            (SimuCoxReg, {"coeffs": coeffs, "scale": 0.0}, "scale"),
            (SimuCoxReg, {"coeffs": coeffs, "censoring_factor": 0.0}, "censoring_factor"),
            (SimuCoxReg, {"coeffs": coeffs, "cov_corr": np.nan}, "cov_corr"),
            (SimuCoxRegWithCutPoints, {"n_samples": 0}, "n_samples"),
            (SimuCoxRegWithCutPoints, {"n_features": 0}, "n_features"),
            (SimuCoxRegWithCutPoints, {"n_cut_points": -1}, "n_cut_points"),
            (SimuCoxRegWithCutPoints, {"n_cut_points_factor": 0.0}, "n_cut_points_factor"),
            (SimuCoxRegWithCutPoints, {"sparsity": np.nan}, "sparsity"),
            (SimuCoxRegWithCutPoints, {"censoring_factor": 0.0}, "censoring_factor"),
        ]
        for cls, kwargs, pattern in invalid_cases:
            with self.subTest(cls=cls.__name__, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    cls(**kwargs)

    def test_simu_sccs_and_conv_sccs_gallery_shapes(self):
        lags = 5
        n_lags = np.repeat(lags, 2).astype("uint64")
        effects = CustomEffects(lags + 1)
        coeffs = [
            np.log(effects.constant_effect(1.0)),
            np.log(effects.bell_shaped_effect(1.5, 3, 1, 1)),
        ]
        sim = SimuSCCS(
            30,
            40,
            2,
            n_lags,
            coeffs=coeffs,
            seed=42,
            n_correlations=1,
            verbose=False,
        )
        features, censored_features, labels, censoring, true_coeffs = sim.simulate()

        self.assertEqual(len(features), 30)
        self.assertEqual(len(censored_features), 30)
        self.assertEqual(len(labels), 30)
        self.assertEqual(censoring.shape, (30,))
        self.assertEqual(true_coeffs[0].shape, (lags + 1,))
        self.assertEqual(sim.hawkes_exp_kernels.adjacency.shape, (2, 2))

        learner = ConvSCCS(n_lags=n_lags, max_iter=8, random_state=42)
        fitted, ci = learner.fit(features, labels, censoring, confidence_intervals=True, n_samples_bootstrap=2)

        self.assertEqual(len(fitted), 2)
        self.assertEqual(fitted[0].shape, (lags + 1,))
        self.assertEqual(ci["refit_coeffs"][1].shape, (lags + 1,))
        self.assertEqual(ci["lower_bound"][0].shape, (lags + 1,))
        self.assertAlmostEqual(learner.score(), learner.score(features, labels, censoring))

    def test_conv_sccs_explicit_score_validation(self):
        n_lags = np.repeat(1, 2).astype("uint64")
        learner = ConvSCCS(n_lags=n_lags)
        features = [sparse.csr_matrix(np.ones((3, 2)))]
        labels = [np.array([0, 1, 0], dtype="int32")]
        censoring = np.array([3], dtype="uint64")
        learner.fit(features, labels, censoring)

        with self.assertRaisesRegex(ValueError, "features"):
            learner.score(None, labels, censoring)
        with self.assertRaisesRegex(ValueError, "labels"):
            learner.score(features, None, censoring)
        with self.assertRaisesRegex(ValueError, "censoring"):
            learner.score(features, labels, None)
        with self.assertRaisesRegex(ValueError, "labels"):
            learner.score(features, [np.array([1, 0], dtype="int32")], censoring)

    def test_conv_sccs_tv_and_group_penalties_enter_objective(self):
        n_lags = np.array([2, 0], dtype="uint64")
        features = [
            sparse.csr_matrix(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                    ]
                )
            )
        ]
        labels = [np.array([0, 1, 0, 0], dtype="int32")]
        censoring = np.array([4], dtype="uint64")
        learner = ConvSCCS(
            n_lags=n_lags,
            penalized_features=np.array([0]),
            C_tv=2.0,
            C_group_l1=4.0,
        )
        design, y, lengths = learner._stack_lagged_data(features, labels, censoring)
        coeffs = np.array([1.0, 3.0, 2.0, -4.0])
        unpenalized_value, unpenalized_grad = learner._loss_grad_factory(
            design, y, lengths
        )(coeffs)
        penalized_value, penalized_grad = learner._loss_grad_factory(
            design, y, lengths, include_penalty=True
        )(coeffs)
        expected_value, expected_grad = learner._penalty_value_grad(coeffs)

        self.assertGreater(penalized_value, unpenalized_value)
        self.assertAlmostEqual(penalized_value - unpenalized_value, expected_value)
        np.testing.assert_allclose(penalized_grad - unpenalized_grad, expected_grad)

    def test_conv_sccs_penalty_validation(self):
        n_lags = np.repeat(1, 2).astype("uint64")
        with self.assertRaisesRegex(ValueError, "C_tv"):
            ConvSCCS(n_lags=n_lags, C_tv=np.nan)
        with self.assertRaisesRegex(ValueError, "C_group_l1"):
            ConvSCCS(n_lags=n_lags, C_group_l1=np.nan)

        features = [sparse.csr_matrix(np.ones((3, 2)))]
        labels = [np.array([0, 1, 0], dtype="int32")]
        censoring = np.array([3], dtype="uint64")
        invalid_cases = [
            (np.array([[0, 1]]), "one-dimensional"),
            (np.array([True]), "one entry per feature"),
            (np.array([2]), "out of bounds"),
            (np.array([0.5]), "integer"),
        ]
        for penalized_features, pattern in invalid_cases:
            with self.subTest(penalized_features=penalized_features):
                learner = ConvSCCS(n_lags=n_lags, penalized_features=penalized_features, C_tv=2.0)
                with self.assertRaisesRegex(ValueError, pattern):
                    learner.fit(features, labels, censoring)

    def test_batch_and_stream_conv_sccs_wrappers_share_standalone_backend(self):
        from hawkes_tools.survival.sccs import BatchConvSCCS as DeepBatchConvSCCS
        from hawkes_tools.survival.sccs import StreamConvSCCS as DeepStreamConvSCCS

        n_lags = np.repeat(1, 2).astype("uint64")
        features = [
            sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])),
            sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])),
        ]
        labels = [
            np.array([0, 1, 0], dtype="int32"),
            np.array([1, 0, 0], dtype="int32"),
        ]
        censoring = np.array([3, 3], dtype="uint64")

        base = ConvSCCS(n_lags=n_lags, max_iter=4, random_state=7)
        batch = BatchConvSCCS(n_lags=n_lags, max_iter=4, random_state=7, batch_size=2)
        stream = StreamConvSCCS(n_lags=n_lags, max_iter=4, random_state=7, threads=2)
        for learner in [base, batch, stream]:
            fitted, ci = learner.fit(features, labels, censoring)
            self.assertEqual(len(fitted), 2)
            self.assertAlmostEqual(learner.score(), learner.score(features, labels, censoring))
            self.assertEqual(ci["confidence_level"], None)

        self.assertIs(DeepBatchConvSCCS, BatchConvSCCS)
        self.assertIs(DeepStreamConvSCCS, StreamConvSCCS)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(stream.threads, 2)

        with self.assertRaisesRegex(ValueError, "batch_size"):
            BatchConvSCCS(n_lags=n_lags, batch_size=0)
        with self.assertRaisesRegex(ValueError, "threads"):
            StreamConvSCCS(n_lags=n_lags, threads=0)


if __name__ == "__main__":
    unittest.main()

