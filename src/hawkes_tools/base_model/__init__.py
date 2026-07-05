"""Standalone base-model protocols."""

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod

import numpy as np

from hawkes_tools.preprocessing.utils import safe_array

LOSS = "loss"
GRAD = "grad"
LOSS_AND_GRAD = "loss_and_grad"
HESSIAN_NORM = "hessian_norm"

N_CALLS_LOSS = "n_calls_loss"
N_CALLS_GRAD = "n_calls_grad"
N_CALLS_LOSS_AND_GRAD = "n_calls_loss_and_grad"
N_CALLS_HESSIAN_NORM = "n_calls_hessian_norm"
PASS_OVER_DATA = "n_passes_over_data"


class Model(ABC):
    """Abstract zero-order model with fit guards and counters."""

    pass_per_operation = {LOSS: 1}

    def __init__(self):
        self._fitted = False
        self._model = None
        self.n_calls_loss = 0
        self.n_passes_over_data = 0
        self.dtype = None

    def _set(self, name, value):
        setattr(self, name, value)

    def _inc_attr(self, name, step=1):
        setattr(self, name, getattr(self, name) + step)

    def fit(self, *args):
        self._set_data(*args)
        self._set("_fitted", True)
        self._set(N_CALLS_LOSS, 0)
        self._set(PASS_OVER_DATA, 0)
        return self

    @abstractmethod
    def _get_n_coeffs(self) -> int:
        pass

    @property
    def n_coeffs(self):
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``n_coeffs``")
        return self._get_n_coeffs()

    @abstractmethod
    def _set_data(self, *args):
        pass

    def _cast_coeffs(self, coeffs):
        coeffs = np.asarray(coeffs)
        if self.dtype is not None and coeffs.dtype != self.dtype:
            warnings.warn(
                "coeffs vector of type {} has been cast to {}".format(
                    coeffs.dtype, self.dtype
                )
            )
            coeffs = coeffs.astype(self.dtype)
        return coeffs

    def loss(self, coeffs: np.ndarray) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``loss``")
        coeffs = self._cast_coeffs(coeffs)
        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                "``coeffs`` has size %i while the model expects %i coefficients"
                % (coeffs.shape[0], self.n_coeffs)
            )
        self._inc_attr(N_CALLS_LOSS)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[LOSS])
        return self._loss(coeffs)

    @abstractmethod
    def _loss(self, coeffs: np.ndarray) -> float:
        pass

    def astype(self, dtype_or_object_with_dtype):
        dtype = getattr(dtype_or_object_with_dtype, "dtype", dtype_or_object_with_dtype)
        new_model = copy.deepcopy(self)
        new_model.dtype = np.dtype(dtype)
        for name, value in list(new_model.__dict__.items()):
            if isinstance(value, np.ndarray):
                setattr(new_model, name, value.astype(new_model.dtype))
        return new_model


class ModelFirstOrder(Model):
    """Abstract model with first-order gradient information."""

    pass_per_operation = {
        **Model.pass_per_operation,
        GRAD: 1,
        LOSS_AND_GRAD: 2,
    }

    def __init__(self):
        super().__init__()
        self.n_calls_grad = 0
        self.n_calls_loss_and_grad = 0

    def fit(self, *args):
        super().fit(*args)
        self._set(N_CALLS_GRAD, 0)
        self._set(N_CALLS_LOSS_AND_GRAD, 0)
        return self

    def grad(self, coeffs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``grad``")
        coeffs = self._cast_coeffs(coeffs)
        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                "``coeffs`` has size %i while the model expects %i coefficients"
                % (coeffs.shape[0], self.n_coeffs)
            )
        grad = out if out is not None else np.empty(self.n_coeffs, dtype=self.dtype)
        self._inc_attr(N_CALLS_GRAD)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[GRAD])
        self._grad(coeffs, out=grad)
        return grad

    @abstractmethod
    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        pass

    def loss_and_grad(
        self, coeffs: np.ndarray, out: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``loss_and_grad``")
        coeffs = self._cast_coeffs(coeffs)
        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                "``coeffs`` has size %i while the model expects %i coefficients"
                % (coeffs.shape[0], self.n_coeffs)
            )
        grad = out if out is not None else np.empty(self.n_coeffs, dtype=self.dtype)
        self._inc_attr(N_CALLS_LOSS_AND_GRAD)
        self._inc_attr(N_CALLS_LOSS)
        self._inc_attr(N_CALLS_GRAD)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[LOSS_AND_GRAD])
        loss = self._loss_and_grad(coeffs, out=grad)
        return loss, grad

    def _loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray) -> float:
        self._grad(coeffs, out=out)
        return self._loss(coeffs)


class ModelSecondOrder(ModelFirstOrder):
    """Abstract first-order model with Hessian norm information."""

    pass_per_operation = {
        **ModelFirstOrder.pass_per_operation,
        HESSIAN_NORM: 1,
    }

    def __init__(self):
        super().__init__()
        self.n_calls_hessian_norm = 0

    def fit(self, *args):
        super().fit(*args)
        self._set(N_CALLS_HESSIAN_NORM, 0)
        return self

    def hessian_norm(self, coeffs: np.ndarray, point: np.ndarray) -> float:
        if not self._fitted:
            raise Exception("Must must fit data before calling ``hessian_norm``")
        coeffs = self._cast_coeffs(coeffs)
        point = self._cast_coeffs(point)
        if len(coeffs) != self.n_coeffs:
            raise ValueError(
                "``coeffs`` has size %i while the model expects %i coefficients"
                % (len(coeffs), self.n_coeffs)
            )
        if len(point) != self.n_coeffs:
            raise ValueError(
                "``point`` has size %i while the model expects %i coefficients"
                % (len(point), self.n_coeffs)
            )
        self._inc_attr(N_CALLS_HESSIAN_NORM)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[HESSIAN_NORM])
        return self._hessian_norm(coeffs, point)

    @abstractmethod
    def _hessian_norm(self, coeffs: np.ndarray, point: np.ndarray) -> float:
        pass


class ModelLabelsFeatures(Model):
    """Abstract model whose data is a feature matrix and labels vector."""

    def __init__(self):
        super().__init__()
        self.features = None
        self.labels = None
        self.n_features = None
        self.n_samples = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        return super().fit(features, labels)

    def _set_data(self, features, labels):
        self.dtype = np.dtype(features.dtype)
        n_samples, n_features = features.shape
        if n_samples != labels.shape[0]:
            raise ValueError(
                "Features has %i samples while labels have %i"
                % (n_samples, labels.shape[0])
            )
        self._set("features", safe_array(features, dtype=self.dtype))
        self._set("labels", safe_array(labels, dtype=self.dtype))
        self._set("n_features", n_features)
        self._set("n_samples", n_samples)

    @property
    def _epoch_size(self):
        return self.n_samples

    @property
    def _rand_max(self):
        return self.n_samples


class ModelGeneralizedLinear(ModelLabelsFeatures):
    """Abstract generalized linear model with optional intercept."""

    def __init__(self, fit_intercept: bool = True):
        super().__init__()
        self.fit_intercept = bool(fit_intercept)

    def _get_n_coeffs(self):
        if self._model is not None and hasattr(self._model, "get_n_coeffs"):
            return self._model.get_n_coeffs()
        return int(self.n_features + int(self.fit_intercept))


class ModelLipschitz(Model):
    """Abstract model exposing Lipschitz constants."""

    def __init__(self):
        super().__init__()
        self._ready_lip_best = False
        self._lip_best = None

    def fit(self, *args):
        self._set("_ready_lip_best", False)
        self._set("_lip_best", None)
        return super().fit(*args)

    def get_lip_max(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before calling ``get_lip_max``")
        if self._model is None or not hasattr(self._model, "get_lip_max"):
            raise NotImplementedError("get_lip_max requires a backend or subclass")
        return self._model.get_lip_max()

    def get_lip_mean(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``get_lip_max``")
        if self._model is None or not hasattr(self._model, "get_lip_mean"):
            raise NotImplementedError("get_lip_mean requires a backend or subclass")
        return self._model.get_lip_mean()

    def get_lip_best(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before calling ``get_lip_best``")
        if self._ready_lip_best:
            return self._lip_best
        lip_best = self._get_lip_best()
        self._set("_lip_best", lip_best)
        self._set("_ready_lip_best", True)
        return lip_best

    @abstractmethod
    def _get_lip_best(self) -> float:
        pass


class ModelSelfConcordant(Model):
    """Abstract model exposing a self-concordant constant."""

    @property
    def _sc_constant(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``sc_constant``")
        return self._get_sc_constant()

    @abstractmethod
    def _get_sc_constant(self) -> float:
        pass


__all__ = [
    "Model",
    "ModelFirstOrder",
    "ModelSecondOrder",
    "ModelLabelsFeatures",
    "ModelSelfConcordant",
    "ModelGeneralizedLinear",
    "ModelLipschitz",
]
