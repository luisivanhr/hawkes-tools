"""Compatibility aliases for historical Hawkes deep import paths."""

from __future__ import annotations

import sys
from types import ModuleType

from . import inference as _inference
from . import kernels as _kernels
from . import models as _models
from . import simulation as _simulation


def _mark_as_package(module: ModuleType) -> None:
    if not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]


def _attach_to_parent(fullname: str, module: ModuleType) -> None:
    parent_name, _, attr = fullname.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attr, module)


def _alias_module(fullname: str, attrs: dict[str, object], *, package: bool = False) -> ModuleType:
    module = sys.modules.get(fullname)
    if module is None:
        module = ModuleType(fullname)
    module.__dict__.update(attrs)
    module.__package__ = fullname.rpartition(".")[0]
    module.__all__ = [name for name in attrs if not name.startswith("_")]  # type: ignore[attr-defined]
    if package:
        _mark_as_package(module)
    sys.modules[fullname] = module
    _attach_to_parent(fullname, module)
    return module


def _single_name(fullname: str, name: str, value: object) -> None:
    _alias_module(fullname, {name: value})


def register_tick_deep_import_aliases() -> None:
    """Expose Hawkes deep modules without duplicating implementation files."""

    package = __name__.rsplit(".", 1)[0]

    sim_prefix = f"{package}.simulation"
    _mark_as_package(_simulation)
    _attach_to_parent(sim_prefix, _simulation)

    sim_base = _alias_module(
        f"{sim_prefix}.base",
        {"SimuPointProcess": _simulation.SimuPointProcess},
        package=True,
    )
    _single_name(f"{sim_base.__name__}.simu_point_process", "SimuPointProcess", _simulation.SimuPointProcess)

    kernel_attrs = {
        "HawkesKernel": _kernels.HawkesKernel,
        "HawkesKernel0": _kernels.HawkesKernel0,
        "HawkesKernelExp": _kernels.HawkesKernelExp,
        "HawkesKernelPowerLaw": _kernels.HawkesKernelPowerLaw,
        "HawkesKernelSumExp": _kernels.HawkesKernelSumExp,
        "HawkesKernelTimeFunc": _kernels.HawkesKernelTimeFunc,
    }
    kernels_package = _alias_module(f"{sim_prefix}.hawkes_kernels", kernel_attrs, package=True)
    for module_name, class_name in {
        "hawkes_kernel": "HawkesKernel",
        "hawkes_kernel_0": "HawkesKernel0",
        "hawkes_kernel_exp": "HawkesKernelExp",
        "hawkes_kernel_power_law": "HawkesKernelPowerLaw",
        "hawkes_kernel_sum_exp": "HawkesKernelSumExp",
        "hawkes_kernel_time_func": "HawkesKernelTimeFunc",
    }.items():
        _single_name(f"{kernels_package.__name__}.{module_name}", class_name, kernel_attrs[class_name])

    for module_name, class_name in {
        "simu_hawkes": "SimuHawkes",
        "simu_hawkes_exp_kernels": "SimuHawkesExpKernels",
        "simu_hawkes_multi": "SimuHawkesMulti",
        "simu_hawkes_sumexp_kernels": "SimuHawkesSumExpKernels",
        "simu_inhomogeneous_poisson": "SimuInhomogeneousPoisson",
        "simu_poisson_process": "SimuPoissonProcess",
    }.items():
        _single_name(f"{sim_prefix}.{module_name}", class_name, getattr(_simulation, class_name))

    model_prefix = f"{package}.model"
    model_attrs = {
        "ModelHawkes": _models.ModelHawkes,
        "ModelHawkesExpKernLeastSq": _models.ModelHawkesExpKernLeastSq,
        "ModelHawkesExpKernLogLik": _models.ModelHawkesExpKernLogLik,
        "ModelHawkesSumExpKernLeastSq": _models.ModelHawkesSumExpKernLeastSq,
        "ModelHawkesSumExpKernLogLik": _models.ModelHawkesSumExpKernLogLik,
    }
    _alias_module(model_prefix, model_attrs, package=True)
    model_base = _alias_module(f"{model_prefix}.base", {"ModelHawkes": _models.ModelHawkes}, package=True)
    _single_name(f"{model_base.__name__}.model_hawkes", "ModelHawkes", _models.ModelHawkes)
    for module_name, class_name in {
        "model_hawkes_expkern_leastsq": "ModelHawkesExpKernLeastSq",
        "model_hawkes_expkern_loglik": "ModelHawkesExpKernLogLik",
        "model_hawkes_sumexpkern_leastsq": "ModelHawkesSumExpKernLeastSq",
        "model_hawkes_sumexpkern_loglik": "ModelHawkesSumExpKernLogLik",
    }.items():
        _single_name(f"{model_prefix}.{module_name}", class_name, model_attrs[class_name])

    inference_prefix = f"{package}.inference"
    _mark_as_package(_inference)
    _attach_to_parent(inference_prefix, _inference)
    inference_base_attrs = {
        "LearnerHawkesNoParam": _inference._LearnerBase,
        "LearnerHawkesParametric": _inference._ParametricHawkesLearner,
    }
    inference_base = _alias_module(f"{inference_prefix}.base", inference_base_attrs, package=True)
    _single_name(
        f"{inference_base.__name__}.learner_hawkes_noparam",
        "LearnerHawkesNoParam",
        _inference._LearnerBase,
    )
    _single_name(
        f"{inference_base.__name__}.learner_hawkes_param",
        "LearnerHawkesParametric",
        _inference._ParametricHawkesLearner,
    )
    for module_name, attrs in {
        "hawkes_adm4": {"HawkesADM4": _inference.HawkesADM4},
        "hawkes_basis_kernels": {"HawkesBasisKernels": _inference.HawkesBasisKernels},
        "hawkes_conditional_law": {"HawkesConditionalLaw": _inference.HawkesConditionalLaw},
        "hawkes_cumulant_matching": {
            "HawkesCumulantMatching": _inference.HawkesCumulantMatching,
            "HawkesCumulantMatchingPyT": _inference.HawkesCumulantMatchingPyT,
            "HawkesCumulantMatchingTf": _inference.HawkesCumulantMatchingTf,
            "HawkesTheoreticalCumulant": _inference.HawkesTheoreticalCumulant,
        },
        "hawkes_em": {"HawkesEM": _inference.HawkesEM},
        "hawkes_expkern_fixeddecay": {"HawkesExpKern": _inference.HawkesExpKern},
        "hawkes_sumexpkern_fixeddecay": {"HawkesSumExpKern": _inference.HawkesSumExpKern},
        "hawkes_sumgaussians": {"HawkesSumGaussians": _inference.HawkesSumGaussians},
    }.items():
        _alias_module(f"{inference_prefix}.{module_name}", attrs)
