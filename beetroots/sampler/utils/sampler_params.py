"""Defines a Dataclass that stores the parameters of the augmented PSGLD sampler
"""
from typing import List, Union

try:
    import cupy as xp
except:
    import numpy as xp

import importlib
import pandas as pd
import beetroots.sampler.proposal_mtm as proposal_mtm


class MySamplerParams(object):
    r"""Dataclass that stores the parameters of the sampler proposed in :cite:t:`paludEfficientSamplingNon2023`"""

    __slots__ = (
        "initial_step_size",
        "extreme_grad",
        "history_weight",
        "selection_probas",
        "k_mtm",
        "is_stochastic",
        "compute_correction_term",
    )

    def __init__(
        self,
        initial_step_size: float,
        extreme_grad: float,
        history_weight: float,
        selection_probas: Union[xp.ndarray, List[float]],
        k_mtm: int,
        is_stochastic: bool = True,
        compute_correction_term: bool = True,
    ) -> None:
        r"""

        Parameters
        ----------
        initial_step_size : float
            step size used in the Position-dependent MALA transition kernel, denoted :math:`\epsilon` in the article
        extreme_grad : float
            limit value that avoids division by zero when computing the RMSProp preconditioner, denoted :math:`\eta` in the article
        history_weight : float
            weight of past values of :math:`v` in the exponential decay (cf RMSProp preconditioner), denoted :math:`\alpha` in the article
        selection_probas : xp.ndarray of shape (2,)
            vector of selection probabilities for the MTM and PMALA kernels, respectively, i.e., :math:`[p_{MTM}, 1 - p_{MTM}]`
        k_mtm : int
            number of candidates in the MTM kernel, denoted :math:`K` in the article
        is_stochastic : bool
            if True, the algorithm performs sampling, and optimization otherwise, by default True
        compute_correction_term : bool
            wether or not to use the correction term (denoted :math:`\gamma` in the article) during the sampling (only used if `is_stochastic=True`), by default True
        """
        assert initial_step_size > 0
        assert extreme_grad > 0
        assert 0 < history_weight < 1

        assert isinstance(k_mtm, int) and k_mtm >= 1

        if isinstance(selection_probas, list):
            selection_probas = xp.array(selection_probas)

        assert xp.all(0 <= selection_probas)
        assert selection_probas.sum() == 1

        assert isinstance(is_stochastic, bool)
        assert isinstance(compute_correction_term, bool)

        self.initial_step_size = initial_step_size
        self.extreme_grad = extreme_grad
        self.history_weight = history_weight

        self.selection_probas = selection_probas
        self.k_mtm = k_mtm

        self.is_stochastic = is_stochastic
        self.compute_correction_term = compute_correction_term

class MyGibbsSamplerParams(object):
    r"""Dataclass that stores the parameters of the sampler proposed in :cite:t:`paludEfficientSamplingNon2023`"""

    __slots__ = (
        "initial_step_size",
        "extreme_grad",
        "history_weight",
        "selection_probas",
        "k_mtm",
        "is_stochastic",
        "compute_correction_term",
        "proposal_distributions_mtm",
    )

    def __init__(
        self,
        initial_step_size: Union[dict[float], float],
        extreme_grad: Union[dict[float], float],
        history_weight: Union[dict[float], float],
        selection_probas: Union[dict[xp.ndarray], dict[List[float]], xp.ndarray, List[float]],
        k_mtm: Union[dict[int], int],
        proposal_distributions_mtm_params: dict[str],
        is_stochastic: bool = True,
        compute_correction_term: Union[dict[bool], bool] = True,
    ) -> None:
        r"""

        Parameters
        ----------
        initial_step_size : dict[float]
            step size used in the Position-dependent MALA transition kernel, denoted :math:`\epsilon` in the article
        extreme_grad : dict[float]
            limit value that avoids division by zero when computing the RMSProp preconditioner, denoted :math:`\eta` in the article
        history_weight : dict[float]
            weight of past values of :math:`v` in the exponential decay (cf RMSProp preconditioner), denoted :math:`\alpha` in the article
        selection_probas : Union[dict[xp.ndarray], dict[List[float]]], xp.ndarray or list of shape (2,)
            vector of selection probabilities for the MTM and PMALA kernels, respectively, i.e., :math:`[p_{MTM}, 1 - p_{MTM}]`
        k_mtm : dict[int]
            number of candidates in the MTM kernel, denoted :math:`K` in the article
        is_stochastic : bool
            if True, the algorithm performs sampling, and optimization otherwise, by default True
        compute_correction_term : bool
            wether or not to use the correction term (denoted :math:`\gamma` in the article) during the sampling (only used if `is_stochastic=True`), by default True,
        generate_function: str
            name of the function to generate the random start in the MTM kernel.
        **kwargs : dict
        """
        # CHECK CONDITIONS
        assert isinstance(initial_step_size, dict) or isinstance(initial_step_size, float)
        if isinstance(initial_step_size, float):
            initial_step_size = {k: initial_step_size for k in proposal_distributions_mtm_params.keys()}

        for v in initial_step_size.values():
            assert v > 0

        assert isinstance(extreme_grad, dict) or isinstance(extreme_grad, float)
        if isinstance(extreme_grad, float):
            extreme_grad = {k: extreme_grad for k in proposal_distributions_mtm_params.keys()}
        for v in extreme_grad.values():
            assert v > 0

        assert isinstance(history_weight, dict) or isinstance(history_weight, float)
        if isinstance(history_weight, float):
            history_weight = {k: history_weight for k in proposal_distributions_mtm_params.keys()}
        for v in history_weight.values():
            assert 0 < v < 1

        assert isinstance(k_mtm, dict) or isinstance(k_mtm, int)
        if isinstance(k_mtm, int):
            k_mtm = {k: k_mtm for k in proposal_distributions_mtm_params.keys()}
        for v in k_mtm.values():
            assert isinstance(v, int) and v >= 1

        assert isinstance(selection_probas, dict) or isinstance(selection_probas, xp.ndarray) or isinstance(selection_probas, list)
        if isinstance(selection_probas, xp.ndarray) or isinstance(selection_probas, list):
            selection_probas = {k: selection_probas for k in proposal_distributions_mtm_params.keys()}
        for v in selection_probas.values():
            if isinstance(v, list):
                v = xp.array(v)
            assert xp.all(0 <= v)
            assert v.sum() == 1, f"{v} should sum to 1"

        assert isinstance(is_stochastic, bool) or isinstance(is_stochastic, dict)
        if isinstance(is_stochastic, bool):
            is_stochastic = {k: is_stochastic for k in proposal_distributions_mtm_params.keys()}
        for v in is_stochastic.values():
            assert isinstance(v, bool)

        assert isinstance(compute_correction_term, bool)

        self.initial_step_size = initial_step_size
        self.extreme_grad = extreme_grad
        self.history_weight = history_weight

        self.selection_probas = selection_probas
        self.k_mtm = k_mtm

        self.is_stochastic = is_stochastic
        self.compute_correction_term = compute_correction_term

        self.proposal_distributions_mtm = dict()
        module_name = proposal_mtm.__name__
        for key, dict_target in proposal_distributions_mtm_params.items():
            complete_module_name = f"{module_name}.{dict_target['module_name']}"
            module_ = importlib.import_module(complete_module_name)
            class_ = getattr(module_, dict_target['class_name'])
            kwargs = dict_target['params']
            self.proposal_distributions_mtm[key] = class_(**kwargs)
