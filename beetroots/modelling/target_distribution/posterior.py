from typing import Dict, Optional, Tuple, Union

from beetroots.modelling.target_distribution.abstract_target_distribution import TargetDistribution
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood
from beetroots.modelling.priors.smooth_indicator_prior import SmoothIndicatorPrior
from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior

try:
    import cupy as xp
except:
    import numpy as xp


class Posterior(TargetDistribution): #TODO: generalize for any number of likelihoods or priors.

    __slots__ = (
        "D",
        "L",
        "N",
        "likelihood",
        "prior_spatial",
        "prior_indicator",
        "dict_sites",
        "var_shape",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        likelihood: Likelihood,
        var_name: str,
        prior_spatial: Optional[Union[None, SpatialPrior]] = None,
        prior_indicator: Optional[Union[None, SmoothIndicatorPrior]] = None,
        separable: bool = True,
        dict_sites: Optional[Dict[int, xp.ndarray]] = None,
    ):
        distribution_components = {'likelihood':likelihood, 'prior_spatial': prior_spatial, 'prior_indicator': prior_indicator} # not filtered for None
        super().__init__(D, L, N, var_name, {key: val for key, val in distribution_components.items() if val is not None}, separable)

        self.likelihood = likelihood
        """Likelihood: data-fidelity term"""

        self.prior_spatial = prior_spatial
        """SpatialPrior: spatial prior term"""

        self.prior_indicator = prior_indicator
        """SmoothIndicatorPrior: prior term encoding validity intervals"""

        self.dict_sites = {}
        """dict[int, xp.ndarray]: sites for pixels to be sampled in parallel in the MTM-chromoatic Gibbs kernel"""
        if dict_sites is not None:
            self.dict_sites = dict_sites
        elif self.prior_spatial is not None:
            self.dict_sites = self.prior_spatial.dict_sites
        elif separable is True: # all terms are independent (separable)
            self.dict_sites = {0: xp.arange(self.N)}
        else:
            self.dict_sites = {n: xp.array([n]) for n in range(self.N)}

        self.var_shape = (self.N, self.D)

        return

    def neglog_pdf_priors(
        self,
        pixelwise: bool = False,
    ) -> Union[float, xp.ndarray]:
        if pixelwise:
            size_ = self.likelihood.nlpdf_utils['n_pix']
            nl_priors = xp.zeros((size_, self.L)) if not self.likelihood.nlpdf_utils['mtm'] else xp.zeros((size_, self.likelihood.nlpdf_utils['k_mtm'], self.L))
        else:
            nl_priors = 0.0 if not self.likelihood.nlpdf_utils['mtm'] else xp.zeros((self.likelihood.nlpdf_utils['k_mtm']))

        if self.prior_spatial is not None:
            nl_prior_spatial = self.prior_spatial.neglog_pdf(pixelwise=pixelwise)
            if pixelwise:
                # nl_prior_spatial has shape (N, D), which needs to be
                # converted to (N, L)
                nl_prior_spatial = xp.sum(nl_prior_spatial, axis=1)  # (N,)
                nl_priors += nl_prior_spatial[..., None]  # (N, L)
            else:
                nl_priors += xp.sum(nl_prior_spatial)

        if self.prior_indicator is not None:
            nl_prior_ind = self.prior_indicator.neglog_pdf(pixelwise=pixelwise)
            if pixelwise:
                nl_priors += nl_prior_ind[..., None]
            else:
                nl_priors += xp.sum(nl_prior_ind)

        return nl_priors

    def neglog_pdf(
        self,
        current: dict[str, Union[dict, float, xp.ndarray]]=None,
        idx_pix: Optional[xp.ndarray] = None,
        pixelwise: bool = False,
        update_nlpdf_utils: bool = True,
    ) -> float:
        if update_nlpdf_utils and current is None:
            raise ValueError("current is None, cannot update nlpdf_utils")
        elif update_nlpdf_utils and current is not None:
            self.update_nlpdf_utils(current, idx_pix=idx_pix, compute_derivatives=False, compute_derivatives_2nd_order=False)

        if pixelwise:
            size_ = self.N if idx_pix is None else idx_pix.size
            out = xp.zeros((size_, self.L)) if not self.likelihood.nlpdf_utils['mtm'] else xp.zeros((size_, self.likelihood.nlpdf_utils['k_mtm'], self.L))
        else:
            out = 0.0 if not self.likelihood.nlpdf_utils['mtm'] else xp.zeros((self.likelihood.nlpdf_utils['k_mtm']))

        out += self.likelihood.neglog_pdf(pixelwise=pixelwise,) # NOTE: idx_pix not required because it is only required in the evaluate_all_nlpdf_utils method. It computes everything while taking care of nlpdf_utils.

        out += self.neglog_pdf_priors(pixelwise=pixelwise)

        # assert xp.sum(xp.isnan(nll)) == 0, xp.sum(xp.isnan(nll))
        # assert xp.sum(xp.isnan(nl_priors)) == 0, xp.sum(xp.isnan(nl_priors)) 
        return out

    def grad_neglog_pdf(
        self,
        current: dict[dict[str, xp.ndarray]]=None,
        idx_pix: Optional[xp.ndarray] = None,
        update_nlpdf_utils: bool = True,
    ) -> xp.ndarray:
        if update_nlpdf_utils and current is None:
            raise ValueError("current is None, cannot update nlpdf_utils")
        elif update_nlpdf_utils and current is not None:
            self.update_nlpdf_utils(current, idx_pix=idx_pix, compute_derivatives=True, compute_derivatives_2nd_order=False)

        grad_ = self.likelihood.gradient_neglog_pdf()  # (N, D)
        # assert grad_.shape == (self.N, self.D), grad_nll.shape

        if self.prior_spatial is not None:
            # grad_nl_prior_spatial = self.prior_spatial.gradient_neglog_pdf()
            # assert grad_nl_prior_spatial.shape == (self.N, self.D)
            # assert (
            #     xp.sum(xp.isnan(grad_nl_prior_spatial)) == 0
            # ), f"nan grad prior spatial {xp.sum(xp.isnan(grad_nl_prior_spatial))}"
            grad_ += self.prior_spatial.gradient_neglog_pdf()

        if self.prior_indicator is not None:
            # grad_nl_prior_indicator = self.prior_indicator.gradient_neglog_pdf()
            # assert grad_nl_prior_indicator.shape == (self.N, self.D)
            # assert (
            #     xp.sum(xp.isnan(grad_nl_prior_indicator)) == 0
            # ), f"nan grad prior indicator {xp.sum(xp.isnan(grad_nl_prior_indicator))}"
            grad_ += self.prior_indicator.gradient_neglog_pdf()

        grad_ = xp.nan_to_num(grad_)
        return grad_

    def hess_diag_neglog_pdf(
        self,
        current: dict[dict[str, xp.ndarray]] = None,
        idx_pix: Optional[xp.ndarray] = None,
        update_nlpdf_utils: bool = True,
    ) -> xp.ndarray:
        if update_nlpdf_utils and current is None:
            raise ValueError("current is None, cannot update nlpdf_utils")
        elif update_nlpdf_utils and current is not None:
            self.update_nlpdf_utils(current, idx_pix=idx_pix, compute_derivatives=True, compute_derivatives_2nd_order=True)
    
        hess_diag = self.likelihood.hess_diag_neglog_pdf()  # (N, D)
        # assert hess_diag.shape == (self.N, self.D)

        if self.prior_spatial is not None:
            # hess_diag_nl_prior_spatial = self.prior_spatial.hess_diag_neglog_pdf()
            # assert xp.sum(xp.isnan(hess_diag_nl_prior_spatial)) == 0
            # assert hess_diag_nl_prior_spatial.shape == (self.N, self.D)
            hess_diag += self.prior_spatial.hess_diag_neglog_pdf()

        if self.prior_indicator is not None:
            # hess_diag_nl_prior_indicator = self.prior_indicator.hess_diag_neglog_pdf()
            # assert xp.sum(xp.isnan(hess_diag_nl_prior_indicator)) == 0
            # assert hess_diag_nl_prior_indicator.shape == (self.N, self.D)
            hess_diag += self.prior_indicator.hess_diag_neglog_pdf()

        hess_diag = xp.nan_to_num(hess_diag)
        return hess_diag

    def compute_all_for_saver(
        self,
        current: dict[str, dict], 
    ) -> Tuple[dict[str, Union[float, xp.ndarray]], xp.ndarray]:
        """computes negative log pdf of likelihood, priors and posterior (detailed values to be saved, not to be used in sampling)

        Parameters
        ----------
        Theta : xp.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, Union[float, xp.ndarray]]
            output of the ``likelihood.evaluate_all_forward_map()`` method
        nll_utils : [str, Union[float, xp.ndarray]]
            output of the ``likelihood.evaluate_all_nll_utils()`` method

        Returns
        -------
        dict[str, Union[float, xp.ndarray]]
            values to be saved
        """
        # TODO: Adapt to the new implementation
        assert xp.sum(xp.isnan(current[self.var_name]["var"])) == 0, xp.sum(xp.isnan(current[self.var_name]["var"]))
        dict_objective = dict()

        nll_full = self.likelihood.neglog_pdf(
            full=True,
        )  # (N, L)

        assert isinstance(
            nll_full, xp.ndarray
        ), "nll_full shoud be an array, check likelihood.neglog_pdf method"
        assert nll_full.shape == (
            self.N,
            self.L,
        ), f"nll_full with wrong shape. is {nll_full.shape}, should be {(self.N, self.L)}"

        dict_objective["nll"] = xp.sum(nll_full)  # float

        if self.prior_spatial is not None:
            nl_prior_spatial = self.prior_spatial.neglog_pdf()
            dict_objective["nl_prior_spatial"] = nl_prior_spatial  # (D,)
        else:
            nl_prior_spatial = xp.zeros((self.D,))

        if self.prior_indicator is not None:
            nl_prior_indicator = self.prior_indicator.neglog_pdf()
            dict_objective["nl_prior_indicator"] = nl_prior_indicator  # (D,)
        else:
            nl_prior_indicator = xp.zeros((self.D,))

        nl_posterior = xp.sum(nll_full) + xp.sum(nl_prior_spatial)
        nl_posterior += xp.sum(nl_prior_indicator)
        dict_objective["objective"] = nl_posterior

        return dict_objective, nll_full

    def compute_all(
        self,
        current: dict[str, dict], 
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        update_nlpdf_utils: bool = True,
    ) -> dict:
        r"""compute negative log pdf and derivatives of the posterior distribution

        Parameters
        ----------
        Theta : xp.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, xp.ndarray], optional
            output of the ``likelihood.evaluate_all_forward_map()`` method, by default {}
        nll_utils : dict[str, xp.ndarray], optional
            output of the ``likelihood.evaluate_all_nll_utils()`` method, by default {}
        compute_derivatives : bool, optional
            wether to compte derivatives, by default True

        Returns
        -------
        dict[str, Union[float, xp.ndarray]]
            negative log pdf and derivatives of the posterior distribution
        """
        # TODO: adapt to current_sampler instead of Theta, forward_map_evals, nll_utils
        assert xp.sum(xp.isnan(current[self.var_name]["var"])) == 0, xp.sum(xp.isnan(current[self.var_name]["var"]))

        if update_nlpdf_utils:
            self.update_nlpdf_utils(current, compute_derivatives=compute_derivatives, compute_derivatives_2nd_order=compute_derivatives_2nd_order)

        nlpdf_utils = {}
        nll = self.likelihood.neglog_pdf()
        nlpdf_utils["nll"] = nll  # float

        if self.prior_indicator is not None:
            nl_prior_indicator = self.prior_indicator.neglog_pdf()
        else:
            nl_prior_indicator = xp.zeros((self.D,))
        nlpdf_utils["nl_prior_indicator"] = nl_prior_indicator

        nlpdf_pixelwise = self.neglog_pdf(
            current,
            full=True,
            update_nlpdf_utils=False
        )  # (N, L)
        nlpdf_pix = xp.sum(nlpdf_pixelwise, axis=1)

        iterate = {
            "var": current[self.var_name]["var"],
            "forward_map_evals": self.likelihood.forward_map_evals,
            "nll_utils": nlpdf_utils,
            "objective_pix": nlpdf_pix,
        }

        iterate["objective"] = xp.sum(nlpdf_pix)

        if compute_derivatives:
            iterate["grad"] = self.grad_neglog_pdf(update_nlpdf_utils=False)
            if compute_derivatives_2nd_order:
                iterate["hess_diag"] = self.hess_diag_neglog_pdf(update_nlpdf_utils=False)
            else:
                iterate['hess_diag'] = None

        return iterate
