from typing import Dict, Optional, Union

import numpy as np


class Posterior:

    __slots__ = (
        "D",
        "L",
        "N",
        "likelihood",
        "prior",
        "prior_spatial",
        "prior_indicator",
        "dict_sites",
    )

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        likelihood,
        prior=None,
        prior_spatial=None,
        prior_indicator=None,
        separable: bool = True,
        dict_sites: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.D = D
        """int: number of distinct physical parameters"""

        self.L = L
        """int: number of observables per pixel"""

        self.N = N
        """int: number of pixels"""

        self.likelihood = likelihood
        """Likelihood: data-fidelity term"""

        self.prior = prior
        """Deprecated"""

        self.prior_spatial = prior_spatial
        """SpatialPrior: spatial prior term"""

        self.prior_indicator = prior_indicator
        """SmoothIndicatorPrior: prior term encoding validity intervals"""

        self.dict_sites = {}
        """dict[int, np.ndarray]: sites for pixels to be sampled in parallel in the MTM-chromoatic Gibbs kernel"""
        if dict_sites is not None:
            self.dict_sites = dict_sites
        elif self.prior_spatial is not None:
            self.dict_sites = self.prior_spatial.dict_sites
        elif separable is True:
            self.dict_sites = {0: np.arange(self.N)}
        else:
            self.dict_sites = {n: np.array([n]) for n in range(self.N)}

        return

    def partial_neglog_pdf_priors(
        self,
        Theta: np.ndarray,
        idx_pix: np.ndarray,
        list_pixel_candidates: np.ndarray,
        spatial_weights: Optional[np.ndarray] = None,
        use_indicator_prior: bool = True,
        use_spatial_prior: bool = True
        # compute_indicator: bool = False,
    ) -> np.ndarray:
        r"""computes the neg log-prior when only one pixel is modified

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate
        idx_pix : int
            the index of the pixel to consider (0 <= idx_pix <= N - 1)
        list_pixel_candidates : np.ndarray of shape (N_candidates, D)
            the list of all candidates for pixel idx_pi
        spatial_weights : Optional[np.ndarray], optional
            vector of shape (D,) containing the weights of the spatial prior, by default None
        use_indicator_prior : bool, optional
            wether to use the indicator prior term, by default True
        use_spatial_prior : bool, optional
            wether to use the spatial prior term, by default True

        Returns
        -------
        np.ndarray of shape (N_candidates,)
            the negative log-prior of the candidates
        """
        n_pix, k_mtm, D = list_pixel_candidates.shape
        assert n_pix == idx_pix.size
        assert D == Theta.shape[1]

        nl_priors = np.zeros((n_pix, k_mtm))
        if self.prior_spatial is not None and use_spatial_prior:
            nl_priors_spatial = self.prior_spatial.neglog_pdf_one_pix(
                Theta, idx_pix, list_pixel_candidates, spatial_weights
            )  # (n_pix, k_mtm)
            nl_priors += nl_priors_spatial

        if self.prior_indicator is not None and use_indicator_prior:
            list_pixel_candidates_reshaped = list_pixel_candidates.reshape(
                (n_pix * k_mtm, D)
            )
            nl_priors_indicator = self.prior_indicator.neglog_pdf_one_pix(
                list_pixel_candidates_reshaped
            )
            nl_priors += nl_priors_indicator.reshape((n_pix, k_mtm))
        return nl_priors

    def neglog_pdf_priors(
        self,
        Theta: np.ndarray,
        update_prior: bool = False,
        theta: dict = None,
        full: bool = False,
    ) -> Union[float, np.ndarray]:
        if full:
            nl_priors = np.zeros((self.N, self.L))
        else:
            nl_priors = 0.0

        # TODO: to be revised here (make sure the gradient is properly computed)
        if self.prior is not None:
            # ! plugging Theta as "observations" for the prior (any likelihood object here) (rather inefficient, since it needs to be done each time...)
            if update_prior:
                self.prior._update_observations(Theta)

            nl_priors_prior_full = self.prior.neglog_pdf(theta, {}, full=full)
            if full:
                nl_priors += nl_priors_prior_full
            else:
                nl_priors += np.sum(nl_priors_prior_full)

        if self.prior_spatial is not None:
            nl_prior_spatial = self.prior_spatial.neglog_pdf(Theta, pixelwise=full)
            if full:
                # nl_prior_spatial has shape (N, D), which needs to be
                # converted to (N, D)
                nl_prior_spatial = np.sum(nl_prior_spatial, axis=1)  # (N,)
                nl_priors += nl_prior_spatial[:, None]  # (N, D)
            else:
                nl_priors += np.sum(nl_prior_spatial)

        if self.prior_indicator is not None:
            nl_prior_ind = self.prior_indicator.neglog_pdf(Theta, pixelwise=full)
            if full:
                nl_priors += nl_prior_ind[:, None]
            else:
                nl_priors += np.sum(nl_prior_ind)

        return nl_priors

    def neglog_pdf(
        self,
        Theta: np.ndarray,
        forward_map_evals: dict,
        nll_utils: dict,
        update_prior: bool = False,
        theta: dict = None,
        full: bool = False,
    ) -> float:
        if full:
            out = np.zeros((self.N, self.L))
        else:
            out = 0.0

        out += self.likelihood.neglog_pdf(
            forward_map_evals,
            nll_utils,
            full=full,
        )

        out += self.neglog_pdf_priors(Theta, update_prior, theta, full=full)

        # assert np.sum(np.isnan(nll)) == 0, np.sum(np.isnan(nll))
        # assert np.sum(np.isnan(nl_priors)) == 0, np.sum(np.isnan(nl_priors))

        return out

    def grad_neglog_pdf(
        self,
        Theta: np.ndarray,
        forward_map_evals: dict,
        nll_utils: dict,
        update_prior=False,
        theta: dict = None,  # hyperparameter dictionary
    ) -> np.ndarray:
        grad_ = self.likelihood.gradient_neglog_pdf(
            forward_map_evals, nll_utils
        )  # (N, D)
        # assert grad_.shape == (self.N, self.D), grad_nll.shape

        if self.prior is not None:
            if update_prior:
                self.prior._update_observations(Theta)
            grad_ += self.prior.gradient_variable_neglog_pdf(theta, {})[
                :, None, :
            ]  # (N, 1, L)
            grad_ = np.squeeze(grad_)

        if self.prior_spatial is not None:
            # grad_nl_prior_spatial = self.prior_spatial.gradient_neglog_pdf(Theta)
            # assert grad_nl_prior_spatial.shape == (self.N, self.D)
            # assert (
            #     np.sum(np.isnan(grad_nl_prior_spatial)) == 0
            # ), f"nan grad prior spatial {np.sum(np.isnan(grad_nl_prior_spatial))}"
            grad_ += self.prior_spatial.gradient_neglog_pdf(Theta)

        if self.prior_indicator is not None:
            # grad_nl_prior_indicator = self.prior_indicator.gradient_neglog_pdf(Theta)
            # assert grad_nl_prior_indicator.shape == (self.N, self.D)
            # assert (
            #     np.sum(np.isnan(grad_nl_prior_indicator)) == 0
            # ), f"nan grad prior indicator {np.sum(np.isnan(grad_nl_prior_indicator))}"
            grad_ += self.prior_indicator.gradient_neglog_pdf(Theta)

        grad_ = np.nan_to_num(grad_)
        return grad_

    def hess_diag_neglog_pdf(
        self,
        Theta: np.ndarray,
        forward_map_evals: dict,
        nll_utils: dict,
        update_prior=False,
        theta: dict = None,  # hyperparameter dictionary
    ) -> np.ndarray:
        hess_diag = self.likelihood.hess_diag_neglog_pdf(
            forward_map_evals, nll_utils
        )  # (N, D)
        # assert hess_diag.shape == (self.N, self.D)

        if self.prior is not None:
            if update_prior:
                # ! prior is, in this case, defined using a likelihood object
                self.prior._update_observations(Theta)
            # ! dirty patch to take into account the facet that the forward map is identity (revise the whole code to properly accommodate this option)
            hess_diag += self.prior.hess_diag_variable_neglog_pdf(theta, {})[:, None, :]
            hess_diag = np.squeeze(hess_diag)

        if self.prior_spatial is not None:
            # hess_diag_nl_prior_spatial = self.prior_spatial.hess_diag_neglog_pdf(Theta)
            # assert np.sum(np.isnan(hess_diag_nl_prior_spatial)) == 0
            # assert hess_diag_nl_prior_spatial.shape == (self.N, self.D)
            hess_diag += self.prior_spatial.hess_diag_neglog_pdf(Theta)

        if self.prior_indicator is not None:
            # hess_diag_nl_prior_indicator = self.prior_indicator.hess_diag_neglog_pdf(Theta)
            # assert np.sum(np.isnan(hess_diag_nl_prior_indicator)) == 0
            # assert hess_diag_nl_prior_indicator.shape == (self.N, self.D)
            hess_diag += self.prior_indicator.hess_diag_neglog_pdf(Theta)

        hess_diag = np.nan_to_num(hess_diag)
        return hess_diag

    def compute_all_for_saver(
        self,
        Theta: np.ndarray,
        forward_map_evals: dict,
        nll_utils: dict,
        update_prior=False,
        theta: dict = None,
    ) -> dict:
        """computes negative log pdf of likelihood, priors and posterior (detailed values to be saved, not to be used in sampling)

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, np.ndarray]
            output of the ``likelihood.evaluate_all_forward_map()`` method
        nll_utils : dict[str, np.,ndarray]
            output of the ``likelihood.evaluate_all_nll_utils()`` method

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            values to be saved
        """
        assert Theta.shape == (self.N, self.D)
        dict_objective = dict()

        nll_full = self.likelihood.neglog_pdf(
            forward_map_evals,
            nll_utils,
            full=True,
        )  # (N, L)

        assert isinstance(
            nll_full, np.ndarray
        ), "nll_full shoud be an array, check likelihood.neglog_pdf method"
        assert nll_full.shape == (
            self.N,
            self.L,
        ), f"nll_full with wrong shape. is {nll_full.shape}, should be {(self.N, self.L)}"

        dict_objective["nll"] = np.sum(nll_full)  # float

        if self.prior_spatial is not None:
            nl_prior_spatial = self.prior_spatial.neglog_pdf(Theta)
            dict_objective["nl_prior_spatial"] = nl_prior_spatial  # (D,)
        else:
            nl_prior_spatial = np.zeros((self.D,))

        if self.prior_indicator is not None:
            nl_prior_indicator = self.prior_indicator.neglog_pdf(Theta)
            dict_objective["nl_prior_indicator"] = nl_prior_indicator  # (D,)
        else:
            nl_prior_indicator = np.zeros((self.D,))

        nl_posterior = np.sum(nll_full) + np.sum(nl_prior_spatial)
        nl_posterior += np.sum(nl_prior_indicator)
        dict_objective["objective"] = nl_posterior

        # ! the two methods are equivalent (checked with commented assert)
        # nl_posterior_v2 = self.neglog_pdf(
        #     Theta, forward_map_evals, nll_utils, update_prior, theta
        # )
        # assert np.isclose(
        #     nl_posterior, nl_posterior_v2
        # ), f"{nl_posterior}, {nl_posterior_v2}"

        # dict_objective["objective"] = self.neglog_pdf(
        #     Theta, forward_map_evals, nll_utils, update_prior, theta
        # )

        return dict_objective, nll_full

    def compute_all(
        self,
        Theta: np.ndarray,
        forward_map_evals: dict = {},
        nll_utils: dict = {},
        update_prior: bool = False,
        theta: dict = {},
        compute_derivatives: bool = True,
    ) -> dict:
        r"""compute negative log pdf and derivatives of the posterior distribution

        Parameters
        ----------
        Theta : np.ndarray of shape (N, D)
            current iterate
        forward_map_evals : dict[str, np.ndarray], optional
            output of the ``likelihood.evaluate_all_forward_map()`` method, by default {}
        nll_utils : dict[str, np.ndarray], optional
            output of the ``likelihood.evaluate_all_nll_utils()`` method, by default {}
        update_prior : bool, optional
            Deprecated, by default False
        theta : dict, optional
            Deprecated, by default {}
        compute_derivatives : bool, optional
            wether to compte derivatives, by default True

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            negative log pdf and derivatives of the posterior distribution
        """
        assert np.sum(np.isnan(Theta)) == 0, np.sum(np.isnan(Theta))

        if forward_map_evals == {}:
            forward_map_evals = self.likelihood.evaluate_all_forward_map(
                Theta, compute_derivatives
            )

        if nll_utils == {}:
            nll_utils = self.likelihood.evaluate_all_nll_utils(
                forward_map_evals, None, compute_derivatives
            )

        nll = self.likelihood.neglog_pdf(forward_map_evals, nll_utils)
        nll_utils["nll"] = nll  # float

        if self.prior is not None:
            # need to evaluate the prior, using the hyperparameters theta
            if update_prior:
                self.prior._update_observations(Theta)
                # self.prior._update_observations(theta["f_Theta"])
                assert np.sum(np.isnan(self.prior.logy)) == 0, f"{Theta}"
                assert np.all(self.prior.y > 0), f"{Theta}"

            nl_prior = self.prior.neglog_pdf(theta, {})
            nll_utils["nl_prior"] = nl_prior  # (D,)

        if self.prior_indicator is not None:
            nl_prior_indicator = self.prior_indicator.neglog_pdf(Theta)
        else:
            nl_prior_indicator = np.zeros((self.D,))
        nll_utils["nl_prior_indicator"] = nl_prior_indicator

        nlpdf_full = self.neglog_pdf(
            Theta,
            forward_map_evals,
            nll_utils,
            update_prior=False,
            theta=theta,
            full=True,
        )  # (N, L)
        nlpdf_pix = np.sum(nlpdf_full, axis=1)

        iterate = {
            "Theta": Theta,
            "forward_map_evals": forward_map_evals,
            "theta": theta,
            "nll_utils": nll_utils,
            "objective_pix": nlpdf_pix,
        }
        iterate["objective"] = np.sum(nlpdf_pix)
        if compute_derivatives:
            iterate["grad"] = self.grad_neglog_pdf(
                Theta,
                forward_map_evals,
                nll_utils,
                update_prior=False,
                theta=theta,
            )
            iterate["hess_diag"] = self.hess_diag_neglog_pdf(
                Theta,
                forward_map_evals,
                nll_utils,
                update_prior=False,
                theta=theta,
            )

        return iterate
