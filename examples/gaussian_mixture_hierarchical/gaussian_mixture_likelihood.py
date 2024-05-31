"""Implementation of a Gaussian mixture model likelihood
"""

from typing import Optional, Union

import numba as nb
try:
    import cupy as xp
    decorator_nb = nb.cuda.jit
except:
    import numpy as xp
    decorator_nb = nb.njit

from beetroots.modelling.forward_maps.abstract_base import ForwardMap
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood


@decorator_nb(nopython=True, cache=True)
def u_i(Var: xp.ndarray, mu_i: xp.ndarray, cov_i_inv: xp.ndarray) -> xp.ndarray:
    u_ = xp.exp(-0.5 * xp.dot(Var - mu_i, xp.dot(cov_i_inv, Var - mu_i)))
    u_ /= xp.linalg.det(cov_i_inv) ** 0.5
    return u_


@decorator_nb(nopython=True, cache=True)
def grad_u_i(Var: xp.ndarray, mu_i: xp.ndarray, cov_i_inv: xp.ndarray) -> xp.ndarray:
    return -xp.dot(cov_i_inv, Var - mu_i) * u_i(Var, mu_i, cov_i_inv)  # (N, D)


def hess_u_i(Var: xp.ndarray, mu_i: xp.ndarray, cov_i_inv: xp.ndarray) -> xp.ndarray:
    result_1 = -(Var - mu_i) * grad_u_i(Var, mu_i, cov_i_inv)
    result_2 = -u_i(Var, mu_i, cov_i_inv) * xp.diag(cov_i_inv)
    result_1 = result_1.reshape((1, 2))
    result_2 = result_2.reshape((1, 2))
    assert result_1.shape == (1, 2), result_1.shape
    assert result_2.shape == (1, 2), result_2.shape
    return result_1 + result_2


class GaussianMixtureLikelihood(Likelihood):
    """Class implementing a likelihood a Gaussian Mixture model"""

    __slots__ = (
        "forward_map",
        "D",
        "L",
        "N",
        "y",
        "n_means",
        "list_means",
        "list_cov_inv",
        "var_name"
    )

    def __init__(
        self,
        forward_map: ForwardMap,
        D: int,
        list_means: xp.ndarray,
        list_cov: xp.ndarray,
        var_name: str,
    ) -> None:
        """Constructor of the MixingGaussianLikelihood object.

        Parameters
        ----------
        forward_map : ForwardMap instance
            forward map
        D : int
            number of disinct physical parameters in input space.
        L : int
            number of distinct observed physical parameters.
        N : int
            number of pixels in each physical dimension
        y : xp.ndarray of shape (N, L)
            mean of the gaussian distribution
        sigma : float or xp.ndarray of shape (N, L)
            variance of the Gaussian distribution

        Raises
        ------
        ValueError
            y must have the shape (N, L)
        """
        L = D * 1  # make sure that theta space and y space are equal
        N = 1  # force only "one pixel"
        y = xp.zeros((N, L))
        super().__init__(forward_map, D, L, N, y, var_name)

        assert isinstance(list_means, xp.ndarray)
        assert list_means.shape[1] == self.D
        self.n_means = list_means.shape[0]
        self.list_means = list_means

        self.list_cov_inv = xp.linalg.inv(list_cov)

        self.nlpdf_utils = {}

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
    ) -> Union[float, xp.ndarray]:
        
        N_pix = self.nlpdf_utils['N_pix']

        nlpdf = -xp.log(
            xp.sum(
                [
                    [
                        u_i(f_Var, mu, self.list_cov_inv[i])
                        for i, mu in enumerate(self.list_means)
                    ]
                    for f_Var in self.forward_map_evals["f_Var"]
                ],
                axis=1,
            )
        )
        msg = f"should be ({N_pix},), is {nlpdf.shape}"
        assert nlpdf.shape == (N_pix,), msg

        if self.nlpdf_utils['k_mtm'] > 0:
            nlpdf = nlpdf.reshape((self.nlpdf_utils['n_pix'], self.nlpdf_utils['k_mtm'],))

        if pixelwise:
            return nlpdf  # (n_pix,) or (n_pix, k_mtm)
        if full:
            return 1 / self.D * xp.ones((*nlpdf.shape, self.L)) * nlpdf[..., None]  # (N_pix, L)

        return xp.sum(nlpdf)  # float

    def gradient_neglog_pdf(
        self,
    ) -> xp.ndarray:
        u = xp.sum(
            [
                [
                    u_i(f_Var, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Var in self.forward_map_evals["f_Var"]
            ],
            # axis=1,
        )  # float

        sum_grad_u_i = xp.sum(
            [
                [
                    grad_u_i(f_Var, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Var in self.forward_map_evals["f_Var"]
            ],
            axis=1,
        )  # (N_pix, D) = (1, 2)

        grad_ = -1 / u * sum_grad_u_i  # (N_pix, D) = (1, 2)
        # print(u.shape, sum_grad_u_i.shape, grad_.shape)

        assert grad_.shape == (
            self.N,
            self.D,
        ), f"has shape {grad_.shape}, should have ({self.N}, {self.D})"
        return grad_  # (N, D)

    def hess_diag_neglog_pdf(
        self,
    ) -> xp.ndarray:
        u = xp.sum(
            [
                [
                    u_i(f_Var, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Var in self.forward_map_evals["f_Var"]
            ],
            # axis=1,
        )  # float

        sum_grad_u_i = xp.sum(
            [
                [
                    grad_u_i(f_Var, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Var in self.forward_map_evals["f_Var"]
            ],
            axis=1,
        )  # (N_pix, D) = (1, 2)

        sum_hess_diag_u_i = xp.sum(
            [
                [
                    hess_u_i(f_Var, mu, self.list_cov_inv[i])
                    for i, mu in enumerate(self.list_means)
                ]
                for f_Var in self.forward_map_evals["f_Var"]
            ],
            axis=(1, 2),
        )  # (N_pix, D) = (1, 2)

        hess_diag = (
            1
            / u**2
            * (u * xp.log(sum_hess_diag_u_i) - sum_grad_u_i**2 * xp.log(sum_grad_u_i))
        )
        # print(sum_grad_u_i.shape, sum_hess_diag_u_i.shape, hess_diag.shape)
        assert hess_diag.shape == (1, 2)
        return hess_diag  # (N, D)

    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
    ) -> dict:
        shape_var = current[self.var_name]['var'].shape
        assert isinstance(mtm, bool), f"mtm should be a boolean, got {type(mtm)}"
        
        self.nlpdf_utils = dict()
        self.nlpdf_utils['mtm'] = mtm

        forward_var_inputs = current[self.var_name]['var']
        if idx_pix is not None:
            forward_var_inputs = forward_var_inputs[idx_pix]
        if mtm:
            forward_var_inputs = forward_var_inputs.reshape(-1, *shape_var[2:])
        self.evaluate_all_forward_map(forward_var_inputs, compute_derivatives=compute_derivatives, compute_derivatives_2nd_order=compute_derivatives_2nd_order) # TODO: put variable and other required arguments here

        n_pix = idx_pix.size if idx_pix is not None else self.N *1
        k_mtm = shape_var.shape[1] if mtm else 0
        if n_pix == self.N:
            idx_pix = xp.arange(self.N)
        N_pix = self.forward_map_evals["f_Var"].shape[0]
        assert n_pix * k_mtm == N_pix

        self.nlpdf_utils['n_pix'] = n_pix
        self.nlpdf_utils['N_pix'] = N_pix
        self.nlpdf_utils['k_mtm'] = k_mtm

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        # to be disregarded, as model checking does not make sense
        # in this example
        return self.forward_map_evals["f_Var"]
