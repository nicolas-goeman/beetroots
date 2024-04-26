from beetroots.modelling.hierarchical.abstract_hierarchical import Hierarchical
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood
from beetroots.modelling.likelihoods import utils
from typing import Optional, Union

try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed
from scipy.special import log_ndtr


class AuxiliaryGivenTarget(Likelihood):
    r"""Class implementing the conditional distribution :math:`U|\Theta` for both full conditional distributions :math:`\pi(U|\Theta, Y)` and :math:`\pi(\Theta|U, Y)`."""

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        differentiate_auxiliary: bool, # it appears in both conditionals, choose which variable you differentiate with respect to
        sigma_m: Union[float, xp.ndarray],
    ) -> None:
        super().__init__(forward_map, D, L, N)

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = sigma_m * xp.ones((N, L))
        else:
            assert sigma_m.shape == (N, L)
            self.sigma_m = sigma_m
        self.sigma_m2 = xp.square(self.sigma_m)

        if isinstance(differentiate_auxiliary, bool):
            self.differentiate_auxiliary = differentiate_auxiliary

    def neglog_pdf(
        self,
        forward_map_evals: dict,
        nlpdf_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out = nlpdf_utils['log_aux']
        out += xp.square(nlpdf_utils['log_aux'] + self.sigma_m2/2 - forward_map_evals['log_f_Theta'])/(2*self.sigma_m2)
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def gradient_neglog_pdf_wrt_aux(
        self,
        forward_map_evals: dict,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += 3/2 + (nlpdf_utils['log_aux'] - forward_map_evals['log_f_Theta'])/self.sigma_m2
        out /= nlpdf_utils['aux']

        return out
    
    def gradient_neglog_pdf_wrt_target(
        self,
        forward_map_evals: dict,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out -= forward_map_evals['grad_log_f_Theta']*(nlpdf_utils['log_aux'] + self.sigma_m2/2 - forward_map_evals['log_f_Theta'])
        out /= self.sigma_m2

        return out
            
    def gradient_neglog_pdf(
        self,
        forward_map_evals: dict,
        nlpdf_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
         
        if self.differentiate_auxiliary:
            out = self.gradient_neglog_pdf_wrt_aux(forward_map_evals, nlpdf_utils, pixelwise, full, idx)
        else:
            out = self.gradient_neglog_pdf_wrt_target(forward_map_evals, nlpdf_utils, pixelwise, full, idx)

        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def hess_diag_neglog_pdf_wrt_aux(
        self,
        forward_map_evals: dict,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += 3/2 + (nlpdf_utils['log_aux'] - forward_map_evals['log_f_Theta']- 1)/self.sigma_m2
        out /= -nlpdf_utils['aux2']

        return out

    def hess_diag_neglog_pdf_wrt_target(
        self,
        forward_map_evals: dict,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += forward_map_evals['hess_diag_log_f_Theta']*(nlpdf_utils['log_aux'] + self.sigma_m2/2 - forward_map_evals['log_f_Theta'])
        out -= forward_map_evals['grad_log_f_Theta']**2
        out /= self.sigma_m2

        return out
    
    def hess_diag_neglog_pdf(
        self,
        forward_map_evals: dict,
        nlpdf_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        
        if self.differentiate_auxiliary:
            out = self.hess_neglog_pdf_wrt_aux(forward_map_evals, nlpdf_utils, pixelwise, full, idx)
        else:
            out = self.hess_neglog_pdf_wrt_target(forward_map_evals, nlpdf_utils, pixelwise, full, idx)

        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def get_nlpdf_utils_keys(self):
        return ["log_aux", "aux", "aux2"]
    
    def evaluate_all_nlpdf_utils(
        self,
        forward_map_evals: dict,
        idx: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        nll_utils = {}
        return nll_utils

    def sample_observation_model(
        self,
        forward_map_evals: dict,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        
        out = forward_map_evals["log_f_Theta"] - xp.square(self.sigma_m)/2 + self.sigma_m*rng.normal(loc=0, scale=1, size=forward_map_evals["log_f_Theta"].shape), # scale is the std not variance, better to put everything outside to avoid mistakes
        out = xp.exp(out)
    
        return out

    # def evaluate_all_forward_map(
    #     self,
    #     Theta: xp.ndarray,
    #     compute_derivatives: bool,
    #     compute_derivatives_2nd_order: bool = True,
    # ) -> dict:
    #     assert len(Theta.shape) == 2 and Theta.shape[1] == self.D
    #     forward_map_evals = self.forward_map.compute_all(
    #         Theta,
    #         compute_lin=True,
    #         compute_log=False,
    #         compute_derivatives=compute_derivatives,
    #         compute_derivatives_2nd_order=compute_derivatives_2nd_order,
    #     )
    #     return forward_map_evals
            
class ObservationsGivenAuxiliary(Hierarchical):
    r"""Class implementing the conditional distribution :math:`Y|U` for the full conditional distribution :math:`\pi(U|\Theta, Y)`."""

    def __init__(
        self,
        D: int,
        L: int,
        N: int,
        y: xp.ndarray,
        sigma_a: Union[float, xp.ndarray],
        sigma_m: Union[float, xp.ndarray],
        omega: xp.ndarray,
    ) -> None:
        super().__init__(D, L, N)

        if not (y.shape == (N, L)):
            raise ValueError(
                "y must have the shape (N, L) = ({}, {}) elements".format(
                    self.N, self.L
                )
            )
        elif isinstance(y, xp.ndarray):
            self.y = y
        else:
            raise ValueError("y must be an array")
        
        if isinstance(sigma_a, (float, int)):
            self.sigma_a = sigma_a * xp.ones((N, L))
        else:
            assert sigma_a.shape == (N, L)
            self.sigma_a = sigma_a

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = sigma_m * xp.ones((N, L))
        else:
            assert sigma_m.shape == (N, L)
            self.sigma_m = sigma_m

        if isinstance(omega, (float, int)):
            self.omega = omega * xp.ones((N, L))
        else:
            assert omega.shape == (N, L)
            self.omega = omega
    

    def neglog_pdf_u(
        self,
        y: xp.ndarray,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += xp.square(y-nlpdf_utils['aux'])
        out /= 2*nlpdf_utils['s_a2'] 
        
        return out
    
    def neglog_pdf_c(
        self,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = nlpdf_utils['w'] - nlpdf_utils['aux']
        z /= nlpdf_utils['s_a']

        out -= log_ndtr(z)

        return out

    def neglog_pdf(
        self,
        nlpdf_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            nlpdf_utils["censored_mask"],
            nlpdf_utils["nlpdf_c"],
            nlpdf_utils["nlpdf_u"],
        )
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out

    def grad_neglog_pdf_u(
        self,
        y: xp.ndarray,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out -= y-nlpdf_utils['aux']
        out /= nlpdf_utils['s_a2'] 
        
        return out
    
    def grad_neglog_pdf_c(
        self,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = nlpdf_utils['w'] - nlpdf_utils['aux']
        z /= nlpdf_utils['s_a']

        out += utils.norm_pdf_cdf_ratio(z)
        out /= nlpdf_utils['s_a']

        return out

    def grad_neglog_pdf(
        self,
        nlpdf_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            nlpdf_utils["censored_mask"],
            nlpdf_utils["grad_nlpdf_c"],
            nlpdf_utils["grad_nlpdf_u"],
        )
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def hess_diag_diag_neglog_pdf_u(
        self,
        y: xp.ndarray,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.ones((self.N, self.L))

        out /= nlpdf_utils['s_a2'] 
        
        return out
    
    def hess_diag_diag_neglog_pdf_c(
        self,
        nlpdf_utils: dict,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = nlpdf_utils['w'] - nlpdf_utils['aux']
        z /= nlpdf_utils['s_a']

        out += xp.square(utils.norm_pdf_cdf_ratio(z))
        out += utils.norm_pdf_cdf_ratio(z)*z
        out /= nlpdf_utils['s_a2']

        return out

    def hess_diag_diag_neglog_pdf(
        self,
        nlpdf_utils: dict,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            nlpdf_utils["censored_mask"],
            nlpdf_utils["hess_diag_diag_nlpdf_c"],
            nlpdf_utils["hess_diag_diag_nlpdf_u"],
        )
        
        if full:
            assert out.shape == (self.N, self.L)
            return out
        elif pixelwise:
            out = xp.sum(out, axis=1)
            assert out.shape == (self.N,)
            return out
        else:
            out = xp.sum(out)
        
        return out
    
    def evaluate_all_nll_utils(
        self,
        forward_map_evals: dict,
        idx: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
    ) -> dict:
        nll_utils = {}
        return nll_utils

    def sample_observation_model(
        self,
        nlpdf_utils: dict,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        
        out = nlpdf_utils['aux'] + self.sigma_a*rng.normal(loc=0, scale=1, size=nlpdf_utils['aux'].shape), # scale is the std not variance, better to put everything outside to avoid mistakes
    
        return out
    