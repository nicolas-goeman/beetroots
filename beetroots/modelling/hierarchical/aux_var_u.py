from beetroots.modelling.hierarchical.abstract_hierarchical import Hierarchical
from beetroots.modelling.likelihoods.abstract_likelihood import Likelihood
from beetroots.modelling.likelihoods import utils
from typing import Optional, Union

try:
    import cupy as xp
except ImportError:
    import numpy as xp  # Fallback to NumPy if CuPy is not installed
from scipy.special import log_ndtr


class AuxiliaryGivenTarget(Hierarchical): #TODO: check is Likelihood inheritance is relevant
    r"""Class implementing the conditional distribution :math:`U|\Theta` for both full conditional distributions :math:`\pi(U|\Theta, Y)` and :math:`\pi(\Theta|U, Y)`."""

    def __init__(
        self,
        forward_map,
        D: int,
        L: int,
        N: int,
        var_names_dict: dict,
        sigma_m: Union[float, xp.ndarray],
        var_name: str = None,
    ) -> None:
        super().__init__(var_name, vars_involved=list(var_names_dict.values()))

        self.forward_map = forward_map
        self.forward_map_evals = {}
        '''dict: forward map evaluations (log and derivatives)'''

        self.D = D
        self.L = L
        self.N = N

        if isinstance(sigma_m, (float, int)):
            self.sigma_m = sigma_m * xp.ones((self.N, self.L))
        else:
            assert sigma_m.shape == (self.N, self.L)
            self.sigma_m = sigma_m
        self.sigma_m2 = xp.square(self.sigma_m)
        '''xp.ndarray: variance of the multiplicative noise applied to the forward map'''

        assert isinstance(var_names_dict, dict) and list(var_names_dict.keys()) == ["aux", "target"]

        self.var_names_dict = var_names_dict

    def neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out = self.nlpdf_utils['log_aux']
        out += xp.square(self.nlpdf_utils['log_aux'] + self.sigma_m2/2 - self.forward_map_evals['log_f_Var'])/(2*self.sigma_m2)
        
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
    ) -> Union[float, xp.ndarray]:
        """Computes the gradient of the negative log-pdf with respect to the auxiliary variable.

        Returns:
            Union[float, xp.ndarray]: array of shape (N, L)
        """        
        N_pix = self.nlpdf_utils['N_pix']
        out = xp.zeros((N_pix, self.L))

        out += 3/2 + (self.nlpdf_utils['log_aux'] - self.forward_map_evals['log_f_Var'])/self.sigma_m2
        out /= self.nlpdf_utils['aux']

        return out
    
    def gradient_neglog_pdf_wrt_target(
        self,
    ) -> Union[float, xp.ndarray]:
        """Computes the gradient of the negative log-pdf with respect to the target variable.

        Returns:
            Union[float, xp.ndarray]: array of shape (N, D)
        """        
        N_pix = self.nlpdf_utils['N_pix']
        out = xp.zeros((N_pix, self.D, self.L))

        out -= self.forward_map_evals['grad_log_f_Var']*xp.expand_dims(self.nlpdf_utils['log_aux'] + self.sigma_m2/2 - self.forward_map_evals['log_f_Var'], axis=1)
        out /= self.sigma_m2

        return out.sum(axis=-1) # sum over the last axis L, outputs a (N, D) array
            
    def gradient_neglog_pdf(
        self,
        deriv_var_name: str,
        pixelwise: bool = False,
        full: bool = False,
    ) -> Union[float, xp.ndarray]:
        """Computes the gradient of the negative log-pdf with respect to the auxiliary or target variable.

        Args:
            deriv_var_name (str): variable name for which the gradient is computed
            pixelwise (bool, optional): if the gradient of the nlpdf is summed up for eadh pixel. Defaults to False.
            full (bool, optional): keeps the full information (each dimension is kept). Defaults to False.

        Raises:
            ValueError: variable name for differentiation must be either "aux" or "target"

        Returns:
            Union[float, xp.ndarray]: array of shape (n_pix, [k_mtm], L) or (n_pix, [k_mtm], D) if full else if pixelwise array of shape (N,) or (N, [k_mtm]) else float or array of shape (k_mtm,)
        """        
        n_pix = self.nlpdf_utils['n_pix']
        k_mtm = self.nlpdf_utils['k_mtm']
        N_pix = self.nlpdf_utils['N_pix']
         
        if deriv_var_name == self.var_names_dict["aux"]:
            out = self.gradient_neglog_pdf_wrt_aux()
        elif deriv_var_name == self.var_names_dict["target"]:
            out = self.gradient_neglog_pdf_wrt_target()
        else:
            raise ValueError(f"deriv_var_name must be either {self.var_names_dict["aux"]} or {self.var_names_dict["target"]}")

        if full:
            if deriv_var_name == self.var_names_dict["aux"]:
                assert out.shape == (N_pix, self.L)
            elif deriv_var_name == self.var_names_dict["target"]:
                assert out.shape == (N_pix, self.D)
            return out if k_mtm == 0 else out.reshape(n_pix, k_mtm, *out.shape[1:])
        elif pixelwise:
            out = xp.sum(out, axis=tuple(range(1, out.ndim)))
            assert out.shape == (N_pix,)
            return out if k_mtm == 0 else out.reshape(n_pix, k_mtm)
        else:
            if k_mtm > 0:
                out = out.reshape(n_pix, k_mtm, *out.shape[1:])
            return xp.sum(out) if k_mtm == 0 else out.swapaxes(0, 1).sum(axis=tuple(range(1, out.ndim)))
    
    def hess_diag_neglog_pdf_wrt_aux(
        self,
    ) -> Union[float, xp.ndarray]:
        N_pix = self.nlpdf_utils['N_pix']
        out = xp.zeros((N_pix, self.L))

        out += 3/2 + (self.nlpdf_utils['log_aux'] - self.forward_map_evals['log_f_Var']- 1)/self.sigma_m2
        out /= -self.nlpdf_utils['aux']**2

        return out

    def hess_diag_neglog_pdf_wrt_target(
        self,
    ) -> Union[float, xp.ndarray]:
        N_pix = self.nlpdf_utils['N_pix']
        out = xp.zeros((N_pix, self.D, self.L))

        out += self.forward_map_evals['hess_diag_log_f_Var']*xp.expand_dims(self.nlpdf_utils['log_aux'] + self.sigma_m2/2 - self.forward_map_evals['log_f_Var'], axis=1)
        out -= self.forward_map_evals['grad_log_f_Var']**2
        out /= self.sigma_m2

        return out.sum(axis=-1) # sum over the last axis L, outputs a (N, D) array
    
    def hess_diag_neglog_pdf(
        self,
        deriv_var_name: str,
        pixelwise: bool = False,
        full: bool = False,
    ) -> Union[float, xp.ndarray]:
        """Computes the diagonal of the Hessian of the negative log-pdf with respect to the auxiliary or target variable.

        Args:
            deriv_var_name (str): variable name for which the Hessian is computed
            pixelwise (bool, optional): if the diagonal of the hessian of the nlpdf is summed up for eadh pixel. Defaults to False.
            full (bool, optional): keeps the full information (each dimension is kept). Defaults to False.

        Raises:
            ValueError: variable name for differentiation must be either "aux" or "target"

        Returns:
            Union[float, xp.ndarray]: array of shape (n_pix, [k_mtm], L) or (n_pix, [k_mtm], D) if full else if pixelwise array of shape (N,) or (N, [k_mtm]) else float or array of shape (k_mtm,)
        """        
        
        n_pix = self.nlpdf_utils['n_pix']
        k_mtm = self.nlpdf_utils['k_mtm']
        N_pix = self.nlpdf_utils['N_pix']
         
        if deriv_var_name == self.var_names_dict["aux"]:
            out = self.hess_neglog_pdf_wrt_aux()
        elif deriv_var_name == self.var_names_dict["target"]:
            out = self.hess_neglog_pdf_wrt_target()
        else:
            raise ValueError(f"deriv_var_name must be either {self.var_names_dict["aux"]} or {self.var_names_dict["target"]}")

        if full:
            if deriv_var_name == self.var_names_dict["aux"]:
                assert out.shape == (N_pix, self.L)
            elif deriv_var_name == self.var_names_dict["target"]:
                assert out.shape == (N_pix, self.D)
            return out if k_mtm == 0 else out.reshape(n_pix, k_mtm, *out.shape[1:])
        elif pixelwise:
            out = xp.sum(out, axis=tuple(range(1, out.ndim)))
            assert out.shape == (N_pix,)
            return out if k_mtm == 0 else out.reshape(n_pix, k_mtm)
        else:
            if k_mtm > 0:
                out = out.reshape(n_pix, k_mtm, *out.shape[1:])
            return xp.sum(out) if k_mtm == 0 else out.swapaxes(0, 1).sum(axis=tuple(range(1, out.ndim)))
        
    
    def evaluate_all_nlpdf_utils(
        self,
        current: dict[str, dict],
        deriv_var_name: str = None,
        idx_pix: Optional[xp.ndarray] = None,
        compute_derivatives: bool = True,
        compute_derivatives_2nd_order: bool = True,
        mtm: bool = False,
    ) -> dict:
        # TODO: implement this method
        shape_var_aux = current[self.var_names_dict['aux']]['var'].shape
        shape_var_target = current[self.var_names_dict['target']]['var'].shape
        assert isinstance(mtm, bool), f"mtm should be a boolean, got {type(mtm)}"
        
        self.nlpdf_utils = dict()
        self.nlpdf_utils["mtm"] = mtm

        self.nlpdf_utils["aux"] = current[self.var_names_dict['aux']]['var']
        if idx_pix is not None:
            self.nlpdf_utils["aux"] = self.nlpdf_utils["aux"][idx_pix]
        self.nlpdf_utils["log_aux"] = xp.log(self.nlpdf_utils["aux"])


        forward_var_inputs = current[self.var_names_dict['target']]['var']
        if idx_pix is not None:
            forward_var_inputs = forward_var_inputs[idx_pix]
        if mtm:
            forward_var_inputs = forward_var_inputs.reshape(-1, *shape_var_target[2:])
        compute_derivatives_forward = compute_derivatives and deriv_var_name == "target"
        compute_derivatives_2nd_order_forward = compute_derivatives_2nd_order and deriv_var_name == "target"
        self.evaluate_all_forward_map(forward_var_inputs, compute_derivatives=compute_derivatives_forward, compute_derivatives_2nd_order=compute_derivatives_2nd_order_forward) # TODO: put variable and other required arguments here

        n_pix = idx_pix.size if idx_pix is not None else shape_var_aux[0]
        k_mtm = shape_var_aux[1] if mtm else 0
        N_pix = self.forward_map_evals["f_Var"].shape[0]

        if mtm:
            assert n_pix * k_mtm == N_pix
        else:
            assert n_pix == N_pix

        self.nlpdf_utils['n_pix'] = n_pix
        self.nlpdf_utils['N_pix'] = N_pix
        self.nlpdf_utils['k_mtm'] = k_mtm

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        
        out = self.forward_map_evals["log_f_Var"] - xp.square(self.sigma_m)/2 + self.sigma_m*rng.normal(loc=0, scale=1, size=self.forward_map_evals["log_f_Var"].shape), # scale is the std not variance, better to put everything outside to avoid mistakes
        out = xp.exp(out)
    
        return out
    
            
class ObservationsGivenAuxiliary(Likelihood):
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
        super().__init__(forward_map=None, y=y, D=D, L=L, N=N)
        
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
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out += xp.square(y-self.nlpdf_utils['aux'])
        out /= 2*self.nlpdf_utils['s_a2'] 
        
        return out
    
    def neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = self.nlpdf_utils['w'] - self.nlpdf_utils['aux']
        z /= self.nlpdf_utils['s_a']

        out -= log_ndtr(z)

        return out

    def neglog_pdf(
        self,
        current: dict[str, dict],
        pixelwise: bool = False,
        full: bool = False,
        idx_pix: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["nlpdf_c"],
            self.nlpdf_utils["nlpdf_u"],
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
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))

        out -= y-self.nlpdf_utils['aux']
        out /= self.nlpdf_utils['s_a2'] 
        
        return out
    
    def grad_neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = self.nlpdf_utils['w'] - self.nlpdf_utils['aux']
        z /= self.nlpdf_utils['s_a']

        out += utils.norm_pdf_cdf_ratio(z)
        out /= self.nlpdf_utils['s_a']

        return out

    def grad_neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["grad_nlpdf_c"],
            self.nlpdf_utils["grad_nlpdf_u"],
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
    ) -> Union[float, xp.ndarray]:
        out = xp.ones((self.N, self.L))

        out /= self.nlpdf_utils['s_a2'] 
        
        return out
    
    def hess_diag_diag_neglog_pdf_c(
        self,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        
        z = self.nlpdf_utils['w'] - self.nlpdf_utils['aux']
        z /= self.nlpdf_utils['s_a']

        out += xp.square(utils.norm_pdf_cdf_ratio(z))
        out += utils.norm_pdf_cdf_ratio(z)*z
        out /= self.nlpdf_utils['s_a2']

        return out

    def hess_diag_diag_neglog_pdf(
        self,
        pixelwise: bool = False,
        full: bool = False,
        idx: Optional[xp.ndarray] = None,
    ) -> Union[float, xp.ndarray]:
        out = xp.zeros((self.N, self.L))
        out += xp.where(
            self.nlpdf_utils["censored_mask"],
            self.nlpdf_utils["hess_diag_diag_nlpdf_c"],
            self.nlpdf_utils["hess_diag_diag_nlpdf_u"],
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
    
    def evaluate_all_nlpdf_utils(
        self, 
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        ) -> None:
        self.nlpdf_utils = {} #TODO: implement nlpdf_utils method

    def sample_observation_model(
        self,
        rng: xp.random.Generator = xp.random.default_rng(),
    ) -> xp.ndarray:
        
        out = self.nlpdf_utils['aux'] + self.sigma_a*rng.normal(loc=0, scale=1, size=self.nlpdf_utils['aux'].shape), # scale is the std not variance, better to put everything outside to avoid mistakes
    
        return out
    