from typing import Optional

import numba as nb
try:
    import cupy as xp
    decorator_nb = nb.cuda.jit
except:
    import numpy as xp
    decorator_nb = nb.njit

from beetroots.modelling.priors.abstract_spatial_prior import SpatialPrior


@decorator_nb()
def compute_laplacian_local(
    Var: xp.ndarray, idx_pix: xp.ndarray, list_edges: xp.ndarray, k_mtm: int
) -> xp.ndarray:
    """computes the local laplacian only for the one modified pixel

    Parameters
    ----------
    Var : xp.ndarray of shape (N, D)
        current iterate

    n : int
        the index of the pixel to consider (:math:`0 \leq n \leq N - 1`)

    list_pixel_candidates : xp.ndarray of shape (N_candidates, D)
        the list of all candidates for pixel n

    Returns
    -------
    laplacian : xp.ndarray of shape (N_candidates, D)
        the laplacian of the candidates
    """
    # D = Var.shape[1]
    laplacian = xp.zeros((idx_pix.size, k_mtm, *Var.shape[2:]))  if k_mtm>0 else xp.zeros((idx_pix.shape[0], *Var.shape[1:])) # (n_pix, k_mtm, D) or (n_pix, D)


    for i, n in enumerate(idx_pix):
        mask_i_m = list_edges[:, 1] == n # No duplicate edges so some edges store the nth pixel as the first element and some as the second element
        mask_i_p = list_edges[:, 0] == n

        if k_mtm == 0:
            laplacian[i] += xp.sum(
                    (Var[list_edges[mask_i_p, 1]] - Var[list_edges[mask_i_p, 0]]),
                    axis=0,
                )  # (k_mtm, D,)
            laplacian[i] -= xp.sum(
                (Var[list_edges[mask_i_m, 1]] - Var[list_edges[mask_i_m, 0]]),
                axis=0,
            )  # (k_mtm, D,)
        else: 
            for j in range(k_mtm):
                laplacian[i, j] += xp.sum(
                    (Var[list_edges[mask_i_p, 1], j] - Var[list_edges[mask_i_p, 0], j]),
                    axis=0,
                )  # (k_mtm, D,)
                laplacian[i, j] -= xp.sum(
                    (Var[list_edges[mask_i_m, 1], j] - Var[list_edges[mask_i_m, 0], j]),
                    axis=0,
                )  # (k_mtm, D,)
    return laplacian  # (n_pix, k_mtm, D) or (n_pix, D)


# @decorator_nb()
def compute_gradient_from_laplacian(
    laplacian: xp.ndarray, list_edges: xp.ndarray, idx_pix: xp.ndarray,
) -> xp.ndarray:
    """evaluates the gradient from the Laplacian matrix

    Parameters
    ----------
    laplacian_ : _type_
        _description_
    list_edges : _type_
        _description_

    Returns
    -------
    xp.ndarray
        _description_
    """
    g = xp.zeros_like(laplacian) # (n_pix, k_mtm, D) or (n_pix, D)

    for edge in list_edges:
        val = 2 * (laplacian[edge[1]] - laplacian[edge[0]])  # (D,)
        g[edge[0]] += val
        g[edge[1]] -= val

    # TODO: """New version to directly use idx_pix instead of having the whole map. To be tested."""
    # for idx in idx_pix:
    #     mask_i_m = list_edges[:, 1] == idx
    #     mask_i_p = list_edges[:, 0] == idx

    #     g[idx] += 2* (laplacian[list_edges[mask_i_p][:,1]] - laplacian[idx][None,...])  # (k_mtm, D,
    #     g[idx] -= 2* (laplacian[idx][None, ...] - laplacian[list_edges[mask_i_m][:,0]])
    
    return g[idx_pix]  # (n_pix, k_mtm, D) or (n_pix, D)


class L22LaplacianSpatialPrior(SpatialPrior):
    r"""L22 smooth spatial prior, valid for both 1D and 2D tensors. Its pdf is defined as

    .. math::

        \forall d \in [1, D], \quad \pi(\Theta_{\cdot d}) \propto \exp \left[- \tau_d \Vert \Delta \Theta_{\cdot d} \Vert_F^2 \right]

    where  :math:`\Vert \cdot \Vert_F` denotes the Fröbenius norm and :math:`\Delta \Theta_{\cdot d}` is the Laplacian of vector :math:ù\Theta_{\cdot d}`.
    """

    def neglog_pdf(
        self,
        with_weights: bool = True,
        pixelwise: bool = False,
        paramwise: bool = False,
        full: bool = False,
    ) -> xp.ndarray:
        k_mtm = self.nlpdf_utils['k_mtm']
        n_pix = self.nlpdf_utils['n_pix']
        
        assert xp.sum([pixelwise, paramwise, full]) <= 1

        neglog_p = xp.zeros((n_pix, k_mtm, self.D)) if k_mtm > 0 else xp.zeros((n_pix, self.D))

        if self.list_edges.size > 0:
            laplacian_ = self.nlpdf_utils['laplacian_local'] * 1
            if self.nlpdf_utils['compute_derivatives']:
                laplacian_ = laplacian_[self.nlpdf_utils['idx_pix']]
            neglog_p += laplacian_**2  # (n_pix,D)

        if with_weights:
            if k_mtm == 0:
                neglog_p *= self.weights[None, :]
            else:
                neglog_p *= self.weights[None, None, :]
        
        if full:
            return neglog_p  # (n_pix, D) or (n_pix, k_mtm, D)
        elif pixelwise:
            neglog_p = neglog_p.sum(axis=tuple(range(1, neglog_p.ndim))) if k_mtm == 0 else neglog_p.sum(axis=tuple(range(2, neglog_p.ndim)))
        elif paramwise:
            neglog_p = neglog_p.sum(axis=0) if k_mtm == 0 else neglog_p.sum(axis=(0, 1))
        else:
            neglog_p = neglog_p.sum() if k_mtm == 0 else neglog_p.swapaxes(0, 1).sum(axis=tuple(range(1, neglog_p.ndim)))

        # neglog_p /= self.N * self.D
        return neglog_p  # (n_pix, k_mtm, D,) or (n_pix, D) or (D,) or (k_mtm, D) depending on the values of pixelwise and k_mtm


    def gradient_neglog_pdf(self, **kwargs) -> xp.ndarray:
        laplacian_ = self.nlpdf_utils['laplacian_local'] * 1

        assert laplacian_.shape[0] == self.N
        assert laplacian_.shape[-1]== self.D

        g_ = compute_gradient_from_laplacian(laplacian_, self.list_edges, self.nlpdf_utils['idx_pix'])  # (n_pix, k_mtm, D) or (n_pix, D)
        # g /= self.N * self.D
        return self.weights[None, :] * g_ if self.nlpdf_utils['k_mtm'] == 0 else  self.weights[None, None, :] * g_ # (n_pix, k_mtm, D) or (n_pix, D)

    def hess_diag_neglog_pdf(self, **kwargs) -> xp.ndarray:
        laplacian_ = self.nlpdf_utils['laplacian_local'] * 1
        hess_diag = xp.zeros_like(laplacian_, dtype=xp.float64)

        if self.list_edges.size > 0:
            idx, counts = xp.unique(self.list_edges.flatten(), return_counts=True)
            # print(counts.dtype, hess_diag.dtype)
            if self.nlpdf_utils['k_mtm'] == 0:
                hess_diag[idx, :] += 2 * (counts * (counts + 1))[:, None] * xp.ones((idx.size, self.D))
            else:
                hess_diag[idx, :, :] += 2 * (counts * (counts + 1))[:, None, None] * xp.ones((idx.size, self.nlpdf_utils['k_mtm'], self.D))

        return self.weights[None, :] * hess_diag if self.nlpdf_utils['k_mtm'] == 0 else self.weights[None, None, :] * hess_diag

    def evaluate_all_nlpdf_utils(
        self, 
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        mtm: bool = False,
        **kwargs
        ) -> None:
        """Evaluate all utilities for the negative log-pdf and its eventual derivatives"""

        Var = current[self.var_name]["var"] # Should be of full size N not n_pix as in spatial prior there are dependencies between pixels from different idx_pix.

        assert Var.shape[0] == self.N
        assert Var.shape[-1]==self.D

        idx_pix = xp.arange(self.N) if idx_pix is None else idx_pix

        self.nlpdf_utils['mtm'] = mtm
        self.nlpdf_utils['k_mtm'] = Var.shape[1] if mtm else 0
        self.nlpdf_utils['n_pix'] = idx_pix.size
        self.nlpdf_utils['idx_pix'] = idx_pix

        if compute_derivatives:
            self.nlpdf_utils['laplacian_local'] = compute_laplacian_local(Var, xp.arange(self.N), self.list_edges, self.nlpdf_utils['k_mtm']) # if we compute the gradient, we need to compute the laplacian for every pixel.
            self.nlpdf_utils['compute_derivatives'] = True
        else:
            self.nlpdf_utils['laplacian_local'] = compute_laplacian_local(Var, idx_pix, self.list_edges, self.nlpdf_utils['k_mtm'])
            self.nlpdf_utils['compute_derivatives'] = False