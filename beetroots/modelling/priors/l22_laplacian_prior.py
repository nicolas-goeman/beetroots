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
def compute_laplacian(Var: xp.ndarray, list_edges: xp.ndarray) -> xp.ndarray:
    r"""evaluates the image Laplacian for each of the D maps, :math:`\Delta \Theta_{\cdot d}`

    Parameters
    ----------
    Var : xp.ndarray of shape (N, D)
        D vectors of N pixels
    list_edges : xp.ndarray
        set of edges in the graph induced by the spatial regularization

    Returns
    -------
    xp.ndarray of shape (N, D)
        image Laplacian for each of the D maps
    """
    laplacian_ = xp.zeros_like(Var)
    N = laplacian_.shape[0]

    for i in range(N):
        mask_i_m = list_edges[:, 1] == i
        mask_i_p = list_edges[:, 0] == i

        laplacian_[i] += xp.sum(
            (Var[list_edges[mask_i_p, 1], :] - Var[list_edges[mask_i_p, 0], :]),
            axis=0,
        )
        laplacian_[i] -= xp.sum(
            (Var[list_edges[mask_i_m, 1], :] - Var[list_edges[mask_i_m, 0], :]),
            axis=0,
        )
    return laplacian_  # (N, D)


@decorator_nb()
def compute_laplacian_local(
    Var: xp.ndarray, n: int, list_edges: xp.ndarray, list_pixel_candidates: xp.ndarray
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
    laplacian_ : xp.ndarray of shape (N_candidates, D)
        the laplacian of the candidates
    """
    # D = Var.shape[1]
    N_candidates = list_pixel_candidates.shape[0]
    laplacian_ = xp.zeros_like(list_pixel_candidates)

    mask_i_m = list_edges[:, 1] == n
    mask_i_p = list_edges[:, 0] == n

    for i in range(N_candidates):
        Var[n] = list_pixel_candidates[i] * 1

        laplacian_[i] += xp.sum(
            (Var[list_edges[mask_i_p, 1], :] - Var[list_edges[mask_i_p, 0], :]),
            axis=0,
        )  # (D,)
        laplacian_[i] -= xp.sum(
            (Var[list_edges[mask_i_m, 1], :] - Var[list_edges[mask_i_m, 0], :]),
            axis=0,
        )  # (D,)
    return laplacian_  # (N_candidates, D)


@decorator_nb()
def compute_gradient_from_laplacian(
    laplacian_: xp.ndarray, list_edges: xp.ndarray
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
    g = xp.zeros_like(laplacian_)

    for edge in list_edges:
        val = 2 * (laplacian_[edge[1]] - laplacian_[edge[0]])  # (D,)
        g[edge[0]] += val
        g[edge[1]] -= val

    return g  # (N, D)


@decorator_nb()
def _neglog_pdf_one_pix(
    Var: xp.ndarray,
    idx_pix: xp.ndarray,
    list_pixel_candidates: xp.ndarray,
    list_edges: xp.ndarray,
) -> xp.ndarray:
    """computes the neg log-prior when only one pixel is modified

    Parameters
    ----------
    Var : xp.ndarray of shape (N, D)
        current iterate

    idx_pix : xp.ndarray
        array of the indices of the pixels to consider (:math:`0 \leq n \leq N - 1`)

    list_pixel_candidates : xp.ndarray of shape (N_candidates, D)
        the list of all candidates for pixel n

    Returns
    -------
    xp.ndarray of shape (N_candidates,)
        the leg log-prior of the candidates
    """
    n_pix, k_mtm, D = list_pixel_candidates.shape # FIXME: not consistent with the doc above.
    neglog_p = xp.zeros((n_pix, k_mtm, D))
    i = 0
    n_previous = -500_000

    for n in idx_pix:
        if n != n_previous:
            list_edges_pix = list_edges[
                (list_edges[:, 0] == n) | (list_edges[:, 1] == n)
            ]

        if list_edges_pix.size > 0:
            laplacian_ = compute_laplacian_local(
                Var, n, list_edges_pix, list_pixel_candidates[i]
            )  # (k_mtm, D)
            neglog_p[i, :, :] += laplacian_**2  # (n_pix, k_mtm, D)

        i += 1
        n_previous = n * 1

    return neglog_p


class L22LaplacianSpatialPrior(SpatialPrior):
    r"""L22 smooth spatial prior, valid for both 1D and 2D tensors. Its pdf is defined as

    .. math::

        \forall d \in [1, D], \quad \pi(\Theta_{\cdot d}) \propto \exp \left[- \tau_d \Vert \Delta \Theta_{\cdot d} \Vert_F^2 \right]

    where  :math:`\Vert \cdot \Vert_F` denotes the Fröbenius norm and :math:`\Delta \Theta_{\cdot d}` is the Laplacian of vector :math:ù\Theta_{\cdot d}`.
    """

    def neglog_pdf(
        self,
        Var: xp.ndarray,
        idx_pix: Optional[xp.ndarray] = None,
        with_weights: bool = True,
        pixelwise: bool = False,
    ) -> xp.ndarray:
        assert Var.shape == (self.N, self.D)

        if pixelwise:
            neglog_p = xp.zeros((self.N, self.D))
        else:
            neglog_p = xp.zeros((self.D,))

        if self.list_edges.size > 0:
            laplacian_ = compute_laplacian(Var, self.list_edges)
            if pixelwise:
                neglog_p += laplacian_**2  # (N,D)
            else:
                neglog_p += xp.sum(laplacian_**2, axis=0)  # (D,)

        if with_weights:
            if pixelwise:
                neglog_p *= self.weights[None, :]
            else:
                neglog_p *= self.weights

        # neglog_p /= self.N * self.D
        return neglog_p  # (D,) if not pixelwise or (N, D) if pixelwise

    def neglog_pdf_one_pix(
        self,
        Var: xp.ndarray,
        idx_pix: xp.ndarray,
        list_pixel_candidates: xp.ndarray,
        other_weights: Optional[xp.ndarray] = None,
    ) -> xp.ndarray:
        """
        computes the neg log-prior when only one pixel is modified

        Parameters
        ----------
        Var : xp.ndarray of shape (N, D)
            current iterate

        idx_pix : xp.ndarray
            array of the indices of the pixels to consider (:math:`0 \leq n \leq N - 1`)

        list_pixel_candidates : xp.ndarray of shape (N_candidates, D)
            the list of all candidates for pixel n

        Returns
        -------
        xp.ndarray of shape (N_candidates,)
            the neg log-prior of the candidates
        """
        neglog_p = _neglog_pdf_one_pix(
            Var, idx_pix, list_pixel_candidates, self.list_edges
        )  # # (n_pix, k_mtm, D)
        # neglog_p /= self.N * self.D

        if other_weights is None:
            return xp.sum(
                neglog_p * self.weights[None, None, :], axis=2
            )  # (n_pix, k_mtm)
        else:
            return xp.sum(
                neglog_p * other_weights[None, None, :], axis=2
            )  # (n_pix, k_mtm)

    def gradient_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        assert Var.shape == (self.N, self.D)

        laplacian_ = compute_laplacian(Var, self.list_edges)

        g = compute_gradient_from_laplacian(laplacian_, self.list_edges)  # (N, D)
        # g /= self.N * self.D
        return self.weights[None, :] * g  # (N, D)

    def hess_diag_neglog_pdf(self, Var: xp.ndarray) -> xp.ndarray:
        hess_diag = xp.zeros_like(Var, dtype=xp.float64)

        if self.list_edges.size > 0:
            idx, counts = xp.unique(self.list_edges.flatten(), return_counts=True)
            # print(counts.dtype, hess_diag.dtype)
            hess_diag[idx, :] += (
                2 * (counts * (counts + 1))[:, None] * xp.ones((idx.size, self.D))
            )

        # hess_diag /= self.N * self.D
        return self.weights[None, :] * hess_diag  # (N, D)

    def evaluate_all_nlpdf_utils(
        self, 
        current: dict[str, dict],
        idx_pix: Optional[xp.ndarray],
        compute_derivatives: bool,
        compute_derivatives_2nd_order: bool,
        ) -> None:
        """Evaluate all utilities for the negative log-pdf and its eventual derivatives"""
        raise NotImplementedError