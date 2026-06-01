"""
LLMRecommender — latent_factor_model.py
=======================================
Latent-factor model for LLM recommendation using ridge regression.

 Learns per-LLM performance and cost representations via:
     E @ Lp = P  (performance matrix)
     E @ Lc = C  (cost matrix)

 where E is the query representation matrix (num_instances x embed_dim or pca_dim).

 Uses ridge regression with 5-fold cross-validation and Tikhonov regularization.
 Handles missing values in P and C via weighted least squares per column.

Efficiency improvements (ridge mode)
----------------------
The parameter search uses a two-phase coarse-to-fine approach:

 Phase 1 — SVD + coarse log-spaced grid
     Compute SVD of E once: E = U Σ Vᵀ.  For any latent dimension d,
     the PCA design matrix is E_d = U[:,:d] * σ[:d] (free — just slice + scale).
     The ridge solve operates in a d×d system instead of D×D, giving an
     O((d/D)³) speedup per column.

     Evaluate n_coarse log-spaced d candidates × lambda_grid candidates.
     Log spacing is correct because the bias-variance tradeoff scales with log(d).

 Phase 2 — Ternary search on log scale
     Within the best region from phase 1, ternary search finds the integer
     minimum in O(log(hi/lo)) steps (~14 evaluations for a 100-wide window).

 Total evaluations ≈ n_folds × (n_coarse × n_lambda + ~14 × 1)
 versus brute-force ≈ n_folds × (d_max - d_min + 1) × n_lambda
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

EPS = 1e-9




def _log_cost(C: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(C, 0.0) + EPS)


def _inv_log_cost(C_log: np.ndarray) -> np.ndarray:
    return np.exp(C_log) - EPS


@dataclass
class TargetScaler:
    acc_mean: float
    acc_std: float
    cost_mean: float
    cost_std: float
    skip_acc_norm: bool = False

    def fit(self, P: np.ndarray, C: np.ndarray) -> "TargetScaler":
        C_log = _log_cost(C)
        self.acc_mean = np.nanmean(P)
        self.acc_std = max(np.nanstd(P), 1e-8)
        self.cost_mean = np.nanmean(C_log)
        self.cost_std = max(np.nanstd(C_log), 1e-8)
        return self

    def transform(self, P, C):
        C_log = _log_cost(C)
        if self.skip_acc_norm:
            P_out = P.copy()
        else:
            P_out = (P - self.acc_mean) / self.acc_std
        return P_out, (C_log - self.cost_mean) / self.cost_std

    def inverse_transform_acc(self, P_hat):
        if self.skip_acc_norm:
            return P_hat
        return P_hat * self.acc_std + self.acc_mean

    def inverse_transform_cost(self, C_hat_log_norm):
        C_hat_log = C_hat_log_norm * self.cost_std + self.cost_mean
        return _inv_log_cost(C_hat_log)


@dataclass
class CVResult:
    """Full diagnostics returned by ``LatentFactorModelWithCV.fit_cv``.

    Attributes
    ----------
    best_latent_dim : int
        Latent dimension selected by the two-phase search.
    best_alpha : float
        Ridge regularization strength selected.
    latent_dim_grid : list[int]
        Log-spaced candidate latent dimensions evaluated in phase 1.
    alpha_grid : list[float]
        Candidate alpha values evaluated.
    coarse_cv_perf : ndarray (n_d_coarse, n_alpha)
        Mean CV RMSE on performance scores for every (d, α) pair in phase 1.
    coarse_cv_cost : ndarray (n_d_coarse, n_alpha)
        Mean CV RMSE on cost scores for every (d, α) pair in phase 1.
    coarse_cv_combined : ndarray (n_d_coarse, n_alpha)
        (coarse_cv_perf + coarse_cv_cost) / 2 — the selection criterion.
    fine_d_values : list[int]
        d values evaluated in the ternary-search phase 2.
    fine_cv_combined : list[float]
        Mean combined CV RMSE at each fine d value (at best_alpha).
    test_rmse_perf : float
        RMSE on the held-out test entries for performance scores.
    test_rmse_cost : float
        RMSE on the held-out test entries for cost scores.
    test_rmse_combined : float
        (test_rmse_perf + test_rmse_cost) / 2.
    n_train_samples : int
        Number of observed (user, item) entries used for CV.
    n_test_samples : int
        Number of observed (user, item) entries used for final evaluation.
    n_folds : int
        Number of folds (always 5 in this implementation).
    total_fits : int
        Total number of (d, α, fold) evaluations performed.
    """
    best_latent_dim:     int
    best_alpha:          float
    latent_dim_grid:     List[int]
    alpha_grid:          List[float]
    coarse_cv_perf:      np.ndarray
    coarse_cv_cost:      np.ndarray
    coarse_cv_combined:  np.ndarray
    fine_d_values:       List[int]
    fine_cv_combined:    List[float]
    test_rmse_perf:      float
    test_rmse_cost:      float
    test_rmse_combined:  float
    n_train_samples:     int
    n_test_samples:      int
    n_folds:             int
    total_fits:          int

    def summary(self) -> str:
        lines = [
            "── Latent-dim CV summary ────────────────────────────────────",
            f"  Samples   train={self.n_train_samples}  "
            f"test={self.n_test_samples}  folds={self.n_folds}",
            f"  Best d    {self.best_latent_dim}",
            f"  Best α    {self.best_alpha:.2e}",
            f"  Total fits evaluated: {self.total_fits}",
            "",
            "  Phase 1 — coarse grid (d × α), mean combined CV RMSE:",
            f"  {'d':>6}  " + "  ".join(f"{a:>8.1e}" for a in self.alpha_grid),
            f"  {'-'*6}  " + "  ".join(["-"*8] * len(self.alpha_grid)),
        ]
        best_flat = float(self.coarse_cv_combined.min())
        n_cols = self.coarse_cv_combined.shape[1]
        for i, d in enumerate(self.latent_dim_grid):
            row = "  ".join(
                f"{'*' if (self.coarse_cv_combined[i, j] == best_flat) else ' '}"
                f"{self.coarse_cv_combined[i, j]:7.4f}"
                for j in range(n_cols)
            )
            lines.append(f"  {d:>6}  {row}")
        lines += [
            "",
            "  Phase 2 — ternary search (at best α):",
            f"  {'d':>6}  {'CV combined':>12}",
            f"  {'-'*6}  {'-'*12}",
        ]
        for d, cv in zip(self.fine_d_values, self.fine_cv_combined):
            marker = " *" if d == self.best_latent_dim else "  "
            lines.append(f"{marker}{d:>6}  {cv:12.4f}")
        lines += [
            "",
            f"  Test RMSE  perf={self.test_rmse_perf:.4f}  "
            f"cost={self.test_rmse_cost:.4f}  "
            f"combined={self.test_rmse_combined:.4f}",
            "─────────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


def _ridge_solve(E: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ridge regression: L = (EᵀE + αI)⁻¹ EᵀY for all columns of Y.

    Works for Y with multiple right-hand-side columns simultaneously.
    Returns L of shape (D, ncols).
    """
    D = E.shape[1]
    regularization = alpha * np.eye(D)
    try:
        return linalg.solve(E.T @ E + regularization, E.T @ Y, assume_a="sym")
    except linalg.LinAlgError:
        return linalg.lstsq(E.T @ E + regularization, E.T @ Y)[0]


def _masked_ridge_solve(
    E: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Column-wise ridge solve that handles NaN entries in Y.

    For each column l of Y, uses only the rows where mask[:, l] is True.
    Returns L of shape (D, ncols).
    """
    D = E.shape[1]
    L = np.zeros((D, Y.shape[1]))
    regularization = alpha * np.eye(D)

    for l in range(Y.shape[1]):
        col_mask = mask[:, l]
        if not col_mask.any():
            L[:, l] = 0.0
            continue

        E_obs = E[col_mask]
        y_obs = Y[col_mask, l]

        E_T_E = E_obs.T @ E_obs
        E_T_y = E_obs.T @ y_obs
        try:
            L[:, l] = linalg.solve(E_T_E + regularization, E_T_y, assume_a="sym")
        except linalg.LinAlgError:
            L[:, l] = linalg.lstsq(E_T_E + regularization, E_T_y)[0]
    return L



def _get_observed_entries(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return row/col indices of all entries observed in *either* P or C.

    A sample (q, l) is considered observed when at least one of P[q,l] or
    C[q,l] is not NaN. Returns an integer array of shape (N, 2).
    """
    obs_p = ~np.isnan(P)
    obs_c = ~np.isnan(C)
    obs = obs_p | obs_c
    rows, cols = np.where(obs)
    return np.column_stack([rows, cols])


def _mask_matrix(M: np.ndarray, entry_indices: np.ndarray) -> np.ndarray:
    """Return a copy of M with all entries NOT in entry_indices set to NaN."""
    M_masked = np.full_like(M, np.nan)
    rows, cols = entry_indices[:, 0], entry_indices[:, 1]
    M_masked[rows, cols] = M[rows, cols]
    return M_masked


def _rmse_on_entries(
    pred: np.ndarray,
    truth: np.ndarray,
    entry_indices: np.ndarray,
) -> float:
    """Compute RMSE between pred and truth at the given (row, col) entries."""
    rows, cols = entry_indices[:, 0], entry_indices[:, 1]
    t = truth[rows, cols]
    p = pred[rows, cols]
    valid = ~np.isnan(t)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))


def _svd_precompute(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full economy SVD of E.

    Returns U (Q, D), sigma (D,), Vt (D, D).
    Uses economy SVD so U is (Q, min(Q,D)), sigma has min(Q,D) values.
    """
    U, sigma, Vt = np.linalg.svd(E, full_matrices=False)
    return U, sigma, Vt


def _svd_design(U: np.ndarray, sigma: np.ndarray, d: int) -> np.ndarray:
    """Return the d-dimensional PCA design matrix E_d = U[:,:d] * sigma[:d].

    Shape: (Q, d). This is O(1) — just a view + elementwise scale.
    """
    return U[:, :d] * sigma[:d]


def _svd_masked_ridge(
    U: np.ndarray,
    sigma: np.ndarray,
    d: int,
    Y: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Ridge solve in the d-dimensional PCA subspace, handling NaN entries.

    The design matrix is implicitly A = U[:,:d] * sigma[:d] (Q, d).
    For each column l of Y, solves in the d-dim subspace using only
    observed (non-NaN) rows.

    Returns X of shape (d, Y.shape[1]).

    Speed advantage: the system is d×d instead of D×D, and for large D
    with d << D this is O((d/D)³) cheaper per column.
    """
    A = _svd_design(U, sigma, d)
    D_d = d
    X = np.zeros((D_d, Y.shape[1]))
    regularization = alpha * np.eye(D_d)

    for l in range(Y.shape[1]):
        col_mask = mask[:, l]
        if not col_mask.any():
            X[:, l] = 0.0
            continue
        A_obs = A[col_mask]
        y_obs = Y[col_mask, l]
        try:
            X[:, l] = linalg.solve(A_obs.T @ A_obs + regularization, A_obs.T @ y_obs, assume_a="sym")
        except linalg.LinAlgError:
            X[:, l] = linalg.lstsq(A_obs.T @ A_obs + regularization, A_obs.T @ y_obs)[0]
    return X


def _ternary_search_log(
    f: "callable",
    lo: int,
    hi: int,
    tol: int = 3,
    max_iter: int = 25,
    history: Optional[List] = None,
) -> int:
    """Find integer argmin of f on [lo, hi] via ternary search on log scale.

    Assumes f is approximately unimodal (holds empirically for CV loss vs d).
    Operates on log(d) so each step halves the log-scale interval regardless
    of the absolute range — equally efficient for d ∈ [8, 16] or [8, 900].
    """
    log_lo = np.log(max(lo, 1))
    log_hi = np.log(hi)
    cache: Dict[int, float] = {}

    def _eval(d: int) -> float:
        d = int(np.clip(d, lo, hi))
        if d not in cache:
            cache[d] = f(d)
            if history is not None:
                history.append((d, cache[d]))
        return cache[d]

    for _ in range(max_iter):
        span = np.exp(log_hi) - np.exp(log_lo)
        if span <= tol:
            break
        m1 = int(round(np.exp(log_lo + (log_hi - log_lo) / 3)))
        m2 = int(round(np.exp(log_hi - (log_hi - log_lo) / 3)))
        m1 = max(lo, min(m1, hi))
        m2 = max(lo, min(m2, hi))
        if m1 == m2:
            m2 = min(m1 + 1, hi)
        if _eval(m1) <= _eval(m2):
            log_hi = np.log(max(m2, 1))
        else:
            log_lo = np.log(max(m1, 1))

    mid = int(round(np.exp((log_lo + log_hi) / 2)))
    mid = int(np.clip(mid, lo, hi))
    _eval(mid)
    return min(cache, key=cache.__getitem__)



def _augment_embedding(
    E: np.ndarray,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise to embedding matrix for data augmentation.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        Input embedding matrix.
    noise_std : float
        Standard deviation of Gaussian noise (relative to embedding norm).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Augmented embedding matrix (same shape as input).
    """
    if noise_std <= 0:
        return E

    rng = np.random.default_rng(seed)
    emb_norm = np.linalg.norm(E, axis=1, keepdims=True)
    emb_norm = np.maximum(emb_norm, 1e-8)
    noise = rng.normal(0, noise_std * emb_norm, size=E.shape).astype(E.dtype)
    return E + noise


class LatentFactorModel:
    """Latent-factor model using ridge regression.

    Solves:
        E @ Lp = P  (performance)
        E @ Lc = C  (cost)

    with ridge regularization and 5-fold cross-validation.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent representation for each LLM.
    alpha : float, default=1.0
        Ridge regularization strength.
    n_folds : int, default=5
        Number of folds for cross-validation.
    normalize_acc : bool, default=True
        Whether to normalize accuracy targets.
    embed_augment : bool, default=False
        Whether to augment embeddings with noise.
    embed_noise_std : float, default=0.01
        Standard deviation of noise for embedding augmentation.
    embed_augment_seed : int, optional
        Random seed for embedding augmentation.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        alpha: float = 1.0,
        alpha_per_target: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        normalize_acc: bool = True,
        embed_augment: bool = False,
        embed_noise_std: float = 0.01,
        embed_augment_seed: Optional[int] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.alpha_per_target = alpha_per_target or {}
        self.n_folds = n_folds
        self.normalize_acc = normalize_acc
        self.embed_augment = embed_augment
        self.embed_noise_std = embed_noise_std
        self.embed_augment_seed = embed_augment_seed

        self.Lp: Optional[np.ndarray] = None
        self.Lc: Optional[np.ndarray] = None
        self._scaler: Optional["TargetScaler"] = None
        self._is_fitted = False
        self._perf_weight = 0.7
        self._cost_weight = 0.3

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        E: np.ndarray,
        P: np.ndarray,
        C: np.ndarray,
        perf_weight: float = 0.7,
        cost_weight: float = 0.3,
    ) -> "LatentFactorModel":
        """Fit the latent factor model with 5-fold cross-validation.

        Parameters
        ----------
        E : np.ndarray, shape (n_instances, embed_dim or pca_dim)
            Query representation matrix.
        P : np.ndarray, shape (n_instances, n_llms)
            Performance matrix with values in [0, 1] or scores.
        C : np.ndarray, shape (n_instances, n_llms)
            Cost matrix with non-negative values.
        perf_weight : float
            Weight for performance in balance score (default 0.7).
        cost_weight : float
            Weight for cost in balance score (default 0.3).

        Returns
        -------
        self
        """
        E = np.asarray(E, dtype=np.float32)
        P = np.asarray(P, dtype=np.float32)
        C = np.asarray(C, dtype=np.float32)

        if E.ndim != 2:
            raise ValueError(f"E must be a 2D array, got shape {E.shape}")
        if P.ndim != 2:
            raise ValueError(f"P must be a 2D array, got shape {P.shape}")
        if C.ndim != 2:
            raise ValueError(f"C must be a 2D array, got shape {C.shape}")

        n_instances, embed_dim = E.shape
        _, n_llms = P.shape

        if E.shape[0] != P.shape[0]:
            raise ValueError(
                f"E and P must have same number of rows (instances): "
                f"E has {E.shape[0]}, P has {P.shape[0]}"
            )
        if E.shape[0] != C.shape[0]:
            raise ValueError(
                f"E and C must have same number of rows (instances): "
                f"E has {E.shape[0]}, C has {C.shape[0]}"
            )
        if embed_dim < 1:
            raise ValueError(f"Embedding dimension must be >= 1, got {embed_dim}")

        self._perf_weight = perf_weight
        self._cost_weight = cost_weight

        P_mask = ~np.isnan(P)
        C_mask = ~np.isnan(C)

        if self.embed_augment and self.n_folds <= 1:
            E = _augment_embedding(E, noise_std=self.embed_noise_std, seed=self.embed_augment_seed)

        C_log = _log_cost(C)
        skip_norm = not self.normalize_acc
        self._scaler = TargetScaler(
            acc_mean=0.0, acc_std=1.0,
            cost_mean=0.0, cost_std=1.0,
            skip_acc_norm=skip_norm,
        )
        self._scaler.fit(P, C)
        self._fit_ridge(E, P, C_log, P_mask, C_mask)

        self._is_fitted = True

        logger.info(
            f"[LatentFactorModel] Fitted: latent_dim={self.latent_dim}, "
            f"E shape={E.shape}, P shape={P.shape}"
        )
        return self

    def _fit_ridge(
        self,
        E: np.ndarray,
        P: np.ndarray,
        C: np.ndarray,
        P_mask: np.ndarray,
        C_mask: np.ndarray,
    ) -> None:
        """Fit using ridge regression with optional CV.

        Augments E with a bias column (intercept) so the model learns
        per-LLM baseline performance and cost offsets.
        Supports separate alpha regularization for P and C matrices.
        """
        n = E.shape[0]
        E_aug = np.hstack([E, np.ones((n, 1), dtype=E.dtype)])

        alpha_P = self.alpha_per_target.get("P", self.alpha)
        alpha_C = self.alpha_per_target.get("C", self.alpha)

        E_work = np.asarray(E_aug, dtype=np.float32)
        P_work = np.asarray(P, dtype=np.float32)
        C_work = np.asarray(C, dtype=np.float32)

        if self.n_folds > 1:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            cv_errors_P = []
            cv_errors_C = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(E_work)):
                E_train, E_val = E_work[train_idx], E_work[val_idx]
                P_train, P_val = P_work[train_idx], P_work[val_idx]
                C_train, C_val = C_work[train_idx], C_work[val_idx]
                P_mask_train = P_mask[train_idx]
                C_mask_train = C_mask[train_idx]

                Lp_fold = self._solve_ridge_col_wise(E_train, P_train, P_mask_train, alpha_P)
                Lc_fold = self._solve_ridge_col_wise(E_train, C_train, C_mask_train, alpha_C)

                P_pred_val = E_val @ Lp_fold
                C_pred_val = E_val @ Lc_fold

                P_val_mask = ~np.isnan(P_val)
                C_val_mask = ~np.isnan(C_val)

                mse_P = float("nan")
                mse_C = float("nan")

                if P_val_mask.sum() > 0:
                    mse_P = float(np.nanmean((P_pred_val[P_val_mask] - P_val[P_val_mask]) ** 2))
                    cv_errors_P.append(mse_P)

                if C_val_mask.sum() > 0:
                    mse_C = float(np.nanmean((C_pred_val[C_val_mask] - C_val[C_val_mask]) ** 2))
                    cv_errors_C.append(mse_C)

                logger.debug(
                    f"  Fold {fold_idx + 1}/{self.n_folds}: "
                    f"MSE_P={mse_P:.6f}, "
                    f"MSE_C={mse_C:.6f}"
                )

            logger.info(
                f"[LatentFactorModel] {self.n_folds}-fold CV MSE - "
                f"Performance: {np.mean(cv_errors_P):.6f}, "
                f"Cost: {np.mean(cv_errors_C):.6f}"
            )
        else:
            logger.info(
                f"[LatentFactorModel] n_folds={self.n_folds}, skipping CV"
            )

        self.Lp = self._solve_ridge_col_wise(E_work, P_work, P_mask, alpha_P)
        self.Lc = self._solve_ridge_col_wise(E_work, C_work, C_mask, alpha_C)

    def _solve_ridge_col_wise(
        self,
        E: np.ndarray,
        Y: np.ndarray,
        mask: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Solve ridge regression for each column of Y separately.

        Uses weighted least squares where weights are derived from mask.
        Handles missing values by excluding them from the fit.

        When no missing data exists (mask is all True), uses full ridge
        regression: L = (E^T E + αI)^-1 @ E^T @ Y for efficiency.

        Parameters
        ----------
        E : np.ndarray, shape (n_samples, embed_dim)
        Y : np.ndarray, shape (n_samples, n_llms)
        mask : np.ndarray, shape (n_samples, n_llms)
            True where values are observed, False where missing.
        alpha : float, optional
            Regularization strength. Uses self.alpha if not provided.

        Returns
        -------
        L : np.ndarray, shape (embed_dim, n_llms)
        """
        if alpha is None:
            alpha = self.alpha

        n_samples, embed_dim = E.shape
        _, n_llms = Y.shape

        if mask.all():
            regularization = alpha * np.eye(embed_dim)
            E_T_E = E.T @ E + regularization
            E_T_Y = E.T @ Y
            try:
                L = linalg.solve(E_T_E, E_T_Y, assume_a="sym")
            except linalg.LinAlgError:
                L = linalg.lstsq(E_T_E, E_T_Y)[0]
            return L

        L = np.zeros((embed_dim, n_llms))
        ridge_term = alpha * np.eye(embed_dim)

        for j in range(n_llms):
            col_mask = mask[:, j]
            if not col_mask.any():
                L[:, j] = 0.0
                continue

            E_obs = E[col_mask]
            y_obs = Y[col_mask, j]

            E_T_E = E_obs.T @ E_obs
            E_T_y = E_obs.T @ y_obs

            try:
                L[:, j] = linalg.solve(E_T_E + ridge_term, E_T_y, assume_a="sym")
            except linalg.LinAlgError:
                L[:, j] = linalg.lstsq(E_T_E + ridge_term, E_T_y)[0]

        return L

    def predict(
        self,
        E: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict performance and cost for new query embeddings.

        Parameters
        ----------
        E : np.ndarray, shape (n_queries, embed_dim)

        Returns
        -------
        tuple of (P_pred, C_pred)
            P_pred : np.ndarray, shape (n_queries, n_llms)
            C_pred : np.ndarray, shape (n_queries, n_llms)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        E = np.asarray(E, dtype=np.float32)

        n = E.shape[0]
        E_aug = np.hstack([E, np.ones((n, 1), dtype=E.dtype)])
        P_pred = E_aug @ self.Lp
        C_pred_log = E_aug @ self.Lc
        C_pred = _inv_log_cost(C_pred_log)

        P_pred = np.clip(P_pred, 0.0, 1.0)
        C_pred = np.clip(C_pred, 0.0, None)

        return P_pred, C_pred

    def recommend(
        self,
        E: np.ndarray,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Recommend the best LLM for each query based on balance score.

        Balance score = perf_weight * normalized_perf - cost_weight * normalized_cost

        Parameters
        ----------
        E : np.ndarray, shape (n_queries, embed_dim)
            Query embeddings.
        exclude_ids : list of int, optional
            LLM IDs to exclude from recommendation.

        Returns
        -------
        list of (llm_id, balance_score) tuples, one per query
        """
        P_pred, C_pred = self.predict(E)

        n_queries, n_llms = P_pred.shape
        recommendations = []

        if self._scaler is not None and not getattr(self._scaler, 'skip_acc_norm', False):
            p_mean = self._scaler.acc_mean
            p_std = self._scaler.acc_std
            c_mean = self._scaler.cost_mean
            c_std = self._scaler.cost_std
            p_norm_all = np.clip((P_pred - p_mean) / (p_std + 1e-8), 0, 1)
            c_norm_all = np.clip((_log_cost(C_pred) - c_mean) / (c_std + 1e-8), 0, 1)
        else:
            P_max = P_pred.max()
            C_log_max = _log_cost(C_pred).max()
            p_norm_all = P_pred / (P_max + 1e-12)
            c_norm_all = _log_cost(C_pred) / (C_log_max + 1e-12)

        for i in range(n_queries):
            balance_scores = self._perf_weight * p_norm_all[i] - self._cost_weight * c_norm_all[i]

            if exclude_ids:
                for eid in exclude_ids:
                    balance_scores[eid] = -np.inf

            best_llm = int(np.argmax(balance_scores))
            best_score = float(balance_scores[best_llm])
            recommendations.append((best_llm, best_score))
        return recommendations

    def get_latent_representations(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the learned latent representations.

        Returns
        -------
        tuple of (Lp, Lc)
            Lp : np.ndarray
                Per-LLM performance latent representation.
            Lc : np.ndarray
                Per-LLM cost latent representation.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.Lp, self.Lc

    def reconstruct(self, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct P and C from E and learned latent factors.

        Returns
        -------
        tuple of (P_reconstructed, C_reconstructed)
        """
        return self.predict(E)


class LatentFactorModelWithCV:
    """Wrapper for LatentFactorModel that performs efficient cross-validation
    to select optimal hyperparameters using a two-phase coarse-to-fine search.

    Phase 1 — SVD + coarse log-spaced grid
        Compute SVD of E once. For any latent dimension d, the PCA design
        matrix is E_d = U[:,:d] * σ[:d] (free — just slice + scale).
        Evaluate n_coarse log-spaced d candidates × alpha_grid candidates.

    Phase 2 — Ternary search on log scale
        Within the best region from phase 1, ternary search finds the
        integer minimum in O(log(hi/lo)) steps.

    This reduces evaluations from O((d_max - d_min) × n_alpha × n_folds)
    to O(n_coarse × n_alpha × n_folds + log(d_max/d_min) × n_folds).
    """

    def __init__(
        self,
        latent_dim_range: Optional[List[int]] = None,
        alpha_range: Optional[List[float]] = None,
        alpha_P_range: Optional[List[float]] = None,
        alpha_C_range: Optional[List[float]] = None,
        n_folds: int = 5,
        d_min: int = 8,
        d_max: int = 64,
        n_coarse: int = 8,
        ternary_tol: int = 3,
        normalize_acc: bool = True,
        embed_augment: bool = False,
        embed_noise_std: float = 0.01,
    ) -> None:
        if latent_dim_range is None:
            latent_dim_range = [16, 32, 64]
        if alpha_range is None:
            alpha_range = [0.01, 0.1, 1.0, 10.0]

        self.latent_dim_range = latent_dim_range
        self.alpha_range = alpha_range
        self.alpha_P_range = alpha_P_range or alpha_range
        self.alpha_C_range = alpha_C_range or alpha_range
        self.n_folds = n_folds
        self.d_min = d_min
        self.d_max = d_max
        self.n_coarse = n_coarse
        self.ternary_tol = ternary_tol
        self.normalize_acc = normalize_acc
        self.embed_augment = embed_augment
        self.embed_noise_std = embed_noise_std

        self.best_model: Optional[LatentFactorModel] = None
        self.cv_results: List[Dict[str, Any]] = []
        self.cv_result: Optional[CVResult] = None

    def fit(
        self,
        E: np.ndarray,
        P: np.ndarray,
        C: np.ndarray,
        perf_weight: float = 0.7,
        cost_weight: float = 0.3,
    ) -> "LatentFactorModelWithCV":
        """Fit with efficient two-phase CV to find best hyperparameters.

        Parameters
        ----------
        E : np.ndarray
        P : np.ndarray
        C : np.ndarray
        perf_weight : float
        cost_weight : float

        Returns
        -------
        self
        """
        return self._fit_ridge(E, P, C, perf_weight, cost_weight)

    def _fit_ridge(
        self,
        E: np.ndarray,
        P: np.ndarray,
        C: np.ndarray,
        perf_weight: float,
        cost_weight: float,
    ) -> "LatentFactorModelWithCV":
        """Fit using ridge regression with two-phase CV."""
        E = np.asarray(E, dtype=float)
        P = np.asarray(P, dtype=float)
        C = np.asarray(C, dtype=float)

        # Log-transform cost for ridge regression (consistent with LatentFactorModel.fit)
        C_log = _log_cost(C)

        # Augment E with bias column for per-LLM intercepts
        Q, D_orig = E.shape
        E_aug = np.hstack([E, np.ones((Q, 1), dtype=E.dtype)])

        U, sigma, Vt = _svd_precompute(E_aug)

        L = P.shape[1]
        d_max = min(self.d_max, D_orig, Q)
        d_min = max(self.d_min, 1)

        if len(self.latent_dim_range) == 1:
            coarse_d_vals = self.latent_dim_range
        else:
            coarse_d_vals = self.latent_dim_range

        n_lam = len(self.alpha_range)
        n_folds = self.n_folds

        rng = np.random.default_rng(42)

        all_entries = _get_observed_entries(P, C_log)
        N = len(all_entries)
        shuffled = rng.permutation(N)
        n_test = max(1, int(round(N * 0.2)))
        n_train = N - n_test

        test_entries = all_entries[shuffled[-n_test:]]
        train_entries = all_entries[shuffled[:n_train]]

        fold_size = int(np.ceil(n_train / n_folds))
        fold_idx: List[np.ndarray] = [
            np.arange(f * fold_size, min((f + 1) * fold_size, n_train))
            for f in range(n_folds)
        ]

        fit_count = [0]

        def _cv_score(d: int, alpha_P: float, alpha_C: float) -> Tuple[float, float]:
            """5-fold CV RMSE for a given (d, alpha_P, alpha_C) triple."""
            d = int(np.clip(d, d_min, d_max))
            rmse_p_folds = np.zeros(n_folds)
            rmse_c_folds = np.zeros(n_folds)

            for f in range(n_folds):
                val_pos = fold_idx[f]
                train_pos = np.concatenate(
                    [fold_idx[k] for k in range(n_folds) if k != f]
                )
                val_entries_f = train_entries[val_pos]
                train_entries_f = train_entries[train_pos]

                P_tr = _mask_matrix(P, train_entries_f)
                C_tr = _mask_matrix(C_log, train_entries_f)
                P_mask_tr = ~np.isnan(P_tr)
                C_mask_tr = ~np.isnan(C_tr)

                Xp_f = _svd_masked_ridge(U, sigma, d, P_tr, P_mask_tr, alpha_P)
                Xc_f = _svd_masked_ridge(U, sigma, d, C_tr, C_mask_tr, alpha_C)

                A_d = _svd_design(U, sigma, d)
                P_hat = A_d @ Xp_f
                C_hat_log = A_d @ Xc_f

                rmse_p_folds[f] = _rmse_on_entries(P_hat, P, val_entries_f)
                rmse_c_folds[f] = _rmse_on_entries(C_hat_log, C_log, val_entries_f)
                fit_count[0] += 1

            return float(np.mean(rmse_p_folds)), float(np.mean(rmse_c_folds))

        # Coarse 2D grid: (d, alpha) using a shared alpha for both P and C.
        # This gives a proper (n_d, n_alpha) result; the fine search at lines
        # below then independently refines alpha_P and alpha_C separately.
        n_lam = len(self.alpha_range)
        coarse_cv_perf = np.full((len(coarse_d_vals), n_lam), np.nan)
        coarse_cv_cost = np.full((len(coarse_d_vals), n_lam), np.nan)

        for i, d in enumerate(coarse_d_vals):
            for j, alpha in enumerate(self.alpha_range):
                rmse_p, rmse_c = _cv_score(d, alpha, alpha)
                coarse_cv_perf[i, j] = rmse_p
                coarse_cv_cost[i, j] = rmse_c

        coarse_cv_combined = (coarse_cv_perf + coarse_cv_cost) / 2.0

        if np.all(np.isnan(coarse_cv_combined)):
            logger.warning(
                "[LatentFactorModelWithCV] All CV scores are NaN. "
                "This may indicate that there are too few observed entries or all entries have NaN. "
                "Falling back to center of coarse grid."
            )
            best_di = len(coarse_d_vals) // 2
            best_ji = len(self.alpha_range) // 2
        else:
            best_flat = int(np.nanargmin(coarse_cv_combined))
            best_di = best_flat // n_lam
            best_ji = best_flat % n_lam
        best_alpha_P_coarse = self.alpha_range[best_ji]
        best_alpha_C_coarse = self.alpha_range[best_ji]

        lo_idx = max(0, best_di - 1)
        hi_idx = min(len(coarse_d_vals) - 1, best_di + 1)
        ternary_lo = coarse_d_vals[lo_idx]
        ternary_hi = coarse_d_vals[hi_idx]

        fine_history: List[Tuple[int, float]] = []
        
        def _fine_score(d: int) -> float:
            rmse_p, rmse_c = _cv_score(d, best_alpha_P_coarse, best_alpha_C_coarse)
            return (rmse_p + rmse_c) / 2.0
        
        best_d = _ternary_search_log(
            f=_fine_score,
            lo=ternary_lo,
            hi=ternary_hi,
            tol=self.ternary_tol,
            history=fine_history,
        )
        # Clip to valid range: ternary search explores [ternary_lo, ternary_hi] but
        # _cv_score already clips to [d_min, d_max] internally.  When d_max is
        # small, multiple candidate d values evaluate identically, and the search
        # may return an out-of-range d.  Clip here so the final fit is consistent.
        best_d = int(np.clip(best_d, d_min, d_max))

        fine_cv_per_P = [_cv_score(best_d, alpha_P, best_alpha_C_coarse) for alpha_P in self.alpha_P_range]
        fine_cv_per_C = [_cv_score(best_d, best_alpha_P_coarse, alpha_C) for alpha_C in self.alpha_C_range]
        best_alpha_P = self.alpha_P_range[int(np.argmin([p for p, c in fine_cv_per_P]))]
        best_alpha_C = self.alpha_C_range[int(np.argmin([c for p, c in fine_cv_per_C]))]

        P_tr = _mask_matrix(P, train_entries)
        C_tr_log = _mask_matrix(C_log, train_entries)
        P_mask_tr = ~np.isnan(P_tr)
        C_mask_tr = ~np.isnan(C_tr_log)

        # Solve in the best_d-dimensional PCA subspace (consistent with CV evaluation),
        # then back-project to the full embed_dim+1 space so that existing predict()
        # (which computes E_aug @ Lp) remains correct.
        # Back-projection: E_aug @ (Vt[:d].T @ X_d) == A_best @ X_d  (exact identity).
        Xp_final = _svd_masked_ridge(U, sigma, best_d, P_tr, P_mask_tr, best_alpha_P)
        Xc_final = _svd_masked_ridge(U, sigma, best_d, C_tr_log, C_mask_tr, best_alpha_C)
        Lp_final = Vt[:best_d, :].T @ Xp_final
        Lc_final = Vt[:best_d, :].T @ Xc_final

        self.best_model = LatentFactorModel(
            latent_dim=best_d,
            alpha=best_alpha_P,
            alpha_per_target={"P": best_alpha_P, "C": best_alpha_C},
            n_folds=1,
            normalize_acc=self.normalize_acc,
            embed_augment=self.embed_augment,
            embed_noise_std=self.embed_noise_std,
        )
        self.best_model.Lp = Lp_final
        self.best_model.Lc = Lc_final
        self.best_model._is_fitted = True
        self.best_model._perf_weight = perf_weight
        self.best_model._cost_weight = cost_weight

        P_hat_final = E_aug @ Lp_final
        C_hat_final_log = E_aug @ Lc_final
        test_p = _rmse_on_entries(P_hat_final, P, test_entries)
        test_c = _rmse_on_entries(C_hat_final_log, C_log, test_entries)

        fine_d_vals = [h[0] for h in fine_history]
        fine_cv_vals = [h[1] for h in fine_history]

        self.cv_result = CVResult(
            best_latent_dim=best_d,
            best_alpha=best_alpha_P,
            latent_dim_grid=coarse_d_vals,
            alpha_grid=self.alpha_range,
            coarse_cv_perf=coarse_cv_perf,
            coarse_cv_cost=coarse_cv_cost,
            coarse_cv_combined=coarse_cv_combined,
            fine_d_values=fine_d_vals,
            fine_cv_combined=fine_cv_vals,
            test_rmse_perf=test_p,
            test_rmse_cost=test_c,
            test_rmse_combined=(test_p + test_c) / 2,
            n_train_samples=n_train,
            n_test_samples=n_test,
            n_folds=n_folds,
            total_fits=fit_count[0],
        )

        for i, d in enumerate(coarse_d_vals):
            for j, alpha_P in enumerate(self.alpha_P_range):
                for k, alpha_C in enumerate(self.alpha_C_range):
                    if alpha_P == alpha_C and alpha_P in self.alpha_range:
                        idx = self.alpha_range.index(alpha_P)
                        cv_perf = float(coarse_cv_perf[i, idx])
                        cv_cost = float(coarse_cv_cost[i, idx])
                    else:
                        cv_perf = float("nan")
                        cv_cost = float("nan")
                    self.cv_results.append({
                        "latent_dim": d,
                        "alpha_P": alpha_P,
                        "alpha_C": alpha_C,
                        "cv_score_perf": cv_perf,
                        "cv_score_cost": cv_cost,
                    })

        logger.info(
            f"[LatentFactorModelWithCV] Best: d={best_d}, alpha_P={best_alpha_P:.2e}, alpha_C={best_alpha_C:.2e}, "
            f"fits={fit_count[0]}"
        )
        return self

    def predict(self, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict performance and cost for new query embeddings.

        Delegates to the best model's predict method.

        Parameters
        ----------
        E : np.ndarray, shape (n_queries, embed_dim)

        Returns
        -------
        tuple of (P_pred, C_pred)
            P_pred : np.ndarray, shape (n_queries, n_llms)
            C_pred : np.ndarray, shape (n_queries, n_llms)
        """
        if self.best_model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.best_model.predict(E)

    def recommend(
        self,
        E: np.ndarray,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Recommend the best LLM for each query based on balance score.

        Delegates to the best model's recommend method.

        Parameters
        ----------
        E : np.ndarray, shape (n_queries, embed_dim)
            Query embeddings.
        exclude_ids : list of int, optional
            LLM IDs to exclude from recommendation.

        Returns
        -------
        list of (llm_id, balance_score) tuples, one per query
        """
        if self.best_model is None:
            raise RuntimeError("Model must be fitted before recommendation.")
        return self.best_model.recommend(E, exclude_ids=exclude_ids)

    def get_latent_representations(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the learned latent representations from the best model.

        Returns
        -------
        tuple of (Lp, Lc)
            Lp : np.ndarray
                Per-LLM performance latent representation.
            Lc : np.ndarray
                Per-LLM cost latent representation.
        """
        if self.best_model is None:
            raise RuntimeError("Model must be fitted first.")
        return self.best_model.get_latent_representations()

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best hyperparameters found during CV.

        Returns
        -------
        dict with 'latent_dim' and 'alpha'
        """
        if self.best_model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        params = {
            "latent_dim": self.best_model.latent_dim,
            "alpha": self.best_model.alpha,
        }
        return params


class ResidualBoostedLatentFactorModel:
    """Two-stage model: base linear latent factor model + per-LLM gradient boosting on residuals.

    Stage 1 — Linear latent factor model (ridge regression):
        E @ Lp = P  (performance)
        E @ Lc = C  (cost)

    Stage 2 — Histogram gradient boosting on residuals:
        For each LLM j, train a regressor on (E, R_j) where R_j = P_j - P_pred_j
        This captures non-linear patterns that the linear model misses.

    At inference, the final prediction is:
        P_final = P_linear + boost_correction

    Parameters
    ----------
    base_model : LatentFactorModel or LatentFactorModelWithCV
        The base linear model to use for initial prediction.
    boost_rounds_per_llm : int, default=100
        Number of boosting iterations (n_estimators) for each LLM's residual model.
    boost_max_depth : int, default=3
        Maximum depth of each boosting tree.
    boost_lr : float, default=0.1
        Learning rate for the gradient boosting.
    boost_min_samples_leaf : int, default=20
        Minimum samples required in a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_model: Optional[Union["LatentFactorModel", "LatentFactorModelWithCV"]] = None,
        boost_rounds_per_llm: int = 100,
        boost_max_depth: int = 3,
        boost_lr: float = 0.1,
        boost_min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        self.base_model = base_model
        self.boost_rounds_per_llm = boost_rounds_per_llm
        self.boost_max_depth = boost_max_depth
        self.boost_lr = boost_lr
        self.boost_min_samples_leaf = boost_min_samples_leaf
        self.random_state = random_state

        self._boost_models_P: Optional[List[HistGradientBoostingRegressor]] = None
        self._boost_models_C: Optional[List[HistGradientBoostingRegressor]] = None
        self._is_fitted = False
        self._n_llms = 0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        E: np.ndarray,
        P: np.ndarray,
        C: np.ndarray,
        perf_weight: float = 0.7,
        cost_weight: float = 0.3,
    ) -> "ResidualBoostedLatentFactorModel":
        """Fit the two-stage residual boosting model.

        Parameters
        ----------
        E : np.ndarray, shape (n_instances, embed_dim or pca_dim)
            Query representation matrix.
        P : np.ndarray, shape (n_instances, n_llms)
            Performance matrix with values in [0, 1] or scores.
        C : np.ndarray, shape (n_instances, n_llms)
            Cost matrix with non-negative values.
        perf_weight : float
            Weight for performance in balance score (default 0.7).
        cost_weight : float
            Weight for cost in balance score (default 0.3).

        Returns
        -------
        self
        """
        E = np.asarray(E, dtype=np.float32)
        P = np.asarray(P, dtype=np.float32)
        C = np.asarray(C, dtype=np.float32)

        n_instances, embed_dim = E.shape
        self._n_llms = P.shape[1]

        if self.base_model is None:
            self.base_model = LatentFactorModel(
                latent_dim=min(embed_dim, 64),
                alpha=1.0,
                n_folds=3,
            )

        logger.info(
            f"[ResidualBoostedLatentFactorModel] Stage 1: Fitting base linear model..."
        )
        self.base_model.fit(E, P, C, perf_weight=perf_weight, cost_weight=cost_weight)

        logger.info(
            f"[ResidualBoostedLatentFactorModel] Stage 2: Training boosting on residuals..."
        )
        P_pred, C_pred = self.base_model.predict(E)

        P_mask = ~np.isnan(P)
        C_mask = ~np.isnan(C)

        self._boost_models_P = []
        self._boost_models_C = []

        for j in range(self._n_llms):
            col_mask_P = P_mask[:, j]
            col_mask_C = C_mask[:, j]

            residual_P = np.zeros(n_instances)
            residual_C = np.zeros(n_instances)

            if col_mask_P.sum() > 0:
                residual_P[col_mask_P] = (
                    P[col_mask_P, j] - P_pred[col_mask_P, j]
                )
            if col_mask_C.sum() > 0:
                residual_C[col_mask_C] = (
                    C[col_mask_C, j] - C_pred[col_mask_C, j]
                )

            boost_P = HistGradientBoostingRegressor(
                max_iter=self.boost_rounds_per_llm,
                max_depth=self.boost_max_depth,
                learning_rate=self.boost_lr,
                min_samples_leaf=self.boost_min_samples_leaf,
                random_state=self.random_state,
                early_stopping=False,
            )
            if col_mask_P.sum() > self.boost_min_samples_leaf:
                boost_P.fit(E[col_mask_P], residual_P[col_mask_P])
                logger.debug(
                    f"  LLM {j}: Boost-P trained on {col_mask_P.sum()} samples, "
                    f"residual std={np.std(residual_P[col_mask_P]):.4f}"
                )
            else:
                boost_P.fit(E, residual_P)
            self._boost_models_P.append(boost_P)

            boost_C = HistGradientBoostingRegressor(
                max_iter=self.boost_rounds_per_llm,
                max_depth=self.boost_max_depth,
                learning_rate=self.boost_lr,
                min_samples_leaf=self.boost_min_samples_leaf,
                random_state=self.random_state,
                early_stopping=False,
            )
            if col_mask_C.sum() > self.boost_min_samples_leaf:
                boost_C.fit(E[col_mask_C], residual_C[col_mask_C])
            else:
                boost_C.fit(E, residual_C)
            self._boost_models_C.append(boost_C)

        self._is_fitted = True

        P_pred_boosted, C_pred_boosted = self.predict(E)
        mae_P_before = float(np.nanmean(np.abs(P - P_pred)))
        mae_P_after = float(np.nanmean(np.abs(P - P_pred_boosted)))
        mae_C_before = float(np.nanmean(np.abs(C - C_pred)))
        mae_C_after = float(np.nanmean(np.abs(C - C_pred_boosted)))

        logger.info(
            f"[ResidualBoostedLatentFactorModel] Training complete. "
            f"MAE P: {mae_P_before:.4f} -> {mae_P_after:.4f} "
            f"(improvement: {(mae_P_before - mae_P_after) / mae_P_before * 100:.1f}%)"
        )
        logger.info(
            f"[ResidualBoostedLatentFactorModel] "
            f"MAE C: {mae_C_before:.6f} -> {mae_C_after:.6f} "
            f"(improvement: {(mae_C_before - mae_C_after) / mae_C_before * 100:.1f}%)"
        )

        return self

    def predict(self, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict performance and cost with residual boosting.

        Parameters
        ----------
        E : np.ndarray, shape (n_queries, embed_dim)

        Returns
        -------
        tuple of (P_pred, C_pred)
            P_pred : np.ndarray, shape (n_queries, n_llms)
            C_pred : np.ndarray, shape (n_queries, n_llms)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        E = np.asarray(E, dtype=np.float32)

        P_linear, C_linear = self.base_model.predict(E)

        n_queries = E.shape[0]
        P_boosted = np.zeros((n_queries, self._n_llms))
        C_boosted = np.zeros((n_queries, self._n_llms))

        for j in range(self._n_llms):
            P_boosted[:, j] = self._boost_models_P[j].predict(E)
            C_boosted[:, j] = self._boost_models_C[j].predict(E)

        P_final = P_linear + P_boosted
        C_final = C_linear + C_boosted

        P_final = np.clip(P_final, 0.0, 1.0)
        C_final = np.clip(C_final, 0.0, None)

        return P_final, C_final

    def recommend(
        self,
        E: np.ndarray,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Recommend the best LLM for each query based on balance score.

        Delegates to the base model's recommend method.

        Parameters
        ----------
        E : np.ndarray, shape (n_queries, embed_dim)
            Query embeddings.
        exclude_ids : list of int, optional
            LLM IDs to exclude from recommendation.

        Returns
        -------
        list of (llm_id, balance_score) tuples, one per query
        """
        P_pred, C_pred = self.predict(E)

        n_queries, n_llms = P_pred.shape

        P_max = P_pred.max()
        C_max = C_pred.max()
        p_norm = P_pred / (P_max + 1e-12)
        c_norm = C_pred / (C_max + 1e-12)

        perf_weight = getattr(self.base_model, '_perf_weight', 0.7)
        cost_weight = getattr(self.base_model, '_cost_weight', 0.3)

        balance_scores = perf_weight * p_norm - cost_weight * c_norm

        if exclude_ids:
            for eid in exclude_ids:
                balance_scores[:, eid] = -np.inf

        best_llm_per_query = np.argmax(balance_scores, axis=1)
        best_score_per_query = np.max(balance_scores, axis=1)

        recommendations = [
            (int(best_llm_per_query[i]), float(best_score_per_query[i]))
            for i in range(n_queries)
        ]
        return recommendations

    def get_latent_representations(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the learned latent representations from the base model.

        Returns
        -------
        tuple of (Lp, Lc)
            Lp : np.ndarray
                Per-LLM performance latent representation from base model.
            Lc : np.ndarray
                Per-LLM cost latent representation from base model.
        """
        if self.base_model is None:
            return None, None
        return self.base_model.get_latent_representations()
