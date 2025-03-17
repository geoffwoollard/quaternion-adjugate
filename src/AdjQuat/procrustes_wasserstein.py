import torch
import ot
import numpy as np
from typing import Tuple
from AdjQuat import solutions


def procrustes_wasserstein_2d_3d_dram(
    xyz: torch.Tensor,
    UV: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-10,
    verbose_log: bool = False,
    cost_d: int = 2,
) -> Tuple[np.ndarray, torch.Tensor, list]:
    """Solves the Procrustes-Wasserstein problem using DRaM (corrected).

    Iteratively optimizes the rotation and point correspondence between two point sets X and Y, with weights p and q, respectively.

    Parameters:
    -----------
    X : A matrix of size n x 3, where n is the number of points and d is the dimensionality of the points.
    Y : A matrix of size m x 2, where m is the number of points and d is the dimensionality of the points.
    max_iter : Maximum number of iterations.
    tol : Tolerance for convergence.

    Notes:
    ------

    transport_plan does not need to be square. It is a matrix of size n x m, where n is the number of points in X and m is the number of points in Y.

    the ensure_determinant_one is a trick to ensure that the determinant of the rotation matrix is 1. This is important to avoid flipping. See ref [3] for details.

    Implements Algorithm 1 from [1]
    See [2] for details on ensure_determinant_one

    [1] Adamo, D., Corneli, M., Vuillien, M., Vila, E., Adamo, D., Corneli, M., … Vila, E. (2025).
    An in depth look at the Procrustes-Wasserstein distance: properties and barycenters, 0–21.

    [2] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020).
    An analysis of SVD for deep rotation estimation. Advances in Neural Information Processing Systems, 2020-Decem(3), 1–18.

    [3] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020).
    An Analysis of SVD for Deep Rotation Estimation. Proceedings of the 34th International Conference on Neural Information Processing Systems, (3), 1–12.
    http://doi.org/10.5555/3495724.3497615
    """
    n, d2 = UV.shape
    m, d3 = xyz.shape
    assert n == m
    assert d2 == 2
    assert d3 == 3
    p = q = np.ones(n)

    rotation = torch.eye(d3).to(xyz.dtype)

    logs = []
    xyz_permuted = xyz.clone()
    UV0 = torch.cat([UV, torch.zeros(n, 1)], dim=1)
    for idx in range(max_iter):
        xyz_permuted_R = xyz_permuted @ rotation.T
        if cost_d == 3:
            cost = cost_3d = torch.cdist(UV0, xyz_permuted_R[:,:d3], p=2) ** 2
        elif cost_d == 2:
            cost = cost_2d = torch.cdist(UV, xyz_permuted_R[:,:d2], p=2) ** 2
        else:
            raise ValueError("cost_d must be 2 or 3")

        # Solve optimal transport problem using EMD to get point correspondence
        transport_plan, log = ot.emd(p, q, cost.numpy(), log=True)
        xyz_permuted = torch.from_numpy(transport_plan) @ xyz
        if n == m:
            point_norm = torch.norm(UV - xyz_permuted@ rotation.T[:,:d2])
        
        if verbose_log:
            log["transport_plan"] = transport_plan
            log["R"] = rotation
            log["xyz_permuted_R"] = xyz_permuted_R
            log['point_norm'] = point_norm

        else:
            del log["u"]  # free up space
            del log["v"]
        logs.append(log)

        rotation_new = solutions.make_M_opt_rot(xyz_permuted.numpy(),UV.numpy())
        rotation = rotation_new

        if len(logs) > 1:
            if np.linalg.norm(log["cost"] - logs[-2]["cost"]) < tol:
                return transport_plan, rotation, logs

        

    return transport_plan, rotation, logs


def procrustes_wasserstein_2d_3d_svd(
    X: torch.Tensor,
    Y: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-10,
    verbose_log: bool = False,
) -> Tuple[np.ndarray, torch.Tensor, list]:
    """Solves the Procrustes-Wasserstein problem.

    Iteratively optimizes the rotation and point correspondence between two point sets X and Y, with weights p and q, respectively.

    Parameters:
    -----------
    Y : A matrix of size m x 3, where m is the number of points and d is the dimensionality of the points.
    X : A matrix of size n x 2, where n is the number of points and d is the dimensionality of the points.
    p : A vector of size n, representing the weights of the points in X.
    q : A vector of size m, representing the weights of the points in Y.
    max_iter : Maximum number of iterations.
    tol : Tolerance for convergence.

    Notes:
    ------

    transport_plan does not need to be square. It is a matrix of size n x m, where n is the number of points in X and m is the number of points in Y.

    the ensure_determinant_one is a trick to ensure that the determinant of the rotation matrix is 1. This is important to avoid flipping. See ref [3] for details.

    Implements Algorithm 1 from [1]
    See [2] for details on ensure_determinant_one

    [1] Adamo, D., Corneli, M., Vuillien, M., Vila, E., Adamo, D., Corneli, M., … Vila, E. (2025).
    An in depth look at the Procrustes-Wasserstein distance: properties and barycenters, 0–21.

    [2] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020).
    An analysis of SVD for deep rotation estimation. Advances in Neural Information Processing Systems, 2020-Decem(3), 1–18.

    [3] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020).
    An Analysis of SVD for Deep Rotation Estimation. Proceedings of the 34th International Conference on Neural Information Processing Systems, (3), 1–12.
    http://doi.org/10.5555/3495724.3497615
    """
    n, d2 = X.shape
    m, d3 = Y.shape
    assert d2 == 2
    assert d3 == 3

    rotation = torch.eye(d3).to(X.dtype)

    logs = []
    for idx in range(max_iter):
        YR = Y @ rotation

        C = torch.cdist(X, YR[:,:d2], p=2) ** 2

        # Solve optimal transport problem using EMD to get point correspondence
        transport_plan, log = ot.emd(p.numpy(), q.numpy(), C.numpy(), log=True)
        if n == m:
            point_norm = torch.norm(X - torch.from_numpy(transport_plan).to(YR.dtype) @ YR[:,:d2])
        if verbose_log:
            log["transport_plan"] = transport_plan
            log["R"] = rotation
            log["YR"] = YR
            log['point_norm'] = point_norm
        else:
            del log["u"]  # free up space
            del log["v"]
        logs.append(log)

        # Update rotation using SVD
        Rxy_estimate = Y.T @ transport_plan.T @ X
        U, _, Vh = torch.linalg.svd(Rxy_estimate, full_matrices=True)
        assert Vh.shape == (d2,d2)
        Rxy_projected = U[:,:2] @ Vh
        # ensure_determinant_one = torch.ones(d).to(X.dtype)
        # ensure_determinant_one[-1] = torch.det(U @ Vh)  # ensure no flipping. see
        # rotation_new = U @ torch.diag(ensure_determinant_one) @ Vh

        Rz_est = torch.linalg.cross(Rxy_projected[:,0],Rxy_projected[:,1])
        rotation_new = torch.cat([Rxy_projected, Rz_est.reshape(3,1)], dim=1)

        if len(logs) > 1:
            if np.linalg.norm(log["cost"] - logs[-2]["cost"]) < tol:
                return transport_plan, rotation, logs

        rotation = rotation_new

    return transport_plan, rotation, logs