"""開口中心への方向から中心を求め、中心固定の楕円を当てはめる。"""

import numpy as np


def center_from_rays(points, directions, weights=None):
    """各点を通る方向直線との二乗距離を最小化する中心を求める。"""
    points = np.asarray(points, dtype=np.float64)
    direction = np.asarray(directions, dtype=np.float64)
    if len(points) < 2 or direction.shape != points.shape:
        return None
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    if np.any(length < 1e-8):
        return None
    direction = direction/length
    w = np.ones(len(points)) if weights is None else np.maximum(np.asarray(weights), 0)
    # 各直線への垂直射影行列を加えた、2×2の線形系である。
    projection = np.eye(2)[None]-direction[:, :, None]*direction[:, None, :]
    matrix = np.sum(w[:, None, None]*projection, axis=0)
    target = np.sum(w[:, None]*np.einsum('nij,nj->ni', projection, points), axis=0)
    if np.linalg.cond(matrix) > 1e8:
        return None
    try:
        return np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        return None


def fit_centered(points, center, weights=None):
    """既知中心に対する対称2×2行列の3成分を、点の通過拘束から推定する。"""
    if center is None or len(points) < 3:
        return None
    centered = np.asarray(points, dtype=np.float64)-center
    scale = max(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))), 1.)
    x, y = (centered/scale).T
    design = np.stack([x*x, 2*x*y, y*y], 1)
    target = np.ones(len(x))
    if weights is not None:
        w = np.sqrt(np.maximum(np.asarray(weights), 0))
        design, target = design*w[:, None], target*w
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < 3:
            return None
        a, b, c = coefficients
        values, vectors = np.linalg.eigh([[a, b], [b, c]])
        if min(values) <= 0 or not np.isfinite(values).all():
            return None
        direction = vectors[:, 0]
        return (tuple(center), tuple(2*scale/np.sqrt(values)), float(np.rad2deg(np.arctan2(direction[1], direction[0]))))
    except np.linalg.LinAlgError:
        return None


def fit_center_votes(points, directions, weights=None):
    """同じ観測点と中心方向から、中心と楕円を一つの仮説として推定する。"""
    center = center_from_rays(points, directions, weights)
    if center is None:
        return None
    # 直線の交点が、投票した向きの後方にある仮説は除く。
    if np.sum(np.sum((center-points)*directions, axis=1) > 0) < max(2, len(points)/2):
        return None
    return fit_centered(points, center, weights)
