import time

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional


@dataclass
class OptResult:
    x: np.ndarray
    f: float
    grad_norm: float
    iterations: int
    f_evals: int
    g_evals: int
    cpu_time: float
    converged: bool
    method: str
    history: list = field(default_factory=list)


def _cauchy_point(g: np.ndarray, B: np.ndarray, delta: float) -> np.ndarray:
    Bg = B @ g
    gBg = g @ Bg
    if gBg <= 0:
        tau = 1.0
    else:
        tau = min(np.linalg.norm(g)**3 / (gBg * delta), 1.0)
    return -tau * delta / np.linalg.norm(g) * g


def _update_radius(delta: float, rho: float, p_norm: float,
                   eta1: float = 0.1, eta2: float = 0.75,
                   gamma1: float = 0.25, gamma2: float = 2.0,
                   delta_max: float = 1e4) -> float:
    if rho < eta1:
        return gamma1 * delta
    elif rho >= eta2 and abs(p_norm - delta) < 1e-10:
        return min(gamma2 * delta, delta_max)
    return delta


def trust_region_cauchy(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray,
    tol: float = 1e-6, max_iter: int = 1000,
    delta0: float = 1.0, delta_max: float = 1e4
) -> OptResult:
    x = x0.copy().astype(float)
    delta = delta0
    f_evals = g_evals = 0
    history = []
    t0 = time.time()

    for k in range(max_iter):
        fk = f(x); f_evals += 1
        gk = grad(x); g_evals += 1
        Bk = hess(x)
        gnorm = np.linalg.norm(gk)
        history.append(gnorm)

        if gnorm < tol:
            return OptResult(x, fk, gnorm, k, f_evals, g_evals,
                             time.time()-t0, True, "Cauchy Point", history)

        p = _cauchy_point(gk, Bk, delta)
        pred = -(gk @ p + 0.5 * p @ Bk @ p)
        f_new = f(x + p); f_evals += 1
        ared = fk - f_new
        rho = ared / pred if abs(pred) > 1e-15 else 0.0

        if rho > 0.1:
            x = x + p
        delta = _update_radius(delta, rho, np.linalg.norm(p), delta_max=delta_max)

    fk = f(x); gk = grad(x)
    return OptResult(x, fk, np.linalg.norm(gk), max_iter,
                     f_evals, g_evals, time.time()-t0, False, "Cauchy Point", history)


def trust_region_dogleg(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray,
    tol: float = 1e-6, max_iter: int = 1000,
    delta0: float = 1.0, delta_max: float = 1e4
) -> OptResult:
    x = x0.copy().astype(float)
    delta = delta0
    f_evals = g_evals = 0
    history = []
    t0 = time.time()

    for k in range(max_iter):
        fk = f(x); f_evals += 1
        gk = grad(x); g_evals += 1
        Bk = hess(x)
        gnorm = np.linalg.norm(gk)
        history.append(gnorm)

        if gnorm < tol:
            return OptResult(x, fk, gnorm, k, f_evals, g_evals,
                             time.time()-t0, True, "Dogleg", history)

        try:
            L = np.linalg.cholesky(Bk + 1e-12 * np.eye(len(gk)))
            pN = -np.linalg.solve(L.T, np.linalg.solve(L, gk))
        except np.linalg.LinAlgError:
            pN = _cauchy_point(gk, Bk, delta)

        pN_norm = np.linalg.norm(pN)

        if pN_norm <= delta:
            p = pN
        else:
            Bg = Bk @ gk
            gBg = gk @ Bg
            pU = -(gnorm**2 / gBg) * gk if gBg > 0 else -delta / gnorm * gk
            pU_norm = np.linalg.norm(pU)

            if pU_norm >= delta:
                p = -(delta / gnorm) * gk
            else:
                d = pN - pU
                a_ = d @ d
                b_ = 2 * pU @ d
                c_ = pU @ pU - delta**2
                disc = b_**2 - 4*a_*c_
                tau = (-b_ + np.sqrt(max(disc, 0))) / (2*a_) if a_ > 1e-15 else 1.0
                tau = np.clip(tau, 0, 1)
                p = pU + tau * d

        pred = -(gk @ p + 0.5 * p @ Bk @ p)
        f_new = f(x + p); f_evals += 1
        ared = fk - f_new
        rho = ared / pred if abs(pred) > 1e-15 else 0.0

        if rho > 0.1:
            x = x + p
        delta = _update_radius(delta, rho, np.linalg.norm(p), delta_max=delta_max)

    fk = f(x); gk = grad(x)
    return OptResult(x, fk, np.linalg.norm(gk), max_iter,
                     f_evals, g_evals, time.time()-t0, False, "Dogleg", history)


def trust_region_steihaug(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray,
    tol: float = 1e-6, max_iter: int = 1000,
    delta0: float = 1.0, delta_max: float = 1e4,
    cg_tol_factor: float = 0.5
) -> OptResult:
    x = x0.copy().astype(float)
    delta = delta0
    f_evals = g_evals = 0
    history = []
    t0 = time.time()

    for k in range(max_iter):
        fk = f(x); f_evals += 1
        gk = grad(x); g_evals += 1
        Bk = hess(x)
        gnorm = np.linalg.norm(gk)
        history.append(gnorm)

        if gnorm < tol:
            return OptResult(x, fk, gnorm, k, f_evals, g_evals,
                             time.time()-t0, True, "Steihaug-CG", history)

        cg_tol = min(cg_tol_factor, np.sqrt(gnorm)) * gnorm
        p = np.zeros_like(gk)
        r = gk.copy()
        d = -gk.copy()

        for _ in range(len(gk)):
            Bd = Bk @ d
            dBd = d @ Bd
            if dBd <= 0:
                a_ = d @ d
                b_ = 2 * p @ d
                c_ = p @ p - delta**2
                disc = b_**2 - 4*a_*c_
                sigma = (-b_ + np.sqrt(max(disc, 0))) / (2*a_) if a_ > 1e-15 else 0.0
                p = p + sigma * d
                break

            alpha_cg = (r @ r) / dBd
            p_new = p + alpha_cg * d

            if np.linalg.norm(p_new) >= delta:
                a_ = d @ d
                b_ = 2 * p @ d
                c_ = p @ p - delta**2
                disc = b_**2 - 4*a_*c_
                sigma = (-b_ + np.sqrt(max(disc, 0))) / (2*a_) if a_ > 1e-15 else 0.0
                p = p + sigma * d
                break

            p = p_new
            r_new = r + alpha_cg * Bd

            if np.linalg.norm(r_new) < cg_tol:
                break

            beta = (r_new @ r_new) / (r @ r)
            d = -r_new + beta * d
            r = r_new

        pred = -(gk @ p + 0.5 * p @ Bk @ p)
        f_new = f(x + p); f_evals += 1
        ared = fk - f_new
        rho = ared / pred if abs(pred) > 1e-15 else 0.0

        if rho > 0.1:
            x = x + p
        delta = _update_radius(delta, rho, np.linalg.norm(p), delta_max=delta_max)

    fk = f(x); gk = grad(x)
    return OptResult(x, fk, np.linalg.norm(gk), max_iter,
                     f_evals, g_evals, time.time()-t0, False, "Steihaug-CG", history)
