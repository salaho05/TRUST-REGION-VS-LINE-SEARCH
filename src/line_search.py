"""Méthodes de recherche linéaire : BFGS (Armijo / Wolfe fort) et Newton amorti.

Toutes les méthodes renvoient un objet ``OptResult`` (défini dans ``trust_region``),
afin d'exposer la même interface de sortie que les méthodes de région de confiance.
"""
import time

import numpy as np

from trust_region import OptResult


def backtracking_armijo(f, grad_f, xk, dk, alpha0=1.0, rho=0.5, c=1e-4):
    """Recherche du pas par rebroussement d'Armijo.

    Renvoie le pas ``alpha`` et le nombre d'évaluations de ``f`` effectuées.
    """
    alpha = alpha0
    fk = f(xk)
    slope = c * np.dot(grad_f(xk), dk)
    n_eval = 1
    while f(xk + alpha * dk) > fk + alpha * slope:
        alpha *= rho
        n_eval += 1
        if alpha < 1e-14:
            break
    return alpha, n_eval


def _bfgs_update(Hk, sk, yk, n):
    """Mise à jour BFGS de l'inverse de la Hessienne approchée."""
    sy = sk @ yk
    if sy <= 1e-12:
        return Hk
    rho_k = 1.0 / sy
    I = np.eye(n)
    A = I - rho_k * np.outer(sk, yk)
    B = I - rho_k * np.outer(yk, sk)
    return A @ Hk @ B + rho_k * np.outer(sk, sk)


def _make_result(xk, f, grad_norm, n_iter, converged, method, hist):
    return OptResult(
        x=xk,
        f=float(f(xk)),
        grad_norm=grad_norm,
        iterations=n_iter,
        f_evals=0,   # le comptage réel est assuré par les oracles du benchmark
        g_evals=0,
        cpu_time=0.0,
        converged=converged,
        method=method,
        history=hist,
    )


def bfgs(f, grad_f, x0, tol=1e-6, max_iter=500, alpha0=1.0, rho=0.5, c=1e-4):
    """BFGS avec recherche linéaire d'Armijo."""
    t0 = time.time()
    n = len(x0)
    xk = np.array(x0, dtype=float)
    Hk = np.eye(n)
    gk = grad_f(xk)
    hist = []
    converged = False

    for _ in range(max_iter):
        gnorm = np.linalg.norm(gk)
        hist.append(gnorm)
        if gnorm < tol:
            converged = True
            break
        dk = -Hk @ gk
        alpha, _ = backtracking_armijo(f, grad_f, xk, dk, alpha0, rho, c)
        xk_new = xk + alpha * dk
        gk_new = grad_f(xk_new)
        Hk = _bfgs_update(Hk, xk_new - xk, gk_new - gk, n)
        xk, gk = xk_new, gk_new

    res = _make_result(xk, f, np.linalg.norm(gk), len(hist), converged, "BFGS-Armijo", hist)
    res.cpu_time = time.time() - t0
    return res


def strong_wolfe(f, grad_f, xk, dk, alpha_max=10.0, c1=1e-4, c2=0.9):
    """Recherche du pas satisfaisant les conditions de Wolfe fortes."""
    phi = lambda a: f(xk + a * dk)
    dphi = lambda a: np.dot(grad_f(xk + a * dk), dk)
    phi0 = phi(0.0)
    dphi0 = dphi(0.0)
    alpha_prev, alpha_i = 0.0, 1.0
    phi_prev = phi0
    for i in range(20):
        phi_i = phi(alpha_i)
        if phi_i > phi0 + c1 * alpha_i * dphi0 or (phi_i >= phi_prev and i > 0):
            return _zoom(phi, dphi, alpha_prev, alpha_i, phi_prev, phi_i, phi0, dphi0, c1, c2)
        dphi_i = dphi(alpha_i)
        if abs(dphi_i) <= -c2 * dphi0:
            return alpha_i
        if dphi_i >= 0:
            return _zoom(phi, dphi, alpha_i, alpha_prev, phi_i, phi_prev, phi0, dphi0, c1, c2)
        alpha_prev = alpha_i
        phi_prev = phi_i
        alpha_i = min(2.0 * alpha_i, alpha_max)
    return alpha_i


def _zoom(phi, dphi, alo, ahi, phi_lo, phi_hi, phi0, dphi0, c1, c2):
    for _ in range(30):
        alpha_j = _cubic_interp(alo, ahi, phi_lo, phi_hi, dphi(alo), dphi(ahi))
        phi_j = phi(alpha_j)
        if phi_j > phi0 + c1 * alpha_j * dphi0 or phi_j >= phi_lo:
            ahi = alpha_j
            phi_hi = phi_j
        else:
            dphi_j = dphi(alpha_j)
            if abs(dphi_j) <= -c2 * dphi0:
                return alpha_j
            if dphi_j * (ahi - alo) >= 0:
                ahi = alo
                phi_hi = phi_lo
            alo = alpha_j
            phi_lo = phi_j
        if abs(ahi - alo) < 1e-14:
            break
    return alo


def _cubic_interp(a, b, fa, fb, dfa, dfb):
    d1 = dfa + dfb - 3.0 * (fb - fa) / (b - a)
    disc = d1**2 - dfa * dfb
    if disc < 0:
        return (a + b) / 2.0
    d2 = np.sqrt(disc)
    alpha = b - (b - a) * (dfb + d2 - d1) / (dfb - dfa + 2.0 * d2)
    return float(np.clip(alpha, min(a, b), max(a, b)))


def bfgs_wolfe(f, grad_f, x0, tol=1e-6, max_iter=500, c1=1e-4, c2=0.9):
    """BFGS avec recherche linéaire satisfaisant les conditions de Wolfe fortes."""
    t0 = time.time()
    n = len(x0)
    xk = np.array(x0, dtype=float)
    Hk = np.eye(n)
    gk = grad_f(xk)
    hist = []
    converged = False

    for _ in range(max_iter):
        gnorm = np.linalg.norm(gk)
        hist.append(gnorm)
        if gnorm < tol:
            converged = True
            break
        dk = -Hk @ gk
        alpha = strong_wolfe(f, grad_f, xk, dk, c1=c1, c2=c2)
        xk_new = xk + alpha * dk
        gk_new = grad_f(xk_new)
        Hk = _bfgs_update(Hk, xk_new - xk, gk_new - gk, n)
        xk, gk = xk_new, gk_new

    res = _make_result(xk, f, np.linalg.norm(gk), len(hist), converged, "BFGS-Wolfe", hist)
    res.cpu_time = time.time() - t0
    return res


def newton_backtracking(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100,
                        alpha0=1.0, rho=0.5, c=1e-4):
    """Newton amorti par recherche linéaire d'Armijo.

    Si la direction de Newton n'est pas de descente (Hessienne non définie positive),
    on se rabat sur la direction de plus forte pente.
    """
    t0 = time.time()
    xk = np.array(x0, dtype=float)
    hist = []
    converged = False

    for _ in range(max_iter):
        gk = grad_f(xk)
        Hk = hess_f(xk)
        gnorm = np.linalg.norm(gk)
        hist.append(gnorm)
        if gnorm < tol:
            converged = True
            break
        try:
            dk = np.linalg.solve(Hk, -gk)
        except np.linalg.LinAlgError:
            dk = -gk
        if np.dot(gk, dk) >= 0:
            dk = -gk
        alpha, _ = backtracking_armijo(f, grad_f, xk, dk, alpha0, rho, c)
        xk = xk + alpha * dk

    res = _make_result(xk, f, np.linalg.norm(grad_f(xk)), len(hist), converged, "Newton-BT", hist)
    res.cpu_time = time.time() - t0
    return res
