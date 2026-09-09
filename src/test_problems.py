# -*- coding: utf-8 -*-
"""Bibliothèque des problèmes de test pour le benchmark d'optimisation non linéaire.

Regroupe les 180 instances réparties en trois catégories de difficulté :

- Catégorie A : formes quadratiques (dont des cas mal conditionnés)
- Catégorie B : problèmes non linéaires classiques (Rosenbrock, etc.)
- Catégorie C : problèmes non convexes avec points selle

Chaque problème fournit sa fonction objectif, son gradient et sa Hessienne,
ainsi qu'un point de départ, pour des dimensions allant de n = 10 à n = 1000.

Interface publique :
    * get_problem(pid, n)  -> dictionnaire décrivant une instance
    * adjust_n(pid, n)     -> ajuste la dimension aux contraintes du problème
    * PROBLEMS_A_IDS / PROBLEMS_B_IDS / PROBLEMS_C_IDS / PROBLEMS_ALL_IDS

Exécuté directement, ce module affiche une validation par catégorie.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm, eigvalsh

# =============================================================================
# Configuration globale
# =============================================================================
DIMS = [10, 50, 100, 500, 1000]
EXCLUDED_IDS = {"B9_Levy"}
FIXED_DIM = {
    "A06": 4,
    "A08": 2,
    "A11": 2,
    "A13": 3,
    "C02": 2,
    "C04": 2,
}


# =============================================================================
# Utilitaires communs
# =============================================================================
def _check_n_even(n: int, name: str):
    if n % 2 != 0:
        raise ValueError(f"{name} requiert n pair, reçu n={n}")


def _check_n_multiple_of_4(n: int, name: str):
    if n % 4 != 0:
        raise ValueError(f"{name} requiert n multiple de 4, reçu n={n}")


def adjust_n(pid: str, n: int) -> int:
    if pid in FIXED_DIM:
        return FIXED_DIM[pid]
    if pid == "A09":
        return max(8, (n // 4) * 4)
    if pid in ("C07", "C09"):
        return max(4, (n // 2) * 2)
    if pid == "B8_ChainWoo":
        return max(4, (n // 4) * 4)
    return n


def _normalize_problem(pb: dict, category: str, pid: str, n: int) -> dict:
    out = dict(pb)
    x0 = out.get("x0")
    if callable(x0):
        x0 = x0(n)
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64)
    out.setdefault("id", pid)
    out.setdefault("category", category)
    out["x0"] = x0
    if "grad_f" not in out and "grad" in out:
        out["grad_f"] = out["grad"]
    if "hess_f" not in out and "hess" in out:
        out["hess_f"] = out["hess"]
    out["n"] = int(out.get("n", n))
    out.setdefault("f_opt", 0.0)
    out.setdefault("kappa", "?")
    return out


def gradient_check(pb: dict, eps: float = 1e-6, tol: float = 1e-4) -> Tuple[str, float]:
    x = np.asarray(pb["x0"], dtype=float)
    g_ana = np.asarray(pb["grad_f"](x), dtype=float)
    g_num = np.zeros_like(x)
    for i in range(len(x)):
        ei = np.zeros_like(x)
        ei[i] = eps
        g_num[i] = (pb["f"](x + ei) - pb["f"](x - ei)) / (2 * eps)
    err = norm(g_ana - g_num) / (1 + norm(g_ana))
    return ("OK" if err < tol else "ECHEC"), err


def hessian_check(pb: dict, eps: float = 1e-5, tol: float = 1e-3) -> Tuple[str, float]:
    x = np.asarray(pb["x0"], dtype=float)
    H_ana = np.asarray(pb["hess_f"](x), dtype=float)
    n = len(x)
    H_num = np.zeros((n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        H_num[:, i] = (pb["grad_f"](x + ei) - pb["grad_f"](x - ei)) / (2 * eps)
    H_num = 0.5 * (H_num + H_num.T)
    err = norm(H_ana - H_num, ord='fro') / (1 + norm(H_ana, ord='fro'))
    return ("OK" if err < tol else "ECHEC"), err


# =============================================================================
# CATÉGORIE A
# =============================================================================
_A_DENSE_CACHE: Dict[Tuple[int, float, int], np.ndarray] = {}
_A_GEOM_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def _make_diagonal_quadratic_problem(pid: str, kappa: float):
    def f(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        d = np.linspace(1.0, float(kappa), n)
        return float(d @ (x ** 2))

    def grad_f(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        d = np.linspace(1.0, float(kappa), n)
        return 2.0 * d * x

    def hess_f(x):
        n = len(x)
        d = np.linspace(1.0, float(kappa), n)
        return np.diag(2.0 * d)

    return {
        "id": pid,
        "name": f"Quadratique diagonale κ={kappa:g}",
        "f": f,
        "grad_f": grad_f,
        "hess_f": hess_f,
        "x0": lambda n: np.ones(n),
        "f_opt": 0.0,
        "kappa": kappa,
        "dims": "all",
    }


A01 = _make_diagonal_quadratic_problem("A01", 10.0)
A02 = _make_diagonal_quadratic_problem("A02", 1_000.0)
A03 = _make_diagonal_quadratic_problem("A03", 1_000_000.0)


def _make_dense_quadratic(kappa: float, seed: int = 42):
    def _get_A(n: int):
        key = (n, float(kappa), seed)
        if key not in _A_DENSE_CACHE:
            rng = np.random.default_rng(seed)
            M = rng.standard_normal((n, n))
            Q, _ = np.linalg.qr(M)
            d = np.linspace(1.0, float(kappa), n)
            A = Q @ np.diag(d) @ Q.T
            A = 0.5 * (A + A.T)
            _A_DENSE_CACHE[key] = A
        return _A_DENSE_CACHE[key]

    def f(x):
        x = np.asarray(x, dtype=float)
        A = _get_A(len(x))
        return float(x @ A @ x)

    def grad_f(x):
        x = np.asarray(x, dtype=float)
        A = _get_A(len(x))
        return 2.0 * (A @ x)

    def hess_f(x):
        A = _get_A(len(x))
        return 2.0 * A

    return {
        "id": "A04",
        "name": f"Quadratique dense κ={kappa:g}",
        "f": f,
        "grad_f": grad_f,
        "hess_f": hess_f,
        "x0": lambda n: np.ones(n),
        "f_opt": 0.0,
        "kappa": kappa,
        "dims": "all",
    }


A04 = _make_dense_quadratic(1_000.0)


def _rosenbrock_ext_f(x):
    x = np.asarray(x, dtype=float)
    _check_n_even(len(x), "Rosenbrock étendu")
    xi = x[0::2]
    xip = x[1::2]
    return float(np.sum(100.0 * (xip - xi ** 2) ** 2 + (1.0 - xi) ** 2))


def _rosenbrock_ext_grad(x):
    x = np.asarray(x, dtype=float)
    _check_n_even(len(x), "Rosenbrock étendu")
    g = np.zeros_like(x)
    xi = x[0::2]
    xip = x[1::2]
    g[0::2] = -400.0 * xi * (xip - xi ** 2) - 2.0 * (1.0 - xi)
    g[1::2] = 200.0 * (xip - xi ** 2)
    return g


def _rosenbrock_ext_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    _check_n_even(n, "Rosenbrock étendu")
    H = np.zeros((n, n))
    xi = x[0::2]
    xip = x[1::2]
    for i in range(n // 2):
        a, b = 2 * i, 2 * i + 1
        H[a, a] = -400.0 * (xip[i] - xi[i] ** 2) + 800.0 * xi[i] ** 2 + 2.0
        H[a, b] = -400.0 * xi[i]
        H[b, a] = -400.0 * xi[i]
        H[b, b] = 200.0
    return H


def _rosenbrock_ext_x0(n):
    _check_n_even(n, "Rosenbrock étendu")
    x = np.zeros(n)
    x[0::2] = -1.2
    x[1::2] = 1.0
    return x


A05 = {
    "id": "A05",
    "name": "Rosenbrock étendu",
    "f": _rosenbrock_ext_f,
    "grad_f": _rosenbrock_ext_grad,
    "hess_f": _rosenbrock_ext_hess,
    "x0": _rosenbrock_ext_x0,
    "f_opt": 0.0,
    "kappa": "~2500 à x*",
    "dims": "n pair",
}


def _wood_f(x):
    x1, x2, x3, x4 = map(float, x)
    return (
        100.0 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
        + 90.0 * (x4 - x3 ** 2) ** 2 + (1 - x3) ** 2
        + 10.0 * (x2 + x4 - 2) ** 2 + 0.1 * (x2 - x4) ** 2
    )


def _wood_grad(x):
    x1, x2, x3, x4 = map(float, x)
    g1 = -400.0 * x1 * (x2 - x1 ** 2) - 2.0 * (1 - x1)
    g2 = 200.0 * (x2 - x1 ** 2) + 20.0 * (x2 + x4 - 2) + 0.2 * (x2 - x4)
    g3 = -360.0 * x3 * (x4 - x3 ** 2) - 2.0 * (1 - x3)
    g4 = 180.0 * (x4 - x3 ** 2) + 20.0 * (x2 + x4 - 2) - 0.2 * (x2 - x4)
    return np.array([g1, g2, g3, g4], dtype=float)


def _wood_hess(x):
    x1, x2, x3, x4 = map(float, x)
    H = np.zeros((4, 4), dtype=float)
    H[0, 0] = -400.0 * (x2 - 3 * x1 ** 2) + 2.0
    H[0, 1] = H[1, 0] = -400.0 * x1
    H[1, 1] = 200.0 + 20.2
    H[1, 3] = H[3, 1] = 19.8
    H[2, 2] = -360.0 * (x4 - 3 * x3 ** 2) + 2.0
    H[2, 3] = H[3, 2] = -360.0 * x3
    H[3, 3] = 180.0 + 20.2
    return H


A06 = {
    "id": "A06",
    "name": "Wood (n=4)",
    "f": _wood_f,
    "grad_f": _wood_grad,
    "hess_f": _wood_hess,
    "x0": lambda n: np.array([-3.0, -1.0, -3.0, -1.0], dtype=float),
    "f_opt": 0.0,
    "kappa": "~1e4",
    "dims": "n=4",
}


def _make_geom_quadratic(kappa: float):
    def _get_A(n: int):
        key = (n, float(kappa))
        if key not in _A_GEOM_CACHE:
            d = np.geomspace(1.0, float(kappa), n)
            _A_GEOM_CACHE[key] = np.diag(d)
        return _A_GEOM_CACHE[key]

    def f(x):
        x = np.asarray(x, dtype=float)
        A = _get_A(len(x))
        return float(x @ A @ x)

    def grad_f(x):
        x = np.asarray(x, dtype=float)
        A = _get_A(len(x))
        return 2.0 * (A @ x)

    def hess_f(x):
        return 2.0 * _get_A(len(x))

    return {
        "id": "A07",
        "name": f"Quadratique spectre géométrique κ={kappa}",
        "f": f,
        "grad_f": grad_f,
        "hess_f": hess_f,
        "x0": lambda n: np.ones(n),
        "f_opt": 0.0,
        "kappa": kappa,
        "dims": "all",
    }


A07 = _make_geom_quadratic(1e4)


def _beale_f(x):
    x1, x2 = map(float, x)
    r1 = 1.5 - x1 * (1 - x2)
    r2 = 2.25 - x1 * (1 - x2 ** 2)
    r3 = 2.625 - x1 * (1 - x2 ** 3)
    return r1 ** 2 + r2 ** 2 + r3 ** 2


def _beale_grad(x):
    x1, x2 = map(float, x)
    b = np.array([1.5, 2.25, 2.625], dtype=float)
    t = np.array([1 - x2, 1 - x2 ** 2, 1 - x2 ** 3], dtype=float)
    dt = np.array([-1.0, -2.0 * x2, -3.0 * x2 ** 2], dtype=float)
    r = b - x1 * t
    g1 = -2.0 * (r @ t)
    g2 = 2.0 * x1 * (r @ dt)
    return np.array([g1, g2], dtype=float)


def _beale_hess(x):
    x1, x2 = map(float, x)
    b = np.array([1.5, 2.25, 2.625], dtype=float)
    t = np.array([1 - x2, 1 - x2 ** 2, 1 - x2 ** 3], dtype=float)
    dt = np.array([-1.0, -2.0 * x2, -3.0 * x2 ** 2], dtype=float)
    d2t = np.array([0.0, -2.0, -6.0 * x2], dtype=float)
    r = b - x1 * t
    H = np.zeros((2, 2), dtype=float)
    H[0, 0] = 2.0 * (t @ t)
    H[0, 1] = H[1, 0] = 2.0 * np.dot(r, x1 * dt) * 0.0 + 2.0 * x1 * np.dot(t, dt) - 2.0 * np.dot(r, dt)
    H[1, 1] = 2.0 * x1 ** 2 * (dt @ dt) - 2.0 * np.dot(r, x1 * d2t)
    return H


A08 = {
    "id": "A08",
    "name": "Beale (n=2)",
    "f": _beale_f,
    "grad_f": _beale_grad,
    "hess_f": _beale_hess,
    "x0": lambda n: np.array([1.0, 1.0], dtype=float),
    "f_opt": 0.0,
    "kappa": "~1e6",
    "dims": "n=2",
}


def _powell_f(x):
    x = np.asarray(x, dtype=float)
    _check_n_multiple_of_4(len(x), "Powell étendu")
    s = 0.0
    for i in range(len(x) // 4):
        a, b, c, d = x[4 * i:4 * i + 4]
        s += (a + 10 * b) ** 2 + 5 * (c - d) ** 2 + (b - 2 * c) ** 4 + 10 * (a - d) ** 4
    return float(s)


def _powell_grad(x):
    x = np.asarray(x, dtype=float)
    _check_n_multiple_of_4(len(x), "Powell étendu")
    g = np.zeros_like(x)
    for i in range(len(x) // 4):
        a, b, c, d = x[4 * i:4 * i + 4]
        g[4 * i + 0] = 2 * (a + 10 * b) + 40 * (a - d) ** 3
        g[4 * i + 1] = 20 * (a + 10 * b) + 4 * (b - 2 * c) ** 3
        g[4 * i + 2] = 10 * (c - d) - 8 * (b - 2 * c) ** 3
        g[4 * i + 3] = -10 * (c - d) - 40 * (a - d) ** 3
    return g


def _powell_hess(x):
    x = np.asarray(x, dtype=float)
    _check_n_multiple_of_4(len(x), "Powell étendu")
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    for i in range(n // 4):
        a, b, c, d = x[4 * i:4 * i + 4]
        aa, bb, cc, dd = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        H[aa, aa] = 2 + 120 * (a - d) ** 2
        H[aa, bb] = H[bb, aa] = 20.0
        H[aa, dd] = H[dd, aa] = -120 * (a - d) ** 2
        H[bb, bb] = 200 + 12 * (b - 2 * c) ** 2
        H[bb, cc] = H[cc, bb] = -24 * (b - 2 * c) ** 2
        H[cc, cc] = 10 + 48 * (b - 2 * c) ** 2
        H[cc, dd] = H[dd, cc] = -10.0
        H[dd, dd] = 10 + 120 * (a - d) ** 2
    return H


A09 = {
    "id": "A09",
    "name": "Powell étendu (n multiple de 4)",
    "f": _powell_f,
    "grad_f": _powell_grad,
    "hess_f": _powell_hess,
    "x0": lambda n: np.tile(np.array([3.0, -1.0, 0.0, 1.0], dtype=float), n // 4),
    "f_opt": 0.0,
    "kappa": "∞ en x* (H singulier)",
    "dims": "n multiple de 4",
}


def _kappa_n2_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = np.linspace(1.0, float(n ** 2), n)
    return float(d @ (x ** 2))


def _kappa_n2_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = np.linspace(1.0, float(n ** 2), n)
    return 2.0 * d * x


def _kappa_n2_hess(x):
    n = len(x)
    return np.diag(2.0 * np.linspace(1.0, float(n ** 2), n))


A10 = {
    "id": "A10",
    "name": "Quadratique κ=n² (croissant avec dim)",
    "f": _kappa_n2_f,
    "grad_f": _kappa_n2_grad,
    "hess_f": _kappa_n2_hess,
    "x0": lambda n: np.ones(n),
    "f_opt": 0.0,
    "kappa": "n²",
    "dims": "all",
}


def _froth_f(x):
    x1, x2 = map(float, x)
    r1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    r2 = -29 + x1 + ((1 + x2) * x2 - 14) * x2
    return r1 ** 2 + r2 ** 2


def _froth_grad(x):
    x1, x2 = map(float, x)
    r1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    r2 = -29 + x1 + ((1 + x2) * x2 - 14) * x2
    dr1 = 10 * x2 - 3 * x2 ** 2 - 2
    dr2 = 3 * x2 ** 2 + 2 * x2 - 14
    g1 = 2 * r1 + 2 * r2
    g2 = 2 * r1 * dr1 + 2 * r2 * dr2
    return np.array([g1, g2], dtype=float)


def _froth_hess(x):
    x1, x2 = map(float, x)
    r1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    r2 = -29 + x1 + ((1 + x2) * x2 - 14) * x2
    d1 = 10 * x2 - 3 * x2 ** 2 - 2
    d2 = 3 * x2 ** 2 + 2 * x2 - 14
    d1d = 10 - 6 * x2
    d2d = 6 * x2 + 2
    H = np.zeros((2, 2), dtype=float)
    H[0, 0] = 4.0
    H[0, 1] = H[1, 0] = 2 * d1 + 2 * d2
    H[1, 1] = 2 * d1 ** 2 + 2 * r1 * d1d + 2 * d2 ** 2 + 2 * r2 * d2d
    return H


A11 = {
    "id": "A11",
    "name": "Freudenstein-Roth (n=2)",
    "f": _froth_f,
    "grad_f": _froth_grad,
    "hess_f": _froth_hess,
    "x0": lambda n: np.array([0.5, -2.0], dtype=float),
    "f_opt": 0.0,
    "kappa": "~1e5",
    "dims": "n=2",
}


def _make_rosen_alpha(alpha: float):
    def f(x):
        x = np.asarray(x, dtype=float)
        xi = x[:-1]
        xn = x[1:]
        return float(np.sum(alpha * (xn - xi ** 2) ** 2 + (1 - xi) ** 2))

    def grad_f(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        g = np.zeros_like(x)
        for i in range(n - 1):
            g[i] += -4 * alpha * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
            g[i + 1] += 2 * alpha * (x[i + 1] - x[i] ** 2)
        return g

    def hess_f(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        H = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            H[i, i] += -4 * alpha * (x[i + 1] - x[i] ** 2) + 8 * alpha * x[i] ** 2 + 2
            H[i, i + 1] += -4 * alpha * x[i]
            H[i + 1, i] += -4 * alpha * x[i]
            H[i + 1, i + 1] += 2 * alpha
        return H

    return {
        "id": "A12",
        "name": f"Rosenbrock α={int(alpha)} (κ≈{int(4*alpha+2)})",
        "f": f,
        "grad_f": grad_f,
        "hess_f": hess_f,
        "x0": lambda n: np.array([-1.2, 1.0] * (n // 2) + ([-1.2] if n % 2 else []), dtype=float)[:n],
        "f_opt": 0.0,
        "kappa": int(4 * alpha + 2),
        "dims": "all",
    }


A12 = _make_rosen_alpha(500.0)

_BARD_Y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58,
           0.73, 0.96, 1.34, 2.10, 4.39]


def _bard_f(x):
    a, b, c = map(float, x)
    s = 0.0
    for i, y in enumerate(_BARD_Y, 1):
        u = i
        v = 16 - i
        w = min(u, v)
        den = b * v + c * w
        r = y - (a + u / den)
        s += r ** 2
    return float(s)


def _bard_grad(x):
    a, b, c = map(float, x)
    g = np.zeros(3, dtype=float)
    for i, y in enumerate(_BARD_Y, 1):
        u = i
        v = 16 - i
        w = min(u, v)
        den = b * v + c * w
        r = y - (a + u / den)
        g[0] += -2 * r
        g[1] += 2 * r * u * v / den ** 2
        g[2] += 2 * r * u * w / den ** 2
    return g


def _bard_hess(x):
    a, b, c = map(float, x)
    H = np.zeros((3, 3), dtype=float)
    for i, y in enumerate(_BARD_Y, 1):
        u = i
        v = 16 - i
        w = min(u, v)
        den = b * v + c * w
        r = y - (a + u / den)
        J = np.array([-1.0, u * v / den ** 2, u * w / den ** 2], dtype=float)
        H += 2 * np.outer(J, J)
        d2b = -2 * u * v ** 2 / den ** 3
        d2c = -2 * u * w ** 2 / den ** 3
        d2bc = -2 * u * v * w / den ** 3
        H[1, 1] -= 2 * r * d2b
        H[2, 2] -= 2 * r * d2c
        H[1, 2] -= 2 * r * d2bc
        H[2, 1] = H[1, 2]
    return H


A13 = {
    "id": "A13",
    "name": "Bard (n=3, moindres carrés)",
    "f": _bard_f,
    "grad_f": _bard_grad,
    "hess_f": _bard_hess,
    "x0": lambda n: np.array([1.0, 1.0, 1.0], dtype=float),
    "f_opt": 8.21e-3,
    "kappa": "~1e4",
    "dims": "n=3",
}


def _dixon_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = (x[0] - 1) ** 2
    for i in range(1, n):
        s += (i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2
    return float(s)


def _dixon_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n, dtype=float)
    g[0] = 2 * (x[0] - 1)
    for i in range(1, n):
        t = 2 * x[i] ** 2 - x[i - 1]
        g[i] += 8 * (i + 1) * x[i] * t
        g[i - 1] += -2 * (i + 1) * t
    return g


def _dixon_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    H[0, 0] = 2.0
    for i in range(1, n):
        t = 2 * x[i] ** 2 - x[i - 1]
        H[i, i] += 8 * (i + 1) * (t + 4 * x[i] ** 2)
        H[i, i - 1] += -8 * (i + 1) * x[i]
        H[i - 1, i] = H[i, i - 1]
        H[i - 1, i - 1] += 2 * (i + 1)
    return H


A14 = {
    "id": "A14",
    "name": "Dixon-Price",
    "f": _dixon_f,
    "grad_f": _dixon_grad,
    "hess_f": _dixon_hess,
    "x0": lambda n: 2.0 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "croissant avec n",
    "dims": "all",
}


def _tridag_f(x):
    x = np.asarray(x, dtype=float)
    return float(2.0 * np.sum(x ** 2) - 2.0 * np.sum(x[:-1] * x[1:]))


def _tridag_grad(x):
    x = np.asarray(x, dtype=float)
    g = 4.0 * x.copy()
    g[:-1] -= 2.0 * x[1:]
    g[1:] -= 2.0 * x[:-1]
    return g


def _tridag_hess(x):
    n = len(x)
    H = 4.0 * np.eye(n)
    for i in range(n - 1):
        H[i, i + 1] = H[i + 1, i] = -2.0
    return H


A15 = {
    "id": "A15",
    "name": "Quadratique Laplacien 1D (κ≈(2n/π)²)",
    "f": _tridag_f,
    "grad_f": _tridag_grad,
    "hess_f": _tridag_hess,
    "x0": lambda n: np.ones(n),
    "f_opt": 0.0,
    "kappa": "(2n/π)²",
    "dims": "all",
}

PROBLEMS_A: Dict[str, dict] = {
    "A01": A01,
    "A02": A02,
    "A03": A03,
    "A04": A04,
    "A05": A05,
    "A06": A06,
    "A07": A07,
    "A08": A08,
    "A09": A09,
    "A10": A10,
    "A11": A11,
    "A12": A12,
    "A13": A13,
    "A14": A14,
    "A15": A15,
}


# =============================================================================
# CATÉGORIE B (sans B9_Levy)
# =============================================================================
def rosenbrock_f(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosenbrock_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n, dtype=float)
    for i in range(n - 1):
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return g


def rosenbrock_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        H[i, i] += -400.0 * (x[i + 1] - x[i] ** 2) + 800.0 * x[i] ** 2 + 2.0
        H[i, i + 1] += -400.0 * x[i]
        H[i + 1, i] += -400.0 * x[i]
        H[i + 1, i + 1] += 200.0
    return H


B1_Rosenbrock = {
    "id": "B1_Rosenbrock",
    "name": "B1_Rosenbrock",
    "f": rosenbrock_f,
    "grad_f": rosenbrock_grad,
    "hess_f": rosenbrock_hess,
    "x0": lambda n: np.array([-1.2, 1.0] * (n // 2) + ([-1.2] if n % 2 else []), dtype=float)[:n],
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def rastrigin_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rastrigin_grad(x):
    x = np.asarray(x, dtype=float)
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


def rastrigin_hess(x):
    x = np.asarray(x, dtype=float)
    return np.diag(2.0 + 40.0 * np.pi ** 2 * np.cos(2.0 * np.pi * x))


B2_Rastrigin = {
    "id": "B2_Rastrigin",
    "name": "B2_Rastrigin",
    "f": rastrigin_f,
    "grad_f": rastrigin_grad,
    "hess_f": rastrigin_hess,
    "x0": lambda n: 2.5 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def ackley_f(x, a=20.0, b=0.2, c=2.0 * np.pi):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s1 = np.sqrt(np.sum(x ** 2) / n)
    s2 = np.sum(np.cos(c * x)) / n
    return float(-a * np.exp(-b * s1) - np.exp(s2) + a + np.e)


def ackley_grad(x, a=20.0, b=0.2, c=2.0 * np.pi):
    x = np.asarray(x, dtype=float)
    n = len(x)
    norm2 = np.sum(x ** 2)
    s1 = np.sqrt(norm2 / n)
    s2 = np.sum(np.cos(c * x)) / n
    e1 = np.exp(-b * s1)
    e2 = np.exp(s2)
    if s1 < 1e-15:
        grad_e1 = np.zeros(n)
    else:
        grad_e1 = a * b * e1 * x / (n * s1)
    grad_e2 = e2 * c * np.sin(c * x) / n
    return grad_e1 + grad_e2


def ackley_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = 1e-5
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        H[:, i] = (ackley_grad(xp) - ackley_grad(xm)) / (2.0 * h)
    return H


B3_Ackley = {
    "id": "B3_Ackley",
    "name": "B3_Ackley",
    "f": ackley_f,
    "grad_f": ackley_grad,
    "hess_f": ackley_hess,
    "x0": lambda n: 2.0 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def styblinski_f(x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x))


def styblinski_grad(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (4.0 * x ** 3 - 32.0 * x + 5.0)


def styblinski_hess(x):
    x = np.asarray(x, dtype=float)
    return np.diag(0.5 * (12.0 * x ** 2 - 32.0))


B4_StyblinskiTang = {
    "id": "B4_StyblinskiTang",
    "name": "B4_StyblinskiTang",
    "f": styblinski_f,
    "grad_f": styblinski_grad,
    "hess_f": styblinski_hess,
    "x0": lambda n: 2.5 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def griewank_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1, dtype=float)
    return float(np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def griewank_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1, dtype=float)
    cos_terms = np.cos(x / np.sqrt(i))
    prod_all = np.prod(cos_terms)
    g = x / 2000.0
    for j in range(n):
        if abs(cos_terms[j]) > 1e-15:
            g[j] += prod_all * np.tan(x[j] / np.sqrt(i[j])) / np.sqrt(i[j])
    return g


def griewank_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = 1e-5
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        H[:, i] = (griewank_grad(xp) - griewank_grad(xm)) / (2.0 * h)
    return H


B5_Griewank = {
    "id": "B5_Griewank",
    "name": "B5_Griewank",
    "f": griewank_f,
    "grad_f": griewank_grad,
    "hess_f": griewank_hess,
    "x0": lambda n: 5.0 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def broydn3d_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    total = 0.0
    for i in range(n):
        xi_m = x[i - 1] if i > 0 else 0.0
        xi_p = x[i + 1] if i < n - 1 else 0.0
        fi = (3.0 - 2.0 * x[i]) * x[i] - xi_m - 2.0 * xi_p + 1.0
        total += fi ** 2
    return float(total)


def broydn3d_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    F = np.zeros(n, dtype=float)
    for i in range(n):
        xi_m = x[i - 1] if i > 0 else 0.0
        xi_p = x[i + 1] if i < n - 1 else 0.0
        F[i] = (3.0 - 2.0 * x[i]) * x[i] - xi_m - 2.0 * xi_p + 1.0
    g = np.zeros(n, dtype=float)
    for i in range(n):
        g[i] += 2.0 * F[i] * (3.0 - 4.0 * x[i])
        if i < n - 1:
            g[i] += 2.0 * F[i + 1] * (-1.0)
        if i > 0:
            g[i] += 2.0 * F[i - 1] * (-2.0)
    return g


def broydn3d_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        J[i, i] = 3.0 - 4.0 * x[i]
        if i > 0:
            J[i, i - 1] = -1.0
        if i < n - 1:
            J[i, i + 1] = -2.0
    return 2.0 * (J.T @ J)


B6_Broydn3D = {
    "id": "B6_Broydn3D",
    "name": "B6_Broydn3D",
    "f": broydn3d_f,
    "grad_f": broydn3d_grad,
    "hess_f": broydn3d_hess,
    "x0": lambda n: -np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def brybnd_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    total = 0.0
    for i in range(n):
        idx = [j for j in range(max(0, i - 5), min(n, i + 6)) if j != i]
        fi = x[i] * (2.0 + 5.0 * x[i] ** 2) + 1.0 - sum(x[j] * (1.0 + x[j]) for j in idx)
        total += fi ** 2
    return float(total)


def brybnd_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    F = np.zeros(n, dtype=float)
    for i in range(n):
        idx = [j for j in range(max(0, i - 5), min(n, i + 6)) if j != i]
        F[i] = x[i] * (2.0 + 5.0 * x[i] ** 2) + 1.0 - sum(x[j] * (1.0 + x[j]) for j in idx)
    g = np.zeros(n, dtype=float)
    for i in range(n):
        g[i] += 2.0 * F[i] * (2.0 + 15.0 * x[i] ** 2)
        for j in range(max(0, i - 5), min(n, i + 6)):
            if j != i:
                g[i] += 2.0 * F[j] * (-(1.0 + 2.0 * x[i]))
    return g


def brybnd_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        idx = [j for j in range(max(0, i - 5), min(n, i + 6)) if j != i]
        J[i, i] = 2.0 + 15.0 * x[i] ** 2
        for j in idx:
            J[i, j] = -(1.0 + 2.0 * x[j])
    return 2.0 * (J.T @ J)


B7_BryBnd = {
    "id": "B7_BryBnd",
    "name": "B7_BryBnd",
    "f": brybnd_f,
    "grad_f": brybnd_grad,
    "hess_f": brybnd_hess,
    "x0": lambda n: -np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def _chainwoo_adjust_n(n: int) -> int:
    return max(4, (n // 4) * 4)


def chainwoo_f(x):
    x = np.asarray(x, dtype=float)
    _check_n_multiple_of_4(len(x), "ChainWoo")
    total = 0.0
    for i in range(0, len(x), 4):
        x1, x2, x3, x4 = x[i:i + 4]
        total += (
            100.0 * (x2 - x1 ** 2) ** 2 + (1.0 - x1) ** 2
            + 90.0 * (x4 - x3 ** 2) ** 2 + (1.0 - x3) ** 2
            + 10.0 * (x2 + x4 - 2.0) ** 2 + 0.1 * (x2 - x4) ** 2
        )
    return float(total)


def chainwoo_grad(x):
    x = np.asarray(x, dtype=float)
    _check_n_multiple_of_4(len(x), "ChainWoo")
    g = np.zeros_like(x)
    for i in range(0, len(x), 4):
        x1, x2, x3, x4 = x[i:i + 4]
        g[i + 0] = -400.0 * x1 * (x2 - x1 ** 2) - 2.0 * (1.0 - x1)
        g[i + 1] = 200.0 * (x2 - x1 ** 2) + 20.0 * (x2 + x4 - 2.0) + 0.2 * (x2 - x4)
        g[i + 2] = -360.0 * x3 * (x4 - x3 ** 2) - 2.0 * (1.0 - x3)
        g[i + 3] = 180.0 * (x4 - x3 ** 2) + 20.0 * (x2 + x4 - 2.0) - 0.2 * (x2 - x4)
    return g


def chainwoo_hess(x):
    x = np.asarray(x, dtype=float)
    _check_n_multiple_of_4(len(x), "ChainWoo")
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    for i in range(0, n, 4):
        x1, x2, x3, x4 = x[i:i + 4]
        H[i + 0, i + 0] = 1200.0 * x1 ** 2 - 400.0 * x2 + 2.0
        H[i + 0, i + 1] = H[i + 1, i + 0] = -400.0 * x1
        H[i + 1, i + 1] = 220.2
        H[i + 1, i + 3] = H[i + 3, i + 1] = 19.8
        H[i + 2, i + 2] = 1080.0 * x3 ** 2 - 360.0 * x4 + 2.0
        H[i + 2, i + 3] = H[i + 3, i + 2] = -360.0 * x3
        H[i + 3, i + 3] = 200.2
    return H


B8_ChainWoo = {
    "id": "B8_ChainWoo",
    "name": "B8_ChainWoo",
    "f": chainwoo_f,
    "grad_f": chainwoo_grad,
    "hess_f": chainwoo_hess,
    "x0": lambda n: np.tile(np.array([-3.0, -1.0, -3.0, -1.0], dtype=float), _chainwoo_adjust_n(n) // 4),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def schwefel_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def schwefel_grad(x):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i, xi in enumerate(x):
        sq = math.sqrt(abs(xi) + 1e-15)
        g[i] = -(math.sin(sq) + xi * math.cos(sq) * np.sign(xi) / (2.0 * sq))
    return g


def schwefel_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = 1e-5
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        H[i, i] = (schwefel_grad(xp)[i] - schwefel_grad(xm)[i]) / (2.0 * h)
    return H


B10_Schwefel = {
    "id": "B10_Schwefel",
    "name": "B10_Schwefel",
    "f": schwefel_f,
    "grad_f": schwefel_grad,
    "hess_f": schwefel_hess,
    "x0": lambda n: 400.0 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def dixonprice_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float((x[0] - 1.0) ** 2 + sum((i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, n)))


def dixonprice_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n, dtype=float)
    g[0] = 2.0 * (x[0] - 1.0)
    for i in range(1, n):
        fi = 2.0 * x[i] ** 2 - x[i - 1]
        g[i] += (i + 1) * 2.0 * fi * 4.0 * x[i]
        g[i - 1] += (i + 1) * 2.0 * fi * (-1.0)
    return g


def dixonprice_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    H[0, 0] = 2.0
    for i in range(1, n):
        fi = 2.0 * x[i] ** 2 - x[i - 1]
        H[i, i] += 2.0 * (i + 1) * (16.0 * x[i] ** 2 + 4.0 * fi)
        H[i - 1, i - 1] += 2.0 * (i + 1)
        H[i, i - 1] = H[i - 1, i] = 2.0 * (i + 1) * (-4.0 * x[i])
    return H


B11_DixonPrice = {
    "id": "B11_DixonPrice",
    "name": "B11_DixonPrice",
    "f": dixonprice_f,
    "grad_f": dixonprice_grad,
    "hess_f": dixonprice_hess,
    "x0": lambda n: 2.0 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}


def zakharov_f(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1, dtype=float)
    s2 = np.sum(0.5 * i * x)
    return float(np.sum(x ** 2) + s2 ** 2 + s2 ** 4)


def zakharov_grad(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1, dtype=float)
    s2 = np.sum(0.5 * i * x)
    return 2.0 * x + (2.0 * s2 + 4.0 * s2 ** 3) * 0.5 * i


def zakharov_hess(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1, dtype=float)
    s2 = np.sum(0.5 * i * x)
    v = 0.5 * i
    return 2.0 * np.eye(len(x)) + (2.0 + 12.0 * s2 ** 2) * np.outer(v, v)


B12_Zakharov = {
    "id": "B12_Zakharov",
    "name": "B12_Zakharov",
    "f": zakharov_f,
    "grad_f": zakharov_grad,
    "hess_f": zakharov_hess,
    "x0": lambda n: 0.5 * np.ones(n),
    "f_opt": 0.0,
    "kappa": "non-convexe",
}

PROBLEMS_B: Dict[str, dict] = {
    "B1_Rosenbrock": B1_Rosenbrock,
    "B2_Rastrigin": B2_Rastrigin,
    "B3_Ackley": B3_Ackley,
    "B4_StyblinskiTang": B4_StyblinskiTang,
    "B5_Griewank": B5_Griewank,
    "B6_Broydn3D": B6_Broydn3D,
    "B7_BryBnd": B7_BryBnd,
    "B8_ChainWoo": B8_ChainWoo,
    "B10_Schwefel": B10_Schwefel,
    "B11_DixonPrice": B11_DixonPrice,
    "B12_Zakharov": B12_Zakharov,
}


# =============================================================================
# CATÉGORIE C
# =============================================================================
def _c01_build(n):
    d = np.empty(n, dtype=float)
    for i in range(n):
        sign = 1.0 if i % 2 == 0 else -1.0
        d[i] = sign * (i // 2 + 1)
    return d


def c01_f(x):
    x = np.asarray(x, dtype=float)
    d = _c01_build(len(x))
    return float(np.dot(d, x ** 2))


def c01_grad(x):
    x = np.asarray(x, dtype=float)
    return 2.0 * _c01_build(len(x)) * x


def c01_hess(x):
    return np.diag(2.0 * _c01_build(len(x)))


C01 = {
    "id": "C01",
    "name": "Quadratique indéfinie pure",
    "f": c01_f,
    "grad_f": c01_grad,
    "hess_f": c01_hess,
    "x0": lambda n: 0.1 * np.ones(n),
    "x_saddle": lambda n: np.zeros(n),
    "kappa": "n",
    "dims": "all",
}


def c02_f(x):
    a, b = map(float, x)
    return float((a ** 2 + b - 11) ** 2 + (a + b ** 2 - 5) ** 2)


def c02_grad(x):
    a, b = map(float, x)
    r1 = a ** 2 + b - 11
    r2 = a + b ** 2 - 5
    return np.array([4.0 * a * r1 + 2.0 * r2, 2.0 * r1 + 4.0 * b * r2], dtype=float)


def c02_hess(x):
    a, b = map(float, x)
    H = np.zeros((2, 2), dtype=float)
    H[0, 0] = 12.0 * a ** 2 + 4.0 * b - 42.0
    H[0, 1] = H[1, 0] = 4.0 * a + 4.0 * b
    H[1, 1] = 12.0 * b ** 2 + 4.0 * a - 26.0
    return H


C02 = {
    "id": "C02",
    "name": "Himmelblau (n=2, 4 minima)",
    "f": c02_f,
    "grad_f": c02_grad,
    "hess_f": c02_hess,
    "x0": lambda n: np.array([2.0, 0.0], dtype=float),
    "x_saddle": lambda n: np.array([2.0, 0.0], dtype=float),
    "kappa": "~50",
    "dims": "n=2",
}


def c03_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    t1 = -20.0 * np.exp(-0.2 * norm(x) / np.sqrt(n))
    t2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n)
    return float(t1 + t2 + 20.0 + np.e)


def c03_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    r = norm(x)
    eps = 1e-14
    a = 0.2 / np.sqrt(n)
    if r < eps:
        d1 = np.zeros(n)
    else:
        d1 = 20.0 * a * np.exp(-a * r) * x / r
    S = np.sum(np.cos(2.0 * np.pi * x)) / n
    d2 = (2.0 * np.pi / n) * np.exp(S) * np.sin(2.0 * np.pi * x)
    return d1 + d2


def c03_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    r = norm(x)
    eps = 1e-14
    H = np.zeros((n, n), dtype=float)
    a = 0.2 / np.sqrt(n)
    E = np.exp(-a * r)
    if r > eps:
        xx = np.outer(x, x)
        H += 20.0 * a * E * (np.eye(n) / r - xx / r ** 3 - a * xx / r ** 2)
    M = np.sum(np.cos(2.0 * np.pi * x)) / n
    eM = np.exp(M)
    s = np.sin(2.0 * np.pi * x)
    c = np.cos(2.0 * np.pi * x)
    H += eM * (-(2.0 * np.pi) ** 2 / n ** 2 * np.outer(s, s) + (2.0 * np.pi) ** 2 / n * np.diag(c))
    return H


C03 = {
    "id": "C03",
    "name": "Ackley (indéfinie à l'origine)",
    "f": c03_f,
    "grad_f": c03_grad,
    "hess_f": c03_hess,
    "x0": lambda n: np.tile(np.array([0.1, 0.4], dtype=float), -(-n // 2))[:n],
    "x_saddle": lambda n: np.tile(np.array([0.1, 0.4], dtype=float), -(-n // 2))[:n],
    "kappa": "~100",
    "dims": "all",
}


def c04_f(x):
    a, b = map(float, x)
    A = 1.0 + (a + b + 1.0) ** 2 * (19 - 14 * a + 3 * a ** 2 - 14 * b + 6 * a * b + 3 * b ** 2)
    B = 30.0 + (2 * a - 3 * b) ** 2 * (18 - 32 * a + 12 * a ** 2 + 48 * b - 36 * a * b + 27 * b ** 2)
    return float(A * B)


def c04_grad(x):
    x = np.asarray(x, dtype=float)
    h = 1e-6
    g = np.zeros(2, dtype=float)
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = h
        g[i] = (c04_f(x + ei) - c04_f(x - ei)) / (2.0 * h)
    return g


def c04_hess(x):
    x = np.asarray(x, dtype=float)
    h = 1e-5
    H = np.zeros((2, 2), dtype=float)
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = h
        H[:, i] = (c04_grad(x + ei) - c04_grad(x - ei)) / (2.0 * h)
    return 0.5 * (H + H.T)


C04 = {
    "id": "C04",
    "name": "Goldstein-Price (n=2)",
    "f": c04_f,
    "grad_f": c04_grad,
    "hess_f": c04_hess,
    "x0": lambda n: np.array([-0.5, 0.25], dtype=float),
    "x_saddle": lambda n: np.array([-0.5, 0.25], dtype=float),
    "kappa": "~1e6",
    "dims": "n=2",
}

_GAMMA = 0.1


def c05_f(x):
    x = np.asarray(x, dtype=float)
    s = np.sum((x ** 2 - 1.0) ** 2)
    s += _GAMMA * np.sum((x[1:] - x[:-1]) ** 2)
    return float(s)


def c05_grad(x):
    x = np.asarray(x, dtype=float)
    g = 4.0 * x * (x ** 2 - 1.0)
    g[:-1] -= 2.0 * _GAMMA * (x[1:] - x[:-1])
    g[1:] += 2.0 * _GAMMA * (x[1:] - x[:-1])
    return g


def c05_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0 or i == n - 1:
            H[i, i] = 12.0 * x[i] ** 2 - 4.0 + 2.0 * _GAMMA
        else:
            H[i, i] = 12.0 * x[i] ** 2 - 4.0 + 4.0 * _GAMMA
    for i in range(n - 1):
        H[i, i + 1] = H[i + 1, i] = -2.0 * _GAMMA
    return H


C05 = {
    "id": "C05",
    "name": "Chaîne double-puits (2ⁿ minima)",
    "f": c05_f,
    "grad_f": c05_grad,
    "hess_f": c05_hess,
    "x0": lambda n: np.tile(np.array([0.8, 0.4], dtype=float), -(-n // 2))[:n],
    "x_saddle": lambda n: np.tile(np.array([0.8, 0.4], dtype=float), -(-n // 2))[:n],
    "kappa": "~40",
    "dims": "all",
}

_A_RAST = 10.0


def c06_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(_A_RAST * n + np.sum(x ** 2 - _A_RAST * np.cos(2.0 * np.pi * x)))


def c06_grad(x):
    x = np.asarray(x, dtype=float)
    return 2.0 * x + 2.0 * np.pi * _A_RAST * np.sin(2.0 * np.pi * x)


def c06_hess(x):
    x = np.asarray(x, dtype=float)
    return np.diag(2.0 + 4.0 * np.pi ** 2 * _A_RAST * np.cos(2.0 * np.pi * x))


C06 = {
    "id": "C06",
    "name": "Rastrigin (indéfinie presque partout)",
    "f": c06_f,
    "grad_f": c06_grad,
    "hess_f": c06_hess,
    "x0": lambda n: np.tile(np.array([0.1, 0.3], dtype=float), -(-n // 2))[:n],
    "x_saddle": lambda n: np.tile(np.array([0.1, 0.3], dtype=float), -(-n // 2))[:n],
    "kappa": "~200",
    "dims": "all",
}


def c07_f(x):
    x = np.asarray(x, dtype=float)
    n_pairs = min(len(x[0::2]), len(x[1::2]))
    return float(np.sum(x[0:2 * n_pairs:2] ** 2 - x[1:2 * n_pairs:2] ** 2) + 0.01 * np.sum(x ** 4))


def c07_grad(x):
    x = np.asarray(x, dtype=float)
    g = 0.04 * x ** 3
    g[0::2] += 2.0 * x[0::2]
    g[1::2] -= 2.0 * x[1::2]
    return g


def c07_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.diag(0.12 * x ** 2)
    idx_even = np.arange(0, n, 2)
    idx_odd = np.arange(1, n, 2)
    H[idx_even, idx_even] += 2.0
    H[idx_odd, idx_odd] -= 2.0
    return H


C07 = {
    "id": "C07",
    "name": "Selle hyperbolique étendue",
    "f": c07_f,
    "grad_f": c07_grad,
    "hess_f": c07_hess,
    "x0": lambda n: 0.1 * np.ones(n),
    "x_saddle": lambda n: np.zeros(n),
    "kappa": "1",
    "dims": "n pair",
}

_MU = 8.0


def c08_f(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = (x[0] - 1.0) ** 2
    for i in range(1, n):
        s += (i + 1) * (2.0 * x[i] ** 2 - x[i - 1]) ** 2
    return float(s - _MU * np.sum(x ** 2))


def c08_grad(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n, dtype=float)
    g[0] = 2.0 * (x[0] - 1.0)
    for i in range(1, n):
        t = 2.0 * x[i] ** 2 - x[i - 1]
        g[i] += 8.0 * (i + 1) * x[i] * t
        g[i - 1] += -2.0 * (i + 1) * t
    g -= 2.0 * _MU * x
    return g


def c08_hess(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    H[0, 0] = 2.0
    for i in range(1, n):
        t = 2.0 * x[i] ** 2 - x[i - 1]
        H[i, i] += 8.0 * (i + 1) * (t + 4.0 * x[i] ** 2)
        H[i, i - 1] += -8.0 * (i + 1) * x[i]
        H[i - 1, i] = H[i, i - 1]
        H[i - 1, i - 1] += 2.0 * (i + 1)
    H -= 2.0 * _MU * np.eye(n)
    return H


C08 = {
    "id": "C08",
    "name": f"Dixon-Price perturbé (μ={_MU}, H indéfinie)",
    "f": c08_f,
    "grad_f": c08_grad,
    "hess_f": c08_hess,
    "x0": lambda n: 2.0 * np.ones(n),
    "x_saddle": lambda n: np.zeros(n),
    "kappa": "variable",
    "dims": "all",
}

_NU = 3.0


def c09_f(x):
    x = np.asarray(x, dtype=float)
    n2 = len(x) // 2
    s = 0.0
    for k in range(n2):
        a, b = x[2 * k], x[2 * k + 1]
        s += (a + 2.0 * b - 7.0) ** 2 + (2.0 * a + b - 5.0) ** 2 - _NU * (a * b) ** 2
    return float(s)


def c09_grad(x):
    x = np.asarray(x, dtype=float)
    n2 = len(x) // 2
    g = np.zeros_like(x)
    for k in range(n2):
        a, b = x[2 * k], x[2 * k + 1]
        r1 = a + 2.0 * b - 7.0
        r2 = 2.0 * a + b - 5.0
        g[2 * k] = 2.0 * r1 + 4.0 * r2 - 2.0 * _NU * a * b ** 2
        g[2 * k + 1] = 4.0 * r1 + 2.0 * r2 - 2.0 * _NU * a ** 2 * b
    return g


def c09_hess(x):
    x = np.asarray(x, dtype=float)
    n2 = len(x) // 2
    H = np.zeros((len(x), len(x)), dtype=float)
    for k in range(n2):
        a, b = x[2 * k], x[2 * k + 1]
        H[2 * k, 2 * k] = 10.0 - 2.0 * _NU * b ** 2
        H[2 * k + 1, 2 * k + 1] = 10.0 - 2.0 * _NU * a ** 2
        H[2 * k, 2 * k + 1] = H[2 * k + 1, 2 * k] = 8.0 - 4.0 * _NU * a * b
    return H


C09 = {
    "id": "C09",
    "name": f"Booth perturbé (ν={_NU}, courbure croisée négative)",
    "f": c09_f,
    "grad_f": c09_grad,
    "hess_f": c09_hess,
    "x0": lambda n: 1.5 * np.ones(n),
    "x_saddle": lambda n: 1.5 * np.ones(n),
    "kappa": "~5",
    "dims": "n pair",
}

_RHO = 0.7


def _c10_build(n):
    T = 2.0 * np.eye(n)
    for i in range(n - 1):
        T[i, i + 1] = T[i + 1, i] = -1.0
    return T - 4.0 * _RHO * np.eye(n)


def c10_f(x):
    x = np.asarray(x, dtype=float)
    M = _c10_build(len(x))
    return float(0.5 * x @ M @ x)


def c10_grad(x):
    x = np.asarray(x, dtype=float)
    M = _c10_build(len(x))
    return M @ x


def c10_hess(x):
    return _c10_build(len(x))


C10 = {
    "id": "C10",
    "name": f"Laplacien décalé (ρ={_RHO}, ~70% v.p. < 0)",
    "f": c10_f,
    "grad_f": c10_grad,
    "hess_f": c10_hess,
    "x0": lambda n: np.random.default_rng(0).standard_normal(n) * 0.1,
    "x_saddle": lambda n: np.zeros(n),
    "kappa": "~n²",
    "dims": "all",
}

PROBLEMS_C: Dict[str, dict] = {
    "C01": C01,
    "C02": C02,
    "C03": C03,
    "C04": C04,
    "C05": C05,
    "C06": C06,
    "C07": C07,
    "C08": C08,
    "C09": C09,
    "C10": C10,
}


# =============================================================================
# Registres publics
# =============================================================================
PROBLEMS_A_IDS = list(PROBLEMS_A.keys())
PROBLEMS_B_IDS = [pid for pid in PROBLEMS_B.keys() if pid not in EXCLUDED_IDS]
PROBLEMS_C_IDS = list(PROBLEMS_C.keys())
PROBLEMS_ALL_IDS = PROBLEMS_A_IDS + PROBLEMS_B_IDS + PROBLEMS_C_IDS

_CATEGORY_BY_ID = {}
_CATEGORY_BY_ID.update({pid: "A" for pid in PROBLEMS_A_IDS})
_CATEGORY_BY_ID.update({pid: "B" for pid in PROBLEMS_B_IDS})
_CATEGORY_BY_ID.update({pid: "C" for pid in PROBLEMS_C_IDS})


def get_problem_A(pid: str, n: int = 10) -> dict:
    if pid not in PROBLEMS_A:
        raise KeyError(f"Problème A inconnu: {pid}")
    n_use = adjust_n(pid, n)
    return _normalize_problem(PROBLEMS_A[pid], "A", pid, n_use)


def get_problem_B(pid: str, n: int = 10) -> dict:
    if pid in EXCLUDED_IDS:
        raise KeyError(f"Le problème {pid} a été explicitement exclu du benchmark.")
    if pid not in PROBLEMS_B:
        raise KeyError(f"Problème B inconnu: {pid}")
    n_use = adjust_n(pid, n)
    return _normalize_problem(PROBLEMS_B[pid], "B", pid, n_use)


def get_problem_C(pid: str, n: int = 10) -> dict:
    if pid not in PROBLEMS_C:
        raise KeyError(f"Problème C inconnu: {pid}")
    n_use = adjust_n(pid, n)
    return _normalize_problem(PROBLEMS_C[pid], "C", pid, n_use)


def get_problem(pid: str, n: int = 10) -> dict:
    if pid in EXCLUDED_IDS:
        raise KeyError(f"Le problème {pid} a été explicitement exclu du benchmark.")
    if pid not in _CATEGORY_BY_ID:
        raise KeyError(f"Identifiant inconnu: {pid}. Disponibles: {PROBLEMS_ALL_IDS}")
    cat = _CATEGORY_BY_ID[pid]
    if cat == "A":
        return get_problem_A(pid, n)
    if cat == "B":
        return get_problem_B(pid, n)
    return get_problem_C(pid, n)


def build_problem_list(dims: Optional[List[int]] = None,
                       categories: Optional[List[str]] = None,
                       exclude_ids: Optional[List[str]] = None) -> List[dict]:
    if dims is None:
        dims = DIMS
    if categories is None:
        categories = ["A", "B", "C"]
    effective_exclude = set(EXCLUDED_IDS)
    if exclude_ids is not None:
        effective_exclude.update(exclude_ids)
    out = []
    for pid in PROBLEMS_ALL_IDS:
        if pid in effective_exclude:
            continue
        if pid[0] not in categories:
            continue
        for n in dims:
            try:
                out.append(get_problem(pid, n))
            except Exception:
                continue
    return out


def describe_registry() -> dict:
    return {
        "excluded_ids": sorted(EXCLUDED_IDS),
        "n_A": len(PROBLEMS_A_IDS),
        "n_B": len(PROBLEMS_B_IDS),
        "n_C": len(PROBLEMS_C_IDS),
        "n_total": len(PROBLEMS_ALL_IDS),
        "A_ids": PROBLEMS_A_IDS,
        "B_ids": PROBLEMS_B_IDS,
        "C_ids": PROBLEMS_C_IDS,
    }


# =============================================================================
# Affichages de validation par catégorie
# =============================================================================
def _default_n_for_pid(pid: str) -> int:
    if pid in FIXED_DIM:
        return FIXED_DIM[pid]
    if pid == "A09":
        return 12
    if pid in ("C07", "C09"):
        return 10
    if pid == "B8_ChainWoo":
        return 8
    return 10


def _fmt_ok(err: float) -> str:
    return f"OK ({err:.1e})"


def _fmt_status(status: str, err: float) -> str:
    if status == "OK":
        return f"OK({err:.0e})"
    return f"ECHEC({err:.0e})"


def print_category_A_details() -> None:
    print(f"{'ID':>4}  {'Nom':<49} {'n':>6}  {'κ':<23} {'grad_check':>12}")
    print('─' * 100)
    for pid in PROBLEMS_A_IDS:
        try:
            n = _default_n_for_pid(pid)
            pb = get_problem_A(pid, n)
            status, err = gradient_check(pb)
            kappa = str(pb.get('kappa', '?'))
            cell = _fmt_ok(err) if status == 'OK' else f"ECHEC ({err:.1e})"
            print(f"{pb['id']:>4}  {pb['name']:<49} {pb['n']:>6}  {kappa:<23} {cell:>12}")
        except Exception as e:
            print(f"{pid:>4}  {'ERREUR':<49} {'ERR':>6}  {str(e)[:23]:<23}")
    print()


def print_category_B_details() -> None:
    print("=== Vérification des gradients Catégorie B ===")
    for pid in PROBLEMS_B_IDS:
        try:
            n = _default_n_for_pid(pid)
            pb = get_problem_B(pid, n)
            status, err = gradient_check(pb)
            print(f"  {pb['id']:<25}: grad_check = {status} (err={err:.2e})")
        except Exception as e:
            print(f"  {pid:<25}: exception → {e}")
    print()
    print("=== Test get_problem_B ===")
    for pid in PROBLEMS_B_IDS:
        try:
            n = _default_n_for_pid(pid)
            pb = get_problem_B(pid, n)
            fx0 = pb['f'](pb['x0'])
            print(f"  {pb['id']:<25} | n={pb['n']:<4} | f(x0)={fx0:.4f}")
        except Exception as e:
            print(f"  {pid:<25} | ERREUR: {e}")
    print()


def _get_saddle_info(pb: dict) -> Tuple[int, float, float]:
    x_saddle = pb.get('x_saddle', None)
    if callable(x_saddle):
        x_saddle = x_saddle(pb['n'])
    if x_saddle is None:
        x_saddle = pb['x0']
    x_saddle = np.asarray(x_saddle, dtype=float)
    Hs = np.asarray(pb['hess_f'](x_saddle), dtype=float)
    eigs = eigvalsh(Hs)
    n_neg = int(np.sum(eigs < -1e-12))
    return n_neg, float(eigs.min()), float(eigs.max())


def print_category_C_details() -> None:
    print(f"{'ID':>4}  {'Nom':<46} {'n':>4}  {'#v.p.<0':>8}  {'∇check':>10}  {'H check':>10}  {'v.p. à x_saddle (min,max)':>28}")
    print('─' * 110)
    for pid in PROBLEMS_C_IDS:
        try:
            n = _default_n_for_pid(pid)
            pb = get_problem_C(pid, n)
            gs, ge = gradient_check(pb)
            hs, he = hessian_check(pb)
            n_neg, lam_min, lam_max = _get_saddle_info(pb)
            print(
                f"{pb['id']:>4}  {pb['name']:<46} {pb['n']:>4}  {n_neg:>8}  {_fmt_status(gs, ge):>10}  {_fmt_status(hs, he):>10}  [{lam_min:.2f}, {lam_max:.2f}]"
            )
        except Exception as e:
            print(f"{pid:>4}  {'ERREUR':<46} {'ERR':>4}  {str(e)[:60]}")
    print()


def print_all_categories_details() -> None:
    print_category_A_details()
    print_category_B_details()
    print_category_C_details()


if __name__ == "__main__":
    print_all_categories_details()
