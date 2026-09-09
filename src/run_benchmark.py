
# -*- coding: utf-8 -*-
"""
run_benchmark.py
------------------------------------------------------------
Framework benchmark unifié compatible avec :
- test_problems.py
- trust_region.py
- line_search.py

Fonctionnalités :
- exécution sérielle ou parallèle
- métriques : itérations, évaluations f/g/h, temps CPU, statut, norme du gradient
- compatibilité avec OptResult (trust-region) et dictionnaires d'historique (line-search)
- export CSV et résumé console
"""

import argparse
import csv
import importlib.util
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _load(path: str, name: str):
    candidates = [
        path,
        os.path.join(BASE_DIR, path),
        os.path.join(os.getcwd(), path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location(name, candidate)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        f"Impossible de trouver '{path}'. Placez tous les fichiers .py nécessaires dans le même dossier que ce script.\n"
        f"Chemins testés : {candidates}"
    )


_mod_all = _load("test_problems.py", "bench_all")
_tr = _load("trust_region.py", "tr_methods")
_ls = _load("line_search.py", "ls_methods")


def _pick_callable(module, candidates, label):
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"Impossible de trouver la fonction pour '{label}' dans le module '{module.__name__}'.\n"
        f"Noms testés : {candidates}"
    )


TR_CAUCHY = _pick_callable(_tr, ["trust_region_cauchy", "tr_cauchy", "cauchy_point", "trust_region_cp"], "Cauchy Point")
TR_DOGLEG = _pick_callable(_tr, ["trust_region_dogleg", "tr_dogleg", "dogleg"], "Dogleg")
TR_STEIHAUG = _pick_callable(_tr, ["trust_region_steihaug", "tr_steihaug", "steihaug", "steihaug_cg"], "Steihaug CG")

LS_BFGS_ARMIJO = _pick_callable(_ls, ["bfgs", "bfgs_armijo", "BFGS_Armijo", "bfgs_backtracking_armijo"], "BFGS Armijo")
LS_BFGS_WOLFE = _pick_callable(_ls, ["bfgs_wolfe", "BFGS_Wolfe", "bfgs_strong_wolfe"], "BFGS Wolfe")
LS_NEWTON_BT = _pick_callable(_ls, ["newton_backtracking", "newton_bt", "Newton_BT", "newton_backtracking_armijo"], "Newton Backtracking")

DIMS_DEFAULT = list(getattr(_mod_all, "DIMS", [10, 50, 100, 500, 1000]))
TOL = 1e-6
FIELDS = [
    "problem", "name", "category", "method", "n",
    "status", "converged", "n_iter", "n_f_evals", "n_g_evals", "n_h_evals", "n_total",
    "cpu_s", "f_final", "f_error", "grad_norm", "error_msg"
]

PROBLEMS_A_IDS = list(getattr(_mod_all, "PROBLEMS_A_IDS", []))
PROBLEMS_B_IDS = list(getattr(_mod_all, "PROBLEMS_B_IDS", []))
PROBLEMS_C_IDS = list(getattr(_mod_all, "PROBLEMS_C_IDS", []))
PROBLEMS_ALL_IDS = list(getattr(_mod_all, "PROBLEMS_ALL_IDS", []))


def _adjust_n(pid: str, n: int) -> int:
    return _mod_all.adjust_n(pid, n) if hasattr(_mod_all, "adjust_n") else n


def _get_problem(pid: str, n: int = 10) -> dict:
    pb = _mod_all.get_problem(pid, n)
    return {
        "id": pb["id"],
        "name": pb["name"],
        "category": pb["category"],
        "n": int(pb["n"]),
        "f": pb["f"],
        "grad_f": pb["grad_f"],
        "hess_f": pb["hess_f"],
        "x0": np.asarray(pb["x0"], dtype=np.float64),
        "f_opt": pb.get("f_opt", 0.0),
        "kappa": str(pb.get("kappa", "?")),
        "n_neg_eigs": pb.get("n_neg_eigs", "?"),
    }


def build_problem_specs(dims: Optional[List[int]] = None,
                        categories: Optional[List[str]] = None,
                        ids: Optional[List[str]] = None) -> List[Tuple[str, int]]:
    if dims is None:
        dims = DIMS_DEFAULT
    if categories is None:
        categories = ["A", "B", "C"]
    if ids is None:
        ids = list(PROBLEMS_ALL_IDS)

    specs = []
    for pid in ids:
        cat = pid[0].upper() if pid else "?"
        if cat not in categories:
            continue
        for n in dims:
            try:
                n_use = _adjust_n(pid, n)
                _get_problem(pid, n_use)
                specs.append((pid, n_use))
            except Exception:
                continue
    return specs


def _make_counted_oracles(pb: dict):
    counts = {"f": 0, "g": 0, "h": 0}

    def f_counted(x):
        counts["f"] += 1
        return pb["f"](x)

    def g_counted(x):
        counts["g"] += 1
        return pb["grad_f"](x)

    def h_counted(x):
        counts["h"] += 1
        return pb["hess_f"](x)

    return f_counted, g_counted, h_counted, counts


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _hist_get(hist: dict, keys, default=None):
    if not isinstance(hist, dict):
        return default
    for k in keys:
        if k in hist:
            return hist[k]
    return default


def _extract_solver_output(result, pb: dict):
    x_final = None
    hist = {}

    if hasattr(result, "x") and hasattr(result, "iterations"):
        x_final = np.asarray(result.x, dtype=np.float64)
        hist = {
            "n_iter": getattr(result, "iterations", 0),
            "n_f_evals": getattr(result, "f_evals", 0),
            "n_g_evals": getattr(result, "g_evals", 0),
            "cpu_time": getattr(result, "cpu_time", None),
            "converged": getattr(result, "converged", False),
            "history": getattr(result, "history", []),
            "f_final": getattr(result, "f", None),
            "grad_norm": getattr(result, "grad_norm", None),
            "status": "converged" if getattr(result, "converged", False) else "max_iter",
        }
        return x_final, hist

    if isinstance(result, tuple):
        if len(result) >= 1:
            x_final = result[0]
        if len(result) >= 2 and isinstance(result[1], dict):
            hist = result[1]
    elif isinstance(result, dict):
        hist = result
        if "x_opt" in result:
            x_final = result["x_opt"]
        elif "x_final" in result:
            x_final = result["x_final"]
        elif "solution" in result:
            x_final = result["solution"]
        elif "xmin" in result:
            x_final = result["xmin"]
        elif "x" in result:
            x_candidate = result["x"]
            if isinstance(x_candidate, (list, tuple)) and len(x_candidate) > 0:
                x_final = x_candidate[-1]
            else:
                arr = np.asarray(x_candidate)
                if arr.ndim == 1:
                    x_final = arr
                elif arr.ndim >= 2 and len(arr) > 0:
                    x_final = arr[-1]
    else:
        x_final = result

    if x_final is None:
        x_final = pb["x0"]
    return np.asarray(x_final, dtype=np.float64), hist


def _run_tr(method_fn, f, g, h, x0):
    return method_fn(f, g, h, x0.copy())


def _run_ls_first_order(method_fn, f, g, x0):
    return method_fn(f, g, x0.copy())


def _run_ls_second_order(method_fn, f, g, h, x0):
    return method_fn(f, g, h, x0.copy())


METHODS = {
    "Cauchy_Point": ("TR", TR_CAUCHY),
    "Dogleg": ("TR", TR_DOGLEG),
    "Steihaug_CG": ("TR", TR_STEIHAUG),
    "BFGS_Armijo": ("LS1", LS_BFGS_ARMIJO),
    "BFGS_Wolfe": ("LS1", LS_BFGS_WOLFE),
    "Newton_BT": ("LS2", LS_NEWTON_BT),
}


def run_one(pb: dict, method_name: str) -> dict:
    t0 = time.perf_counter()
    f_counted, g_counted, h_counted, counts = _make_counted_oracles(pb)

    try:
        kind, method_fn = METHODS[method_name]
        if kind == "TR":
            result = _run_tr(method_fn, f_counted, g_counted, h_counted, pb["x0"])
        elif kind == "LS1":
            result = _run_ls_first_order(method_fn, f_counted, g_counted, pb["x0"])
        elif kind == "LS2":
            result = _run_ls_second_order(method_fn, f_counted, g_counted, h_counted, pb["x0"])
        else:
            raise ValueError(f"Type de méthode inconnu : {kind}")

        cpu_s = time.perf_counter() - t0
        x_final, hist = _extract_solver_output(result, pb)

        f_final = _safe_float(pb["f"](x_final), np.nan)
        grad_norm = _safe_float(np.linalg.norm(pb["grad_f"](x_final)), np.nan)
        f_opt_ref = _safe_float(pb.get("f_opt", 0.0), 0.0)
        f_error = abs(f_final - f_opt_ref) if np.isfinite(f_final) else np.nan

        history_obj = _hist_get(hist, ["history", "hist", "trajectory", "trace"], None)
        n_iter = _hist_get(hist, ["n_iter", "iterations", "iter", "k"], None)
        if n_iter is None:
            if isinstance(history_obj, (list, tuple)) and len(history_obj) > 0:
                n_iter = len(history_obj)
            else:
                n_iter = 0
        n_iter = _safe_int(n_iter, 0)

        n_f_evals = max(counts["f"], _safe_int(_hist_get(hist, ["n_f_evals", "f_evals", "nfev", "nf"], 0), 0))
        n_g_evals = max(counts["g"], _safe_int(_hist_get(hist, ["n_g_evals", "g_evals", "njev", "ng"], 0), 0))
        n_h_evals = max(counts["h"], _safe_int(_hist_get(hist, ["n_h_evals", "h_evals", "nhev", "nh"], 0), 0))

        status_raw = str(_hist_get(hist, ["status"], "")).lower()
        converged_raw = _hist_get(hist, ["converged", "success"], None)

        if converged_raw is not None:
            converged = bool(converged_raw)
        elif status_raw in {"converged", "success"}:
            converged = True
        else:
            converged = bool(np.isfinite(grad_norm) and grad_norm < TOL)

        if status_raw in {"converged", "success"}:
            status = "success"
        elif status_raw in {"max_iter", "max_iter_or_not_converged"}:
            status = "max_iter_or_not_converged"
        else:
            status = "success" if converged else "max_iter_or_not_converged"

        return {
            "problem": pb["id"],
            "name": pb["name"],
            "category": pb["category"],
            "method": method_name,
            "n": pb["n"],
            "status": status,
            "converged": converged,
            "n_iter": int(n_iter),
            "n_f_evals": int(n_f_evals),
            "n_g_evals": int(n_g_evals),
            "n_h_evals": int(n_h_evals),
            "n_total": int(n_f_evals + n_g_evals + n_h_evals),
            "cpu_s": cpu_s,
            "f_final": f_final,
            "f_error": f_error,
            "grad_norm": grad_norm,
            "error_msg": "",
        }
    except Exception as e:
        cpu_s = time.perf_counter() - t0
        return {
            "problem": pb["id"],
            "name": pb["name"],
            "category": pb["category"],
            "method": method_name,
            "n": pb["n"],
            "status": "exception",
            "converged": False,
            "n_iter": 0,
            "n_f_evals": counts["f"],
            "n_g_evals": counts["g"],
            "n_h_evals": counts["h"],
            "n_total": counts["f"] + counts["g"] + counts["h"],
            "cpu_s": cpu_s,
            "f_final": np.nan,
            "f_error": np.nan,
            "grad_norm": np.nan,
            "error_msg": f"{type(e).__name__}: {e}",
        }


def _run_one_task(args):
    pid, n, method_name = args
    pb = _get_problem(pid, n)
    return run_one(pb, method_name)


def _print_run_header():
    print(f"{'Problème':<16} {'Cat':<4} {'n':>5} {'Méthode':<14} {'Conv':<5} {'Itér':>6} {'Éval.f':>8} {'Éval.g':>8} {'Éval.H':>8} {'CPU(s)':>10} {'||∇f*||':>12}")
    print("-" * 110)


def _print_run_line(res: dict):
    conv = "✓" if res["converged"] else "✗"
    grad_txt = f"{res['grad_norm']:.2e}" if np.isfinite(res['grad_norm']) else "NaN"
    print(f"{res['problem']:<16} {res['category']:<4} {res['n']:>5} {res['method']:<14} {conv:<5} {res['n_iter']:>6} {res['n_f_evals']:>8} {res['n_g_evals']:>8} {res['n_h_evals']:>8} {res['cpu_s']:>10.4f} {grad_txt:>12}")


def run_all(dims: Optional[List[int]] = None,
            methods: Optional[List[str]] = None,
            categories: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            verbose: bool = True,
            parallel: bool = False,
            max_workers: Optional[int] = None):
    if dims is None:
        dims = DIMS_DEFAULT
    if methods is None:
        methods = list(METHODS.keys())
    if categories is None:
        categories = ["A", "B", "C"]

    specs = build_problem_specs(dims=dims, categories=categories, ids=ids)
    tasks = [(pid, n, m) for (pid, n) in specs for m in methods]
    results = []

    if verbose:
        print(f"\nNombre de problèmes instanciés : {len(specs)}")
        print(f"Nombre de méthodes            : {len(methods)}")
        print(f"Nombre total de runs          : {len(tasks)}")
        print(f"Mode parallèle                : {parallel}")
        print()
        _print_run_header()

    if parallel and tasks:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_one_task, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if verbose:
                    _print_run_line(res)
    else:
        for pid, n, m in tasks:
            pb = _get_problem(pid, n)
            res = run_one(pb, m)
            results.append(res)
            if verbose:
                _print_run_line(res)

    return results


def save_csv(results, path: str = "resultats_benchmark.csv"):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n→ Résultats sauvegardés dans : {path}")


def summary(results):
    if not results:
        print("\nAucun résultat à résumer.")
        return

    by_method = defaultdict(lambda: {"runs": 0, "conv": 0, "iters": [], "cpus": [], "f": [], "g": [], "h": []})
    by_cat_method = defaultdict(lambda: {"runs": 0, "conv": 0, "iters": [], "cpus": [], "f": [], "g": [], "h": []})

    for r in results:
        m, c = r["method"], r["category"]
        by_method[m]["runs"] += 1
        by_method[m]["conv"] += int(bool(r["converged"]))
        by_method[m]["iters"].append(r["n_iter"])
        by_method[m]["cpus"].append(r["cpu_s"])
        by_method[m]["f"].append(r["n_f_evals"])
        by_method[m]["g"].append(r["n_g_evals"])
        by_method[m]["h"].append(r["n_h_evals"])

        by_cat_method[(c, m)]["runs"] += 1
        by_cat_method[(c, m)]["conv"] += int(bool(r["converged"]))
        by_cat_method[(c, m)]["iters"].append(r["n_iter"])
        by_cat_method[(c, m)]["cpus"].append(r["cpu_s"])
        by_cat_method[(c, m)]["f"].append(r["n_f_evals"])
        by_cat_method[(c, m)]["g"].append(r["n_g_evals"])
        by_cat_method[(c, m)]["h"].append(r["n_h_evals"])

    print("\n" + "=" * 110)
    print("RÉSUMÉ GLOBAL PAR MÉTHODE")
    print("=" * 110)
    print(f"{'Méthode':<15} {'Runs':>6} {'Succès':>8} {'Taux%':>8} {'Itér.moy':>10} {'f.moy':>10} {'g.moy':>10} {'H.moy':>10} {'CPU.moy(s)':>12}")
    print("-" * 110)
    for m, d in by_method.items():
        runs = d["runs"]
        conv = d["conv"]
        rate = 100.0 * conv / runs if runs else 0.0
        it_mean = np.mean(d["iters"]) if d["iters"] else np.nan
        f_mean = np.mean(d["f"]) if d["f"] else np.nan
        g_mean = np.mean(d["g"]) if d["g"] else np.nan
        h_mean = np.mean(d["h"]) if d["h"] else np.nan
        cpu_mean = np.mean(d["cpus"]) if d["cpus"] else np.nan
        print(f"{m:<15} {runs:>6} {conv:>8} {rate:>8.1f} {it_mean:>10.2f} {f_mean:>10.2f} {g_mean:>10.2f} {h_mean:>10.2f} {cpu_mean:>12.4f}")

    print("\n" + "=" * 122)
    print("RÉSUMÉ PAR CATÉGORIE × MÉTHODE")
    print("=" * 122)
    print(f"{'Cat':<4} {'Méthode':<15} {'Runs':>6} {'Succès':>8} {'Taux%':>8} {'Itér.moy':>10} {'f.moy':>10} {'g.moy':>10} {'H.moy':>10} {'CPU.moy(s)':>12}")
    print("-" * 122)
    for (c, m), d in sorted(by_cat_method.items()):
        runs = d["runs"]
        conv = d["conv"]
        rate = 100.0 * conv / runs if runs else 0.0
        it_mean = np.mean(d["iters"]) if d["iters"] else np.nan
        f_mean = np.mean(d["f"]) if d["f"] else np.nan
        g_mean = np.mean(d["g"]) if d["g"] else np.nan
        h_mean = np.mean(d["h"]) if d["h"] else np.nan
        cpu_mean = np.mean(d["cpus"]) if d["cpus"] else np.nan
        print(f"{c:<4} {m:<15} {runs:>6} {conv:>8} {rate:>8.1f} {it_mean:>10.2f} {f_mean:>10.2f} {g_mean:>10.2f} {h_mean:>10.2f} {cpu_mean:>12.4f}")


def get_problem_by_id(pid: str, n: int):
    return _get_problem(pid, _adjust_n(pid, n))


def debug_single_run(pid: str, method_name: str, n: int):
    pb = get_problem_by_id(pid, n)
    print("\n=== DEBUG MONO-RUN ===")
    print(f"Problème : {pb['id']} | {pb['name']} | catégorie={pb['category']} | n={pb['n']}")
    print(f"Méthode  : {method_name}")
    print(f"f(x0)    : {pb['f'](pb['x0']):.6e}")
    print(f"||g(x0)||: {np.linalg.norm(pb['grad_f'](pb['x0'])):.6e}")
    res = run_one(pb, method_name)
    print("\nRésultat :")
    for k in FIELDS:
        print(f"  {k:<12}: {res.get(k)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark unifié Trust-Region vs Line Search")
    parser.add_argument("--quick", action="store_true", help="Teste seulement n = [10]")
    parser.add_argument("--medium", action="store_true", help="Teste seulement n = [10, 50, 100]")
    parser.add_argument("--category", type=str, default=None, choices=["A", "B", "C"], help="Exécute seulement une catégorie")
    parser.add_argument("--method", type=str, default=None, choices=list(METHODS.keys()), help="Méthode unique à lancer")
    parser.add_argument("--problem", type=str, default=None, help="Identifiant d'un problème unique")
    parser.add_argument("--n", type=int, default=10, help="Dimension si --problem est fourni")
    parser.add_argument("--csv", type=str, default="resultats_benchmark.csv", help="Nom du fichier CSV de sortie")
    parser.add_argument("--quiet", action="store_true", help="Réduit l'affichage console")
    parser.add_argument("--parallel", action="store_true", help="Active l'exécution parallèle")
    parser.add_argument("--workers", type=int, default=None, help="Nombre de workers")
    parser.add_argument("--list-problems", action="store_true", help="Affiche la liste des problèmes puis quitte")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.list_problems:
        print("\n=== Problèmes disponibles ===")
        print(f"A ({len(PROBLEMS_A_IDS)}) : {PROBLEMS_A_IDS}")
        print(f"B ({len(PROBLEMS_B_IDS)}) : {PROBLEMS_B_IDS}")
        print(f"C ({len(PROBLEMS_C_IDS)}) : {PROBLEMS_C_IDS}")
        print(f"Total : {len(PROBLEMS_ALL_IDS)}")
        return

    if args.problem is not None:
        if args.method is None:
            raise ValueError("Si --problem est fourni, vous devez aussi fournir --method.")
        debug_single_run(args.problem, args.method, args.n)
        return

    if args.quick:
        dims = [10]
    elif args.medium:
        dims = [10, 50, 100]
    else:
        dims = DIMS_DEFAULT

    methods = [args.method] if args.method is not None else list(METHODS.keys())
    categories = [args.category] if args.category is not None else ["A", "B", "C"]

    t0 = time.perf_counter()
    results = run_all(dims=dims, methods=methods, categories=categories, verbose=verbose, parallel=args.parallel, max_workers=args.workers)
    total_time = time.perf_counter() - t0

    summary(results)
    save_csv(results, args.csv)
    print(f"\nTemps total benchmark : {total_time:.2f} s")


if __name__ == "__main__":
    main()
