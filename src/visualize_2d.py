
# -*- coding: utf-8 -*-
"""
visualisations_depuis_csv.py
------------------------------------------------------------
Script exécutable pour générer automatiquement les visualisations et tableaux
à partir du fichier resultats_benchmark.csv, conformément au cahier de charge.

Ce script exploite UNIQUEMENT les résultats du framework (CSV) et produit :
- tableaux récapitulatifs (global + catégorie × méthode) ;
- taux de convergence / robustesse ;
- profils de performance de Dolan–Moré ;
- graphiques temps CPU vs dimension ;
- graphiques itérations vs dimension ;
- graphiques coût total (évaluations) vs dimension ;
- étude de cas détaillée pour Rosenbrock 10D/tailles disponibles dans le CSV.

Entrée par défaut : resultats_benchmark.csv
Sortie par défaut : dossier ./figures_benchmark_csv/

Utilisation :
    python visualisations_depuis_csv.py
ou
    python visualisations_depuis_csv.py --csv resultats_benchmark.csv --out figures_benchmark_csv
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paramètres / constantes

METHOD_ORDER = [
    "Cauchy_Point",
    "Dogleg",
    "Steihaug_CG",
    "BFGS_Armijo",
    "BFGS_Wolfe",
    "Newton_BT",
]

METHOD_LABELS = {
    "Cauchy_Point": "Cauchy Point",
    "Dogleg": "Dogleg",
    "Steihaug_CG": "Steihaug-CG",
    "BFGS_Armijo": "BFGS + Armijo",
    "BFGS_Wolfe": "BFGS + Wolfe",
    "Newton_BT": "Newton + Backtracking",
}

METHOD_COLORS = {
    "Cauchy_Point": "tab:blue",
    "Dogleg": "tab:orange",
    "Steihaug_CG": "tab:green",
    "BFGS_Armijo": "tab:red",
    "BFGS_Wolfe": "tab:purple",
    "Newton_BT": "tab:brown",
}

CATEGORY_ORDER = ["A", "B", "C"]

# Utilitaires de chargement / nettoyage


def charger_resultats(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalisation basique
    if "converged" in df.columns and df["converged"].dtype == object:
        df["converged"] = (
            df["converged"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        )

    for col in ["n", "n_iter", "n_f_evals", "n_g_evals", "n_h_evals", "n_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["cpu_s", "f_final", "f_error", "grad_norm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "status" not in df.columns:
        df["status"] = np.where(df.get("converged", False), "success", "unknown")

    # Ordres catégoriels pour affichage
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    if "category" in df.columns:
        df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)

    return df


def creer_dossier(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sauvegarder_tableau(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


# Tableaux synthétiques


def tableau_global(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method", observed=False)
    out = g.agg(
        runs=("method", "size"),
        succes=("converged", "sum"),
        taux_succes=("converged", "mean"),
        iterations_median=("n_iter", "median"),
        iterations_moy=("n_iter", "mean"),
        eval_f_median=("n_f_evals", "median"),
        eval_g_median=("n_g_evals", "median"),
        eval_total_median=("n_total", "median"),
        cpu_median_s=("cpu_s", "median"),
        cpu_moy_s=("cpu_s", "mean"),
        grad_norm_median=("grad_norm", "median"),
        exceptions=("status", lambda s: int((s == "exception").sum())),
        non_converges=("status", lambda s: int((s == "max_iter_or_not_converged").sum())),
    ).reset_index()
    out["taux_succes"] = 100.0 * out["taux_succes"]
    return out.sort_values("method")


def tableau_par_categorie(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["category", "method"], observed=False)
    out = g.agg(
        runs=("method", "size"),
        succes=("converged", "sum"),
        taux_succes=("converged", "mean"),
        iterations_median=("n_iter", "median"),
        cpu_median_s=("cpu_s", "median"),
        eval_total_median=("n_total", "median"),
        grad_norm_median=("grad_norm", "median"),
    ).reset_index()
    out["taux_succes"] = 100.0 * out["taux_succes"]
    return out.sort_values(["category", "method"])



# Graphiques utilitaires


def _save_show(fig, path: Path, show: bool) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def barplot_taux_succes_global(df: pd.DataFrame, outdir: Path, show: bool = False):
    t = tableau_global(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [METHOD_COLORS.get(m, None) for m in t["method"]]
    ax.bar([METHOD_LABELS.get(m, m) for m in t["method"]], t["taux_succes"], color=colors)
    ax.set_title("Taux de convergence global par méthode")
    ax.set_xlabel("Méthode")
    ax.set_ylabel("Taux de succès (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y")
    plt.xticks(rotation=20, ha="right")
    _save_show(fig, outdir / "taux_succes_global.png", show)


def barplot_taux_succes_par_categorie(df: pd.DataFrame, outdir: Path, show: bool = False):
    t = tableau_par_categorie(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, cat in zip(axes, CATEGORY_ORDER):
        sub = t[t["category"] == cat]
        ax.bar([METHOD_LABELS.get(m, m) for m in sub["method"]], sub["taux_succes"],
               color=[METHOD_COLORS.get(m, None) for m in sub["method"]])
        ax.set_title(f"Catégorie {cat}")
        ax.set_xlabel("Méthode")
        ax.grid(True, axis="y")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Taux de succès (%)")
    fig.suptitle("Robustesse par catégorie et par méthode")
    _save_show(fig, outdir / "taux_succes_par_categorie.png", show)


def _lineplot_vs_dimension(df: pd.DataFrame, outdir: Path, ycol: str, ylabel: str,
                           filename: str, categories: Optional[Iterable[str]] = None,
                           seulement_convergents: bool = True, show: bool = False):
    sub = df.copy()
    if categories is not None:
        sub = sub[sub["category"].isin(categories)]
    if seulement_convergents:
        sub = sub[sub["converged"] == True]

    # Agrégation : médiane par méthode, catégorie, dimension
    agg = (
        sub.groupby(["category", "method", "n"], observed=False)[ycol]
        .median()
        .reset_index()
        .dropna()
    )

    cats = CATEGORY_ORDER if categories is None else list(categories)
    fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 4.8), sharey=False)
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        part = agg[agg["category"] == cat]
        for m in METHOD_ORDER:
            tmp = part[part["method"] == m].sort_values("n")
            if tmp.empty:
                continue
            ax.plot(tmp["n"], tmp[ycol], marker="o", label=METHOD_LABELS.get(m, m), color=METHOD_COLORS.get(m, None))
        ax.set_title(f"Catégorie {cat}")
        ax.set_xlabel("Dimension n")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_xscale("log")
        if ycol in ["cpu_s", "n_iter", "n_total", "n_f_evals", "n_g_evals"]:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    fig.suptitle(ylabel + " vs dimension")
    _save_show(fig, outdir / filename, show)


# Profil de performance de Dolan–Moré


def performance_profile(df: pd.DataFrame, metric: str, outdir: Path, filename: str,
                        categories: Optional[Iterable[str]] = None, show: bool = False):
    sub = df.copy()
    if categories is not None:
        sub = sub[sub["category"].isin(categories)]

    # coût = métrique si convergé, +∞ sinon
    sub = sub.copy()
    sub["score"] = np.where(sub["converged"] == True, sub[metric].astype(float), np.inf)
    pivot = sub.pivot_table(index=["problem", "n"], columns="method", values="score", aggfunc="min")
    pivot = pivot.reindex(columns=METHOD_ORDER)
    best = pivot.min(axis=1)
    ratios = pivot.div(best, axis=0)

    finite_vals = ratios.to_numpy().ravel()
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    tau_max = max(5.0, float(np.nanmax(finite_vals)) if finite_vals.size else 5.0)
    taus = np.linspace(1.0, tau_max, 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHOD_ORDER:
        if m not in ratios.columns:
            continue
        r = ratios[m].to_numpy(dtype=float)
        prof = [np.mean(r <= tau) for tau in taus]
        ax.plot(taus, prof, linewidth=2, label=METHOD_LABELS.get(m, m), color=METHOD_COLORS.get(m, None))

    ax.set_title(f"Profil de performance de Dolan–Moré ({metric})")
    ax.set_xlabel("τ")
    ax.set_ylabel("Proportion de problèmes résolus")
    ax.set_xlim(left=1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True)
    ax.legend(fontsize=8)
    _save_show(fig, outdir / filename, show)


=
# Étude de cas recommandée (Rosenbrock)


def figure_etude_de_cas(df: pd.DataFrame, outdir: Path, show: bool = False):
    """
    Le cahier de charge recommande Rosenbrock 10D comme cas d’étude.
    Dans le CSV, on récupère en priorité :
    - A05 (Rosenbrock étendu)
    - B1_Rosenbrock
    - A12 (Rosenbrock α=500)
    et on produit un graphique comparatif sur toutes les tailles disponibles.
    """
    candidats = ["A05", "B1_Rosenbrock", "A12"]
    sub = df[df["problem"].astype(str).isin(candidats)].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("n_iter", "Itérations", True),
        ("cpu_s", "Temps CPU (s)", True),
        ("grad_norm", "Norme finale du gradient", True),
    ]

    for ax, (col, titre, logy) in zip(axes, metrics):
        for m in METHOD_ORDER:
            tmp = sub[sub["method"] == m].sort_values("n")
            if tmp.empty:
                continue
            # médiane sur les problèmes Rosenbrock-like disponibles
            med = tmp.groupby("n", observed=False)[col].median().reset_index()
            ax.plot(med["n"], med[col], marker="o", label=METHOD_LABELS.get(m, m), color=METHOD_COLORS.get(m, None))
        ax.set_title(titre + " — Cas d’étude Rosenbrock")
        ax.set_xlabel("Dimension n")
        ax.grid(True)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel("Valeur")
    axes[2].legend(fontsize=8)
    _save_show(fig, outdir / "etude_de_cas_rosenbrock.png", show)



# Rapport texte automatique


def rapport_texte(df: pd.DataFrame, outdir: Path) -> None:
    glob = tableau_global(df)
    cat = tableau_par_categorie(df)

    lignes = []
    lignes.append("ANALYSE AUTOMATIQUE DU BENCHMARK\n")
    lignes.append("Résumé global par méthode :\n")
    lignes.append(glob.to_string(index=False))
    lignes.append("\n\nRésumé par catégorie × méthode :\n")
    lignes.append(cat.to_string(index=False))

    # Détermination d'un "gagnant" simple par catégorie selon le taux de succès puis CPU médian
    for c in CATEGORY_ORDER:
        sub = cat[cat["category"] == c].sort_values(["taux_succes", "cpu_median_s"], ascending=[False, True])
        if not sub.empty:
            top = sub.iloc[0]
            lignes.append(
                f"\nCatégorie {c} : méthode la plus robuste = {top['method']} "
                f"(taux de succès = {top['taux_succes']:.1f} %, CPU médian = {top['cpu_median_s']:.4g} s)."
            )

    # Commentaire sur les méthodes line-search vs trust-region
    trust = glob[glob["method"].isin(["Cauchy_Point", "Dogleg", "Steihaug_CG"])]
    line = glob[glob["method"].isin(["BFGS_Armijo", "BFGS_Wolfe", "Newton_BT"])]
    if not trust.empty and not line.empty:
        lignes.append("\nSynthèse familles algorithmiques :")
        lignes.append(
            f"Trust-region : taux de succès moyen = {trust['taux_succes'].mean():.1f} %."
        )
        lignes.append(
            f"Recherche linéaire / Newton : taux de succès moyen = {line['taux_succes'].mean():.1f} %."
        )

    path = outdir / "rapport_automatique.txt"
    path.write_text("\n".join(lignes), encoding="utf-8")


# Pipeline principal

def main():
    parser = argparse.ArgumentParser(description="Visualisations benchmark depuis resultats_benchmark.csv")
    parser.add_argument("--csv", type=str, default="resultats_benchmark.csv", help="Chemin vers le fichier CSV")
    parser.add_argument("--out", type=str, default="figures_benchmark_csv", help="Dossier de sortie")
    parser.add_argument("--show", action="store_true", help="Afficher les figures à l'écran")
    args = parser.parse_args()

    df = charger_resultats(args.csv)
    outdir = creer_dossier(args.out)

    # Tables
    sauvegarder_tableau(tableau_global(df), outdir / "tableau_global.csv")
    sauvegarder_tableau(tableau_par_categorie(df), outdir / "tableau_par_categorie.csv")

    # Barplots robustesse
    barplot_taux_succes_global(df, outdir, show=args.show)
    barplot_taux_succes_par_categorie(df, outdir, show=args.show)

    # Graphiques dimension → performance
    _lineplot_vs_dimension(df, outdir, "cpu_s", "Temps CPU médian (s)", "cpu_vs_dimension.png", show=args.show)
    _lineplot_vs_dimension(df, outdir, "n_iter", "Itérations médianes", "iterations_vs_dimension.png", show=args.show)
    _lineplot_vs_dimension(df, outdir, "n_total", "Coût total médian (évaluations)", "cout_total_vs_dimension.png", show=args.show)

    # Profils de performance demandés par le cahier de charge
    performance_profile(df, "cpu_s", outdir, "profil_performance_cpu.png", show=args.show)
    performance_profile(df, "n_iter", outdir, "profil_performance_iterations.png", show=args.show)
    performance_profile(df, "n_total", outdir, "profil_performance_cout_total.png", show=args.show)

    # Étude de cas Rosenbrock-like
    figure_etude_de_cas(df, outdir, show=args.show)

    # Rapport texte automatique
    rapport_texte(df, outdir)

    print("Visualisations générées dans :", outdir.resolve())
    print("Fichiers principaux :")
    for name in [
        "tableau_global.csv",
        "tableau_par_categorie.csv",
        "taux_succes_global.png",
        "taux_succes_par_categorie.png",
        "cpu_vs_dimension.png",
        "iterations_vs_dimension.png",
        "cout_total_vs_dimension.png",
        "profil_performance_cpu.png",
        "profil_performance_iterations.png",
        "profil_performance_cout_total.png",
        "etude_de_cas_rosenbrock.png",
        "rapport_automatique.txt",
    ]:
        p = outdir / name
        if p.exists():
            print(" -", p.name)


if __name__ == "__main__":
    main()
