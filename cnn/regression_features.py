"""
Prédiction des 9 features Spotify à partir de l'audio, pour le k-NN de recommandation.

Entrée  : les embeddings du CNN (128 valeurs par chanson, export_embeddings.py)
Modèle  : perceptron multicouche (régression, perte = erreur quadratique moyenne)
Cible   : danceability, energy, loudness, speechiness, acousticness,
          instrumentalness, liveness, valence, tempo   (vraies valeurs : Data/chansons.csv)

Les hyperparamètres du PMC sont choisis sur la VALIDATION ; le TEST n'est utilisé qu'une fois.

    python regression_features.py                    # embeddings Data/resultats/embeddings_mel_do0.3.csv
    python regression_features.py --model mel_do0.5  # autre expérience CNN

Sorties (Data/resultats/) :
    features_predites_<modèle>.csv      9 features prédites pour les 1 812 chansons (+ split)
    regression_features_<modèle>.json   R² et erreur moyenne par feature, réglages retenus
    regression_features_<modèle>.pkl    modèle entraîné (pour prédire une nouvelle chanson)
"""

import argparse
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = "Data/resultats"
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]


def load_data(model_name):
    emb = pd.read_csv(os.path.join(RESULTS_DIR, f"embeddings_{model_name}.csv"))
    spotify = pd.read_csv("Data/chansons.csv", index_col=0).drop_duplicates("track_id")
    df = emb.merge(spotify[["track_id"] + FEATURES], on="track_id", how="left")
    assert df[FEATURES].notna().all().all(), "features Spotify manquantes pour certaines chansons"
    emb_cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    return df, emb_cols


def r2_per_feature(y_true, y_pred):
    return {f: r2_score(y_true[:, i], y_pred[:, i]) for i, f in enumerate(FEATURES)}


def neighbours(Q, C, k=10, exclude=None):
    """k plus proches voisins (cosinus) des requêtes Q dans le catalogue C."""
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    sims = Qn @ Cn.T
    if exclude is not None:
        sims[np.arange(len(Q)), exclude] = -np.inf          # pas la chanson elle-même
    return np.argsort(-sims, axis=1)[:, :k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mel_do0.3", help="expérience CNN dont on prend les embeddings")
    args = p.parse_args()

    df, emb_cols = load_data(args.model)
    split = df["split"].to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"

    X = df[emb_cols].to_numpy()
    Y = df[FEATURES].to_numpy(dtype=float)

    # standardisation des entrées ET des cibles (le tempo ~120 et la danceability ~0,6
    # n'ont pas la même échelle) ; statistiques calculées sur l'entraînement seulement
    x_scaler = StandardScaler().fit(X[tr])
    y_scaler = StandardScaler().fit(Y[tr])
    Xs, Ys = x_scaler.transform(X), y_scaler.transform(Y)

    # ---- références
    ridge = Ridge(alpha=10.0).fit(Xs[tr], Ys[tr])
    r2_ridge = r2_score(Ys[va], ridge.predict(Xs[va]))

    # ---- PMC : petite recherche d'hyperparamètres sur la validation
    grid = {"hidden_layer_sizes": [(64,), (128,), (128, 64)], "alpha": [1e-3, 1e-2, 1e-1]}
    best = None
    print("Recherche d'hyperparamètres (R² moyen sur la validation) :")
    for hidden, alpha in itertools.product(grid["hidden_layer_sizes"], grid["alpha"]):
        mlp = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, activation="relu",
                           learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                           validation_fraction=0.1, n_iter_no_change=30, random_state=42)
        mlp.fit(Xs[tr], Ys[tr])
        r2 = r2_score(Ys[va], mlp.predict(Xs[va]))
        print(f"  couches {str(hidden):10s} alpha {alpha:<6g} R² = {r2:.3f}")
        if best is None or r2 > best[0]:
            best = (r2, hidden, alpha, mlp)
    r2_val, hidden, alpha, mlp = best
    print(f"Retenu : couches {hidden}, alpha {alpha} (R² validation {r2_val:.3f}) "
          f"| régression linéaire Ridge : {r2_ridge:.3f}")

    # ---- évaluation finale sur le test
    pred = y_scaler.inverse_transform(mlp.predict(Xs))       # unités d'origine
    pred_ridge = y_scaler.inverse_transform(ridge.predict(Xs))
    r2_mlp = r2_per_feature(Y[te], pred[te])
    r2_lin = r2_per_feature(Y[te], pred_ridge[te])
    mae = {f: mean_absolute_error(Y[te, i], pred[te, i]) for i, f in enumerate(FEATURES)}
    print(f"\nTEST ({te.sum()} chansons) - R² par feature (1 = parfait, 0 = pas mieux que la moyenne)")
    print(f"  {'feature':18s} {'PMC':>6s} {'Ridge':>6s}   erreur moyenne (PMC)")
    for f in FEATURES:
        print(f"  {f:18s} {r2_mlp[f]:6.2f} {r2_lin[f]:6.2f}   {mae[f]:.3f}")
    print(f"  {'moyenne':18s} {np.mean(list(r2_mlp.values())):6.2f} {np.mean(list(r2_lin.values())):6.2f}")

    # ---- effet sur la recommandation : requête = features PRÉDITES d'une chanson de test,
    #      catalogue = VRAIES features des 1 812 chansons (comme dans la démo)
    catalog = (Y - Y[tr].mean(axis=0)) / Y[tr].std(axis=0)
    query_pred = (pred - Y[tr].mean(axis=0)) / Y[tr].std(axis=0)
    test_idx = np.flatnonzero(te)
    labels = df["label"].to_numpy()
    nn_true = neighbours(catalog[test_idx], catalog, exclude=test_idx)
    nn_pred = neighbours(query_pred[test_idx], catalog, exclude=test_idx)
    p10_true = (labels[nn_true] == labels[test_idx][:, None]).mean()
    p10_pred = (labels[nn_pred] == labels[test_idx][:, None]).mean()
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn_true, nn_pred)])
    print(f"\nRecommandation (k-NN, 9 features, sans le genre) - chansons de test :")
    print(f"  precision@10 genre avec les VRAIES features   : {p10_true:.1%}")
    print(f"  precision@10 genre avec les features PRÉDITES : {p10_pred:.1%}")
    print(f"  voisins communs aux deux listes de 10         : {overlap:.1%}")

    # ---- sauvegardes
    out = df[["track_id", "artist", "title", "genre", "split"]].copy()
    out[FEATURES] = pred
    out.to_csv(os.path.join(RESULTS_DIR, f"features_predites_{args.model}.csv"),
               index=False, float_format="%.4f")
    with open(os.path.join(RESULTS_DIR, f"regression_features_{args.model}.pkl"), "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "mlp": mlp, "features": FEATURES}, f)
    with open(os.path.join(RESULTS_DIR, f"regression_features_{args.model}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"pmc": {"couches": list(hidden), "alpha": alpha, "r2_validation": r2_val},
                   "test_r2_pmc": r2_mlp, "test_r2_ridge": r2_lin, "test_erreur_moyenne": mae,
                   "recommandation": {"p10_vraies": p10_true, "p10_predites": p10_pred,
                                      "voisins_communs": overlap}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n-> {RESULTS_DIR}/features_predites_{args.model}.csv (+ .json, .pkl)")


if __name__ == "__main__":
    main()
