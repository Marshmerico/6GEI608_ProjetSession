"""
Prédiction des 9 features Spotify à partir de l'audio, pour le k-NN de recommandation.

Entrée  : les embeddings du CNN (128 valeurs par chanson, export_embeddings.py)
Modèle  : perceptron multicouche (régression, perte = erreur quadratique moyenne)
Cible   : danceability, energy, loudness, speechiness, acousticness,
          instrumentalness, liveness, valence, tempo   (vraies valeurs : Data/chansons.csv)

Les hyperparamètres du PMC sont choisis sur la VALIDATION ; le TEST n'est utilisé qu'une fois.

    python regression_features.py                    # embeddings Data/resultats/embeddings_mel_do0.3.csv
    python regression_features.py --model mel_do0.5  # autre expérience CNN

Tempo : s'il existe, Data/resultats/tempo_audio.csv (tempo_audio.py, détection des temps)
remplace le tempo prédit par le PMC, qui est peu fiable.
Liveness : prédite mais EXCLUE du k-NN (imprévisible depuis un extrait, peu utile pour recommander).

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
KNN_FEATURES = [f for f in FEATURES if f != "liveness"]      # features recommandées pour le k-NN
TEMPO_FILE = os.path.join(RESULTS_DIR, "tempo_audio.csv")


def load_data(model_name):
    emb = pd.read_csv(os.path.join(RESULTS_DIR, f"embeddings_{model_name}.csv"))
    spotify = pd.read_csv("Data/chansons.csv", index_col=0).drop_duplicates("track_id")
    df = emb.merge(spotify[["track_id"] + FEATURES], on="track_id", how="left")
    assert df[FEATURES].notna().all().all(), "features Spotify manquantes pour certaines chansons"
    emb_cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    return df, emb_cols


def r2_per_feature(y_true, y_pred):
    return {f: r2_score(y_true[:, i], y_pred[:, i]) for i, f in enumerate(FEATURES)}


def tempo_accuracy(true, est):
    """Part des tempos à ±4 % près, et à ±4 % près en acceptant le double ou la moitié
    (erreur d'octave classique : 70 BPM au lieu de 140)."""
    ratio = est / true
    exact = np.abs(ratio - 1) <= 0.04
    octave = exact | (np.abs(ratio - 2) <= 0.08) | (np.abs(ratio - 0.5) <= 0.02)
    return exact.mean(), octave.mean()


def fold_tempo(bpm, low=70.0):
    """Ramène chaque tempo dans [70, 140[ BPM en le doublant ou le divisant par deux.
    Les détecteurs de temps se trompent souvent d'un facteur 2 (erreur d'octave) ;
    une fois replié, 70 et 140 BPM deviennent la même valeur."""
    bpm = np.asarray(bpm, dtype=float)
    safe = np.where(bpm > 0, bpm, low)                  # tempo 0 (morceau sans pulsation) : laissé tel quel
    return np.where(bpm > 0, safe * 2.0 ** (-np.floor(np.log2(safe / low))), bpm)


def knn_eval(Y, pred, cols, tr, te, labels):
    """Requête = features PRÉDITES d'une chanson de test ; catalogue = VRAIES features
    des 1 812 chansons (comme dans la démo). Renvoie p@10 vraies, p@10 prédites, voisins communs."""
    idx = [FEATURES.index(c) for c in cols]
    Y, pred = Y.copy(), pred.copy()
    t_i = FEATURES.index("tempo")
    Y[:, t_i], pred[:, t_i] = fold_tempo(Y[:, t_i]), fold_tempo(pred[:, t_i])   # tempos repliés
    mean, std = Y[tr][:, idx].mean(axis=0), Y[tr][:, idx].std(axis=0)
    catalog = (Y[:, idx] - mean) / std
    query = (pred[:, idx] - mean) / std
    test_idx = np.flatnonzero(te)
    nn_true = neighbours(catalog[test_idx], catalog, exclude=test_idx)
    nn_pred = neighbours(query[test_idx], catalog, exclude=test_idx)
    p10_true = (labels[nn_true] == labels[test_idx][:, None]).mean()
    p10_pred = (labels[nn_pred] == labels[test_idx][:, None]).mean()
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn_true, nn_pred)])
    return p10_true, p10_pred, overlap


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

    # ---- tempo : mesuré sur le signal plutôt que prédit, si tempo_audio.py a été lancé
    t_i = FEATURES.index("tempo")
    tempo_source = "PMC"
    acc_pmc = tempo_accuracy(Y[te, t_i], pred[te, t_i])
    tempo_info = {"pmc_a_4pct": acc_pmc[0], "pmc_a_4pct_ou_octave": acc_pmc[1]}
    print(f"\nTempo prédit par le PMC : {acc_pmc[0]:.0%} à ±4 %, {acc_pmc[1]:.0%} en acceptant double/moitié")
    if os.path.exists(TEMPO_FILE):
        tempo = df[["track_id"]].merge(pd.read_csv(TEMPO_FILE), on="track_id", how="left")["tempo_audio"]
        ok = tempo.notna().to_numpy()
        pred[ok, t_i] = tempo[ok].to_numpy()
        tempo_source = "signal (librosa)"
        r2_sig = r2_score(fold_tempo(Y[te & ok, t_i]), fold_tempo(pred[te & ok, t_i]))   # tempos repliés
        acc_sig = tempo_accuracy(Y[te & ok, t_i], pred[te & ok, t_i])
        tempo_info.update({"signal_r2": r2_sig, "signal_a_4pct": acc_sig[0],
                           "signal_a_4pct_ou_octave": acc_sig[1], "chansons_sans_tempo": int((~ok).sum())})
        print(f"Tempo mesuré sur le signal : R² {r2_sig:.2f}, {acc_sig[0]:.0%} à ±4 %, "
              f"{acc_sig[1]:.0%} en acceptant double/moitié ({(~ok).sum()} chansons sans tempo)")
    else:
        print(f"(lancer tempo_audio.py pour remplacer le tempo prédit par le tempo mesuré)")

    # ---- effet sur la recommandation
    labels = df["label"].to_numpy()
    reco = {}
    print(f"\nRecommandation (k-NN sans le genre, chansons de test ; tempo : {tempo_source}) :")
    print(f"  {'features':24s} {'p@10 vraies':>12s} {'p@10 prédites':>14s} {'voisins communs':>16s}")
    for label, cols in [("les 9", FEATURES), ("8, sans liveness", KNN_FEATURES)]:
        p10_true, p10_pred, overlap = knn_eval(Y, pred, cols, tr, te, labels)
        reco[label] = {"p10_vraies": p10_true, "p10_predites": p10_pred, "voisins_communs": overlap}
        print(f"  {label:24s} {p10_true:12.1%} {p10_pred:14.1%} {overlap:16.1%}")

    # ---- sauvegardes
    out = df[["track_id", "artist", "title", "genre", "split"]].copy()
    out[FEATURES] = pred                 # tempo = mesuré sur le signal s'il est disponible
    out.to_csv(os.path.join(RESULTS_DIR, f"features_predites_{args.model}.csv"),
               index=False, float_format="%.4f")
    with open(os.path.join(RESULTS_DIR, f"regression_features_{args.model}.pkl"), "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "mlp": mlp, "features": FEATURES}, f)
    with open(os.path.join(RESULTS_DIR, f"regression_features_{args.model}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"pmc": {"couches": list(hidden), "alpha": alpha, "r2_validation": r2_val},
                   "test_r2_pmc": r2_mlp, "test_r2_ridge": r2_lin, "test_erreur_moyenne": mae,
                   "tempo": {"source": tempo_source, **tempo_info},
                   "features_knn_recommandees": KNN_FEATURES, "recommandation": reco},
                  f, ensure_ascii=False, indent=2)
    print(f"\n-> {RESULTS_DIR}/features_predites_{args.model}.csv (+ .json, .pkl)")


if __name__ == "__main__":
    main()
