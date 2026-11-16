"""
Export des embeddings du CNN : un vecteur de 128 valeurs par chanson, pour la recommandation (k-NN).

    python export_embeddings.py                      # modèle Data/resultats/cnn_deezer_mel_do0.3.*
    python export_embeddings.py --model mel_do0.5    # autre expérience

Sortie : Data/resultats/embeddings_<modèle>.csv
    track_id, artist, title, genre, label, split, e0 ... e127
    Chaque embedding = moyenne des embeddings de ses segments de 3 s, centrée (moyenne de
    l'entraînement soustraite) puis normalisée (norme L2 = 1) :
    le produit scalaire de deux lignes est directement leur similarité cosinus.

Vérification affichée : precision@10 de genre (part des 10 plus proches voisins du même genre)
pour les chansons de test, avec les embeddings du CNN et avec les features Spotify, sur le même catalogue.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cnn import CNNNetwork
from deezer_dataset import DeezerDataset

RESULTS_DIR = "Data/resultats"
SPOTIFY_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                    "instrumentalness", "liveness", "valence", "tempo"]


def song_embeddings(model, dataset, device, batch_size=128):
    """Embedding par segment, puis moyenne des segments de chaque chanson."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=batch_size):     # ordre conservé
            chunks.append(model.embed(x.to(device)).cpu())
    seg = torch.cat(chunks)                                          # (segments, 128)
    n_songs = len(dataset.song_labels)
    sums = torch.zeros(n_songs, seg.shape[1]).index_add_(0, dataset.song_ids, seg)
    counts = torch.bincount(dataset.song_ids, minlength=n_songs).clamp(min=1).unsqueeze(1)
    return sums / counts                                             # (chansons, 128)


def l2_normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)


def neighbours(X, query_idx, k=10):
    """Indices des k plus proches voisins (cosinus) de chaque requête, sans la chanson elle-même."""
    Xn = l2_normalize(X)
    sims = Xn[query_idx] @ Xn.T
    sims[np.arange(len(query_idx)), query_idx] = -np.inf
    top = np.argsort(-sims, axis=1)[:, :k]
    return top, np.take_along_axis(sims, top, axis=1)


def precision_at_k(X, labels, query_idx, k=10):
    top, _ = neighbours(X, query_idx, k)
    return (labels[top] == labels[query_idx][:, None]).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mel_do0.3", help="nom de l'expérience dans Data/resultats")
    p.add_argument("--example", default="Blinding Lights", help="titre pour afficher des voisins")
    args = p.parse_args()

    meta_path = os.path.join(RESULTS_DIR, f"cnn_deezer_{args.model}.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    r = meta["reglages"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # même architecture et mêmes poids qu'à l'entraînement
    model = CNNNetwork(input_shape=tuple(meta["input_shape"]), num_classes=len(meta["genres"]),
                       batch_norm=not r["no_batchnorm"], dropout=r["dropout"]).to(device)
    weights = os.path.join(RESULTS_DIR, f"cnn_deezer_{args.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    frames, embeddings = [], []
    for split in ("train", "val", "test"):
        ds = DeezerDataset(split, r["feature"])
        ds.normalize(meta["normalisation"]["mean"], meta["normalisation"]["std"])   # comme à l'entraînement
        embeddings.append(song_embeddings(model, ds, device))
        df = ds.songs[["track_id", "artist", "title", "genre", "label"]].copy()
        df["split"] = split
        frames.append(df)
    songs = pd.concat(frames, ignore_index=True)
    E = torch.cat(embeddings).numpy()
    # Les embeddings sortent d'un ReLU : toutes leurs valeurs sont >= 0, donc tous les vecteurs
    # pointent dans la même direction générale et les similarités cosinus sont toutes proches de 1.
    # On les centre (moyenne de l'ENTRAÎNEMENT) avant de normaliser pour étaler les similarités.
    train_mean = E[songs["split"].to_numpy() == "train"].mean(axis=0)
    E = l2_normalize(E - train_mean)
    assert len(songs) == len(E)

    out = pd.concat([songs, pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])], axis=1)
    out_path = os.path.join(RESULTS_DIR, f"embeddings_{args.model}.csv")
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n{len(out)} chansons x {E.shape[1]} valeurs -> {out_path}")

    # ---- vérification : precision@10 de genre pour les chansons de test, catalogue complet
    labels = songs["label"].to_numpy()
    test_idx = np.flatnonzero(songs["split"].to_numpy() == "test")
    print(f"\nprecision@10 (genre), {len(test_idx)} chansons de test interrogées :")
    print(f"  embeddings CNN    : {precision_at_k(E, labels, test_idx):.1%}")

    spotify = pd.read_csv("Data/chansons.csv", index_col=0).drop_duplicates("track_id")
    feats = songs[["track_id"]].merge(spotify[["track_id"] + SPOTIFY_FEATURES], on="track_id", how="left")
    if feats[SPOTIFY_FEATURES].notna().all().all():
        S = feats[SPOTIFY_FEATURES].to_numpy(dtype=float)
        S = (S - S.mean(axis=0)) / S.std(axis=0)                     # standardisation
        print(f"  features Spotify  : {precision_at_k(S, labels, test_idx):.1%}")
    print(f"  hasard (≈ 1/{len(meta['genres'])}) : {1 / len(meta['genres']):.1%}")

    # ---- exemple de voisins
    match = np.flatnonzero(songs["title"].str.lower().to_numpy() == args.example.lower())
    q = int(match[0]) if len(match) else int(test_idx[0])
    top, sims = neighbours(E, np.array([q]), k=10)
    print(f"\n10 plus proches voisins de : {songs.title[q]} - {songs.artist[q]} ({songs.genre[q]})")
    for rank, (j, s) in enumerate(zip(top[0], sims[0]), 1):
        print(f"  {rank:2d}. {s:.3f}  {songs.title[j][:40]:40s} {songs.artist[j][:22]:22s} {songs.genre[j]}")


if __name__ == "__main__":
    main()
