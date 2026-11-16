"""
Jeu de données Deezer (chansons connues) pour le CNN.

- make_split()   : sépare les chansons collectées en entraînement / validation / test (80 / 10 / 10),
                   stratifié par genre et GROUPÉ PAR ARTISTE (un artiste n'apparaît que d'un côté).
                   Résultat sauvegardé dans Data/split_deezer.csv.
- DeezerDataset  : découpe chaque extrait de 30 s en segments de 3 s et calcule le mel-spectrogramme
                   (ou les MFCC) de chaque segment. Le calcul n'est fait qu'une fois : les
                   caractéristiques sont mises en cache dans Data/cache_deezer/.

Seules les lignes « ok » de Data/collecte_log.csv sont utilisées (les « exclu » sont ignorées).
"""

import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

LOG_FILE = "Data/collecte_log.csv"
SPLIT_FILE = "Data/split_deezer.csv"
CACHE_DIR = "Data/cache_deezer"

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS     # 66 150 échantillons
MAX_SEGMENTS = 10                                   # 10 x 3 s = 30 s
N_FFT = 1024
HOP_LENGTH = 512                                    # 3 s -> 130 trames
N_MELS = 96                                         # 96 bandes mel, comme Choi et al. (2016)


# --------------------------------------------------------------------------- séparation

def make_split(log_file=LOG_FILE, out_file=SPLIT_FILE, seed=42):
    """Séparation 80 / 10 / 10 stratifiée par genre et groupée par artiste."""
    log = pd.read_csv(log_file)
    songs = log[log["statut"] == "ok"].reset_index(drop=True)
    genres = sorted(songs["genre"].unique())
    songs["label"] = songs["genre"].map({g: i for i, g in enumerate(genres)})

    # 10 plis équilibrés en genres, sans artiste partagé entre deux plis :
    # pli 0 -> test, pli 1 -> validation, plis 2 à 9 -> entraînement
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = np.zeros(len(songs), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(songs, songs["label"], groups=songs["artist"])):
        fold[idx] = k
    songs["split"] = np.select([fold == 0, fold == 1], ["test", "val"], default="train")

    cols = ["track_id", "artist", "title", "genre", "label", "fichier", "split"]
    songs[cols].to_csv(out_file, index=False)
    return songs[cols], genres


def load_split(split_file=SPLIT_FILE):
    if not os.path.exists(split_file):
        make_split(out_file=split_file)
    songs = pd.read_csv(split_file)
    genres = (songs[["label", "genre"]].drop_duplicates().sort_values("label")["genre"].tolist())
    return songs, genres


# --------------------------------------------------------------------------- audio

def make_transform(feature):
    if feature == "mel":
        return torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS),
            torchaudio.transforms.AmplitudeToDB(top_db=80),
        )
    if feature == "mfcc":
        return torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE, n_mfcc=20, log_mels=True,
            melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": 64},
        )
    raise ValueError(f"caractéristique inconnue : {feature} (mel ou mfcc)")


def load_audio(path):
    """MP3 -> tenseur mono (1, échantillons) à 22 050 Hz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)    # (échantillons, canaux)
    signal = torch.from_numpy(data.T).mean(dim=0, keepdim=True)   # stéréo -> mono
    if sr != SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, SAMPLE_RATE)
    return signal


def split_segments(signal):
    """Découpe en segments de 3 s ; le dernier segment est gardé s'il contient au moins 1,5 s
    d'audio (complété par du silence). Retourne une liste de tenseurs (1, 66150)."""
    n = signal.shape[1]
    n_seg = min(MAX_SEGMENTS, max(1, (n + SEGMENT_SAMPLES // 2) // SEGMENT_SAMPLES))
    segments = []
    for s in range(n_seg):
        seg = signal[:, s * SEGMENT_SAMPLES:(s + 1) * SEGMENT_SAMPLES]
        if seg.shape[1] < SEGMENT_SAMPLES:
            seg = torch.nn.functional.pad(seg, (0, SEGMENT_SAMPLES - seg.shape[1]))
        segments.append(seg)
    return segments


# --------------------------------------------------------------------------- dataset

class DeezerDataset(Dataset):
    """Un exemple = un segment de 3 s. Attributs utiles pour le vote par chanson :
    song_ids (chanson de chaque segment) et song_labels (genre de chaque chanson)."""

    def __init__(self, split, feature="mel", split_file=SPLIT_FILE, cache_dir=CACHE_DIR):
        songs, self.genres = load_split(split_file)
        songs = songs[songs["split"] == split].reset_index(drop=True)
        self.songs = songs
        self.feature = feature
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"{feature}_{split}.pt")

        if os.path.exists(cache):
            data = torch.load(cache)
            if data["track_ids"] == songs["track_id"].tolist():   # même séparation qu'avant
                self._set(data)
                print(f"{split}: {len(songs)} chansons, {len(self.labels)} segments (cache)")
                return
            print(f"{split}: la séparation a changé, recalcul du cache")

        transform = make_transform(feature)
        features, labels, song_ids = [], [], []
        for i, row in songs.iterrows():
            path = os.path.normpath(str(row["fichier"]).replace("\\", "/"))
            signal = load_audio(path)
            with torch.no_grad():
                for seg in split_segments(signal):
                    # float16 : divise par deux la mémoire et la taille du cache
                    features.append(transform(seg).to(torch.float16))
                    labels.append(int(row["label"]))
                    song_ids.append(i)
            if (i + 1) % 100 == 0 or i + 1 == len(songs):
                print(f"  {split}: {i + 1}/{len(songs)} chansons prétraitées")

        data = {
            "track_ids": songs["track_id"].tolist(),
            "features": torch.stack(features),               # (segments, 1, bandes, 130)
            "labels": torch.tensor(labels),
            "song_ids": torch.tensor(song_ids),
            "song_labels": torch.tensor(songs["label"].tolist()),
        }
        torch.save(data, cache)
        self._set(data)
        print(f"{split}: {len(songs)} chansons, {len(self.labels)} segments")

    def _set(self, data):
        self.features = data["features"]
        self.labels = data["labels"]
        self.song_ids = data["song_ids"]
        self.song_labels = data["song_labels"]
        self.mean, self.std = 0.0, 1.0
        # SpecAugment (Park et al., 2019) : à activer SEULEMENT pour l'entraînement
        self.augment = False
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)   # jusqu'à 12 bandes mel
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)        # jusqu'à 20 trames (~0,5 s)

    def normalize(self, mean, std):
        """Normalisation avec la moyenne et l'écart-type calculés sur l'ENTRAÎNEMENT."""
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = (self.features[index].float() - self.mean) / self.std
        if self.augment:
            # masques tirés au hasard à chaque lecture ; la zone masquée vaut 0,
            # c'est-à-dire la moyenne puisque les données sont normalisées
            x = self.time_mask(self.freq_mask(x))
        return x, self.labels[index]


if __name__ == "__main__":
    songs, genres = make_split()
    print(songs.groupby(["split", "genre"]).size().unstack(0))
    shared = (set(songs[songs.split == "train"].artist) & set(songs[songs.split != "train"].artist))
    print("Artistes partagés entre entraînement et validation/test :", len(shared))
