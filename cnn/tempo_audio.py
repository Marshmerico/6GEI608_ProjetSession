"""
Tempo (BPM) calculé directement à partir de l'audio, par détection des temps (librosa).

Le CNN prédit mal le tempo (R² ≈ 0,07) : un tempo se mesure mieux par traitement du signal
(détection des attaques puis autocorrélation) que par apprentissage.

    python tempo_audio.py

Entrée  : Data/split_deezer.csv (chansons valides et chemin de leur MP3)
Sortie  : Data/resultats/tempo_audio.csv   (track_id, tempo_audio)
Le calcul peut être interrompu et relancé : les chansons déjà traitées sont gardées.
"""

import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SPLIT_FILE = "Data/split_deezer.csv"
OUT_FILE = "Data/resultats/tempo_audio.csv"


def estimate_tempo(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    y = data.mean(axis=1)                                   # stéréo -> mono
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])                   # librosa >= 0.10 renvoie un tableau


def main():
    songs = pd.read_csv(SPLIT_FILE)
    done = {}
    if os.path.exists(OUT_FILE):
        done = dict(pd.read_csv(OUT_FILE).itertuples(index=False, name=None))
    todo = songs[~songs["track_id"].isin(done)]
    print(f"{len(songs)} chansons, {len(done)} déjà calculées, {len(todo)} à faire")

    for n, row in enumerate(todo.itertuples(), 1):
        path = os.path.normpath(str(row.fichier).replace("\\", "/"))
        try:
            done[row.track_id] = estimate_tempo(path)
        except Exception as exc:                            # fichier illisible : on continue
            print(f"  erreur {row.track_id} ({exc})")
        if n % 100 == 0 or n == len(todo):
            pd.DataFrame(list(done.items()), columns=["track_id", "tempo_audio"]).to_csv(
                OUT_FILE, index=False, float_format="%.2f")
            print(f"  {n}/{len(todo)}")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    main()
