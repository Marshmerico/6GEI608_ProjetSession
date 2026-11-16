# Volet CNN — classification de genre sur spectrogrammes

Toutes les commandes se lancent depuis ce dossier (`cnn/`).

## Installation

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Contenu

| Fichier | Rôle |
|---|---|
| `collecte_deezer.py` | Télécharge les extraits Deezer de 30 s des chansons de `Data/chansons.csv` (copie de `data/playlist_2_curated.csv`) |
| `deezer_dataset.py` | Séparation 80/10/10 par artiste (`Data/split_deezer.csv`), segments de 3 s, mel-spectrogrammes (96 bandes) |
| `cnn.py` | Réseau CNN 2D (4 blocs conv, BatchNorm et Dropout en option) |
| `cnn_train_deezer.py` | Entraînement, validation, arrêt anticipé, test, matrice de confusion (options : `--feature mfcc`, `--no-batchnorm`, `--dropout`, `--weight-decay`, `--augment`) |
| `resume_resultats.py` | Tableau récapitulatif de toutes les expériences (`Data/resultats/resume.csv`) |
| `export_embeddings.py` | Embeddings du CNN : 128 valeurs par chanson (centrées, norme 1) |
| `tempo_audio.py` | Tempo mesuré sur le signal (détection des temps, librosa) |
| `regression_features.py` | PMC : embeddings -> 9 features Spotify, pour le k-NN de recommandation |
| `cnn_train.py`, `gtzan_dataset.py`, `inference.py` | Prototype sur GTZAN (données GTZAN non incluses) |
| `Data/collecte_log.csv` | Résultat de la collecte : 1 812 extraits valides (`ok`), 180 introuvables, 8 exclus |
| `Data/split_deezer.csv` | Séparation commune entraînement / validation / test (1 449 / 181 / 182 chansons, aucun artiste partagé) |
| `Data/resultats/` | Poids du meilleur modèle, résultats (json) et matrice de confusion |

## Utilisation

```
python collecte_deezer.py          # ~15-20 min ; les MP3 vont dans Data/audio_deezer/ (non versionné)
python deezer_dataset.py           # (re)crée la séparation et l'affiche
python cnn_train_deezer.py         # mel-spectrogrammes ; options : --feature mfcc, --no-batchnorm, --dropout
python export_embeddings.py        # embeddings du meilleur modèle
python tempo_audio.py              # ~15 min, une seule fois
python regression_features.py      # 9 features prédites -> Data/resultats/features_predites_<modèle>.csv
```

Pour le k-NN : utiliser `features_predites_<modèle>.csv` **sans la colonne liveness** (imprévisible depuis
un extrait) ; le tempo y est mesuré sur le signal. Comparer les tempos après repliement dans [70, 140[ BPM
(`fold_tempo`), car les détecteurs se trompent souvent d'un facteur 2.

L'audio (`Data/audio_deezer/`) et le cache des caractéristiques (`Data/cache_deezer/`) ne sont pas
dans le dépôt : ils se régénèrent avec les commandes ci-dessus.

## Résultats (test, 182 chansons)

| Modèle | Par segment (3 s) | Par chanson (vote) |
|---|---|---|
| CNN, mel 96 bandes, BatchNorm, dropout 0,3 | 40,3 % | 52,7 % |

Recommandation (precision@10 de genre, chansons de test) : embeddings CNN 37,5 %, features Spotify 28,0 %, hasard 10 %.

Prédiction des features (R², test) : loudness 0,88, energy 0,83, acousticness 0,77, instrumentalness 0,64,
danceability 0,63, valence 0,53, speechiness 0,31, tempo 0,07 (PMC ; remplacé par le tempo mesuré), liveness ≈ 0.

## Sources

- Structure de départ de `cnn.py`, `gtzan_dataset.py`, `cnn_train.py` et `inference.py` :
  EnrcDamn, *MusicGenreClassifier-PyTorch*, https://github.com/EnrcDamn/MusicGenreClassifier-PyTorch
  (code modifié : CPU, softmax retiré, padding, segments de 3 s, validation, BatchNorm/Dropout).
- K. Choi, G. Fazekas, M. Sandler, « Explaining deep convolutional neural networks on music classification », arXiv:1607.02444, 2016.
- M. Won et al., « Evaluation of CNN-based Automatic Music Tagging Models », arXiv:2006.00751, 2020.
