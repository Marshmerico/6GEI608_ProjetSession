"""Rassemble les résultats de toutes les expériences (Data/resultats/cnn_deezer_*.json)
dans un tableau, pour le rapport. Sortie : Data/resultats/resume.csv"""

import glob
import json
import os

import pandas as pd

rows = []
for path in sorted(glob.glob("Data/resultats/cnn_deezer_*.json")):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    r = d["reglages"]
    rows.append({
        "experience": os.path.basename(path)[len("cnn_deezer_"):-len(".json")],
        "caracteristiques": r["feature"],
        "batchnorm": not r["no_batchnorm"],
        "dropout": r["dropout"],
        "weight_decay": r.get("weight_decay", 0.0),
        "augmentation": r.get("augment", False),
        "meilleure_epoque": d["meilleure_epoque"],
        "epoques_faites": len(d["historique"]),
        "test_segments_%": round(100 * d["test"]["segments"], 1),
        "test_chansons_%": round(100 * d["test"]["chansons"], 1),
    })

df = pd.DataFrame(rows).sort_values("test_chansons_%", ascending=False)
print(df.to_string(index=False))
df.to_csv("Data/resultats/resume.csv", index=False)
print("\n-> Data/resultats/resume.csv")
