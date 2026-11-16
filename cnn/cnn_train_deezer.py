"""
Entraînement du CNN sur les extraits Deezer (chansons connues).

    python cnn_train_deezer.py                       # mel-spectrogrammes (par défaut)
    python cnn_train_deezer.py --feature mfcc        # MFCC, pour comparer
    python cnn_train_deezer.py --no-batchnorm --dropout 0   # modèle d'origine, pour comparer

Premier lancement : les MP3 sont décodés et les caractéristiques mises en cache
(quelques minutes) ; les lancements suivants partent du cache.

Sorties (dossier Data/resultats/) :
    cnn_deezer_<nom>.pth        poids du meilleur modèle (sur la validation)
    cnn_deezer_<nom>.json       réglages, genres, normalisation et résultats de test
    confusion_<nom>.png         matrice de confusion par chanson sur le test
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")                     # pas de fenêtre : on enregistre l'image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn import CNNNetwork
from cnn_train import train_one_epoch, evaluate
from deezer_dataset import DeezerDataset

RESULTS_DIR = "Data/resultats"


def song_predictions(model, data_loader, device):
    """Genre prédit par chanson : moyenne des probabilités de ses segments."""
    model.eval()
    probs = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            probs.append(torch.softmax(model(inputs.to(device)), dim=1).cpu())
    probs = torch.cat(probs)
    ds = data_loader.dataset
    song_probs = torch.zeros(len(ds.song_labels), probs.shape[1]).index_add_(0, ds.song_ids, probs)
    return song_probs.argmax(dim=1), ds.song_labels


def confusion_matrix(pred, true, n_classes):
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(true.tolist(), pred.tolist()):
        cm[t, p] += 1
    return cm


def save_confusion(cm, genres, title, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(genres)), genres, rotation=45, ha="right")
    ax.set_yticks(range(len(genres)), genres)
    ax.set_xlabel("Genre prédit")
    ax.set_ylabel("Vrai genre")
    ax.set_title(title)
    vmax = cm.max().item()
    for i in range(len(genres)):
        for j in range(len(genres)):
            v = cm[i, j].item()
            if v:
                ax.text(j, i, v, ha="center", va="center",
                        color="white" if v > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", choices=["mel", "mfcc"], default="mel")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-batchnorm", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.0, help="pénalité L2 sur les poids (ex. 1e-4)")
    p.add_argument("--augment", action="store_true", help="SpecAugment sur l'entraînement")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = (f"{args.feature}{'_nobn' if args.no_batchnorm else ''}_do{args.dropout}"
            f"{f'_wd{args.weight_decay:g}' if args.weight_decay else ''}{'_aug' if args.augment else ''}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using {device} device | expérience : {name}")

    train_ds = DeezerDataset("train", args.feature)
    val_ds = DeezerDataset("val", args.feature)
    test_ds = DeezerDataset("test", args.feature)
    genres = train_ds.genres

    # normalisation avec les statistiques de l'ENTRAÎNEMENT seulement
    # (calcul par blocs pour ne pas dupliquer tout le jeu en mémoire)
    total, total_sq, count = 0.0, 0.0, 0
    for chunk in train_ds.features.split(1024):
        c = chunk.double()
        total += c.sum().item()
        total_sq += (c * c).sum().item()
        count += c.numel()
    mean = total / count
    std = (total_sq / count - mean ** 2) ** 0.5
    for ds in (train_ds, val_ds, test_ds):
        ds.normalize(mean, std)
    train_ds.augment = args.augment            # jamais sur la validation ni le test

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)       # pas de mélange : vote par chanson
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    input_shape = tuple(train_ds.features.shape[1:])
    model = CNNNetwork(input_shape=input_shape, num_classes=len(genres),
                       batch_norm=not args.no_batchnorm, dropout=args.dropout).to(device)
    print(f"Entrée {input_shape}, {sum(p.numel() for p in model.parameters()):,} paramètres")

    # poids par classe : compense le léger déséquilibre (ex. 121 chansons classiques vs 155 latines)
    counts = torch.bincount(train_ds.song_labels, minlength=len(genres)).float()
    weights = counts.sum() / (len(genres) * counts)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    weights_path = os.path.join(RESULTS_DIR, f"cnn_deezer_{name}.pth")
    best_acc, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        seg_acc, song_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss, "val_segments": seg_acc, "val_chansons": song_acc})
        marker = ""
        if song_acc > best_acc:
            best_acc, best_epoch = song_acc, epoch
            torch.save(model.state_dict(), weights_path)
            marker = "  <- meilleur, sauvegardé"
        print(f"Epoch {epoch:2d} | perte {loss:.3f} | validation : segments {seg_acc:.1%}, "
              f"chansons {song_acc:.1%}{marker}")
        if epoch - best_epoch >= args.patience:
            print(f"Arrêt anticipé : aucune amélioration depuis l'époque {best_epoch}")
            break

    # évaluation finale, une seule fois, sur le test avec le meilleur modèle
    model.load_state_dict(torch.load(weights_path, map_location=device))
    seg_acc, song_acc = evaluate(model, test_loader, device)
    pred, true = song_predictions(model, test_loader, device)
    cm = confusion_matrix(pred, true, len(genres))
    print(f"\nTEST (meilleure époque {best_epoch}) | segments {seg_acc:.1%}, chansons {song_acc:.1%}")
    print("Précision par genre (chansons) :")
    for i, g in enumerate(genres):
        n = cm[i].sum().item()
        print(f"  {g:26s} {cm[i, i].item():3d}/{n:3d}  ({cm[i, i].item() / n:.0%})")

    cm_path = os.path.join(RESULTS_DIR, f"confusion_{name}.png")
    save_confusion(cm, genres, f"CNN ({name}) - test, par chanson : {song_acc:.1%}", cm_path)

    with open(os.path.join(RESULTS_DIR, f"cnn_deezer_{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"reglages": vars(args), "genres": genres, "input_shape": input_shape,
                   "normalisation": {"mean": mean, "std": std}, "meilleure_epoque": best_epoch,
                   "test": {"segments": seg_acc, "chansons": song_acc},
                   "matrice_confusion": cm.tolist(), "historique": history},
                  f, ensure_ascii=False, indent=2)
    print(f"\nRésultats : {RESULTS_DIR}/ (poids, json, {os.path.basename(cm_path)})")


if __name__ == "__main__":
    main()
