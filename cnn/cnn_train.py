import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader
from gtzan_dataset import GTZANDataset
from cnn import CNNNetwork
import pandas as pd
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 8          # arrêt si la validation ne s'améliore pas pendant 8 époques
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "Data/features_30_sec_final.csv"
AUDIO_DIR = "Data/genres_original"
SAMPLE_RATE = 22050
NUM_SAMPLES = 66150 # -> 3 secondes d'audio par segment
NUM_SEGMENTS = 10    # 10 segments de 3 s par extrait de 30 s
INPUT_SHAPE = (1, 20, 130)   # (canaux, coefficients MFCC, trames) pour 3 s avec hop_length=512

def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_data_loader


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Précision par segment, et par chanson (vote : moyenne des probabilités des segments).
    Le data_loader ne doit PAS être mélangé (shuffle=False)."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            probs = torch.softmax(model(inputs.to(device)), dim=1).cpu()
            all_probs.append(probs)
            all_targets.append(targets)
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    segment_acc = (probs.argmax(dim=1) == targets).float().mean().item()

    dataset = data_loader.dataset
    n_songs = len(dataset.song_labels)
    song_probs = torch.zeros(n_songs, probs.shape[1]).index_add_(0, dataset.song_ids, probs)
    song_acc = (song_probs.argmax(dim=1) == dataset.song_labels).float().mean().item()
    return segment_acc, song_acc

def train(model, train_loader, val_loader, loss_fn, optimiser, device, epochs, patience):
    """Entraîne en surveillant la validation ; sauvegarde le modèle de la meilleure époque
    (précision par chanson sur la validation) et s'arrête après `patience` époques sans progrès."""
    best_acc, best_epoch = -1.0, 0
    for i in range(epochs):
        loss = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        seg_acc, song_acc = evaluate(model, val_loader, device)
        marker = ""
        if song_acc > best_acc:
            best_acc, best_epoch = song_acc, i + 1
            torch.save(model.state_dict(), "cnn.pth")
            marker = "  <- meilleur, sauvegardé"
        print(f"Epoch {i+1:2d} | perte moyenne: {loss:.3f} | précision validation: segments {seg_acc:.1%}, chansons {song_acc:.1%}{marker}")
        if i + 1 - best_epoch >= patience:
            print(f"Arrêt anticipé : aucune amélioration depuis l'époque {best_epoch}")
            break
    print(f"Meilleure époque : {best_epoch} (validation, chansons {best_acc:.1%})")
    return best_epoch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    annotations = pd.read_csv(ANNOTATIONS_FILE)
    train_df, test_df = train_test_split(
        annotations,
        test_size=0.1,                      # 10 % pour le test
        stratify=annotations.iloc[:, -1],   # même proportion de chaque genre des deux côtés
        random_state=42,                    # même séparation à chaque exécution
    )
    # Validation : 10 % du total, prise dans les 90 % restants -> séparation 80 / 10 / 10
    train_df, val_df = train_test_split(
        train_df,
        test_size=1/9,                      # 1/9 x 90 % = 10 % du total
        stratify=train_df.iloc[:, -1],
        random_state=42,
    )
    train_df.to_csv("Data/train.csv", index=False)
    val_df.to_csv("Data/val.csv", index=False)
    test_df.to_csv("Data/test.csv", index=False)
    print(f"Chansons : {len(train_df)} entraînement, {len(val_df)} validation, {len(test_df)} test")

    # instantiate dataset object and create data loader
    mfcc = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=20,
    log_mels=True,
    melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64},
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    gtzan = GTZANDataset(
        "Data/train.csv",
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        NUM_SEGMENTS
    )

    train_data_loader = create_data_loader(train_data=gtzan, batch_size=BATCH_SIZE)

    gtzan_test = GTZANDataset(
        "Data/test.csv",
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        NUM_SEGMENTS
    )
    test_data_loader = DataLoader(gtzan_test, batch_size=BATCH_SIZE)

    gtzan_val = GTZANDataset(
        "Data/val.csv",
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        NUM_SEGMENTS
    )
    val_data_loader = DataLoader(gtzan_val, batch_size=BATCH_SIZE)

    # build model
    cnn = CNNNetwork(input_shape=INPUT_SHAPE).to(device)

    # instantiate loss funcion and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr = LEARNING_RATE)

    # train model (le meilleur modèle sur la validation est sauvegardé dans cnn.pth)
    train(cnn, train_data_loader, val_data_loader, loss_fn, optimiser, device, EPOCHS, PATIENCE)

    # évaluation finale, UNE seule fois, sur le test avec le meilleur modèle
    cnn.load_state_dict(torch.load("cnn.pth", map_location=device))
    seg_acc, song_acc = evaluate(cnn, test_data_loader, device)
    print(f"TEST (meilleur modèle) | segments {seg_acc:.1%}, chansons {song_acc:.1%}")
    print("Trained model is stored at cnn.pth")
