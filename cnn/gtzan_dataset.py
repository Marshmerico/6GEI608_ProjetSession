import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf

class GTZANDataset(Dataset):

    def __init__(self,
                annotations_file,
                audio_dir,
                transformation,
                target_sample_rate,
                num_samples,
                device,
                num_segments=10):
        # num_samples = nombre d'échantillons PAR SEGMENT (ex. 66150 = 3 s à 22 050 Hz)
        self.annotations = pd.read_csv(annotations_file)
        # jazz.00054.wav est corrompu dans GTZAN : on l'exclut
        self.annotations = self.annotations[
            self.annotations["filename"] != "jazz.00054.wav"
        ].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.num_segments = num_segments

        # Précalcul : chaque chanson est lue et transformée UNE seule fois,
        # puis découpée en num_segments segments consécutifs.
        # Tous les segments d'une chanson restent dans le même fichier CSV (train ou test).
        features, labels, song_ids = [], [], []
        n_songs = len(self.annotations)
        for song_idx in range(n_songs):
            signal = self._load_signal(song_idx)
            label = int(self._get_audio_sample_label(song_idx))
            for seg in range(num_segments):
                start = seg * num_samples
                segment = signal[:, start:start + num_samples]
                segment = self._right_pad_if_necessary(segment)
                features.append(self.transformation(segment).cpu())
                labels.append(label)
                song_ids.append(song_idx)
            if (song_idx + 1) % 100 == 0 or song_idx + 1 == n_songs:
                print(f"  {song_idx + 1}/{n_songs} chansons prétraitées")

        self.features = torch.stack(features)        # (n_segments_total, 1, n_mfcc, trames)
        self.labels = torch.tensor(labels)           # genre de chaque segment
        self.song_ids = torch.tensor(song_ids)       # chanson d'origine de chaque segment
        self.song_labels = self.labels[::num_segments]   # genre de chaque chanson (pour le vote)

    def __len__(self):
        # nombre de segments, pas de chansons
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def _load_signal(self, song_idx):
        audio_sample_path = self._get_audio_sample_path(song_idx)
        data, sr = sf.read(audio_sample_path, dtype="float32", always_2d=True)
        signal = torch.from_numpy(data.T).to(self.device)   # (canaux, échantillons)
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        return signal

    def _mix_down_if_necessary(self, signal):
        # signal -> tensor (num_channels, samples) -> (2, 32000) -> (1, 32000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sample_rate
            ).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _cut_if_necessary(self, signal):
        # signal -> tensor (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]   # slicing
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) # (append_left, append_right)
            #  [1, 1, 1] (0, 2) -> [1, 1, 1, 0, 0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"{self.annotations.iloc[index, -2]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 1])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, -1]

    def _spectrogram(self, signal):
        stft = torch.stft(signal, n_fft=1024, hop_length=512)
        out_abs = torch.sqrt(stft**2)
        spec = 2 * torch.log(torch.clamp(out_abs, 1e-10, float("inf")))
        return spec

if __name__ == "__main__":
    ANNOTATIONS_FILE = "Data/features_30_sec_final.csv"
    AUDIO_DIR = "Data/genres_original"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 66150 # -> 3 secondes par segment
    plot = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # la séparation train / validation / test est faite dans cnn_train.py



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

    # objects inside transforms module are callable!
    # ms = mel_spectrogram(signal)

    gtzan = GTZANDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    print(f"There are {len(gtzan)} samples in the dataset")

    if plot:
        signal, label = gtzan[0]
        signal = signal.cpu()
        print(signal.shape)
        
        plt.figure(figsize=(16, 8), facecolor="white")
        plt.imshow(signal[0,:,:], origin='lower')
        plt.autoscale(False)
        plt.xlabel("Trame")
        plt.ylabel("Coefficient MFCC")
        plt.colorbar()
        plt.axis('auto')
        plt.show()
