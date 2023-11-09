import os
import glob
import librosa
import torch as th
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from utils import LogMelSpectrogram


class GTZANDataset(Dataset):
    FRAME_UNIT = 0.01
    LOGMEL_PARAMS = {
        "sample_rate": 44100,
        "n_fft": 2048,
        "hop_length": 441,
        "n_mels": 128,
        "f_min": 30,
        "f_max": 17000,
        "norm": None,
        "is_log": True,
        "eps": 1e-6,
        "center": True
    }

    def __init__(
            self, path=None, beat_type="beats", mode="train", n_frames=512):

        assert beat_type in ["tempo", "beats", "beats+tempo"]
        self.beat_type = beat_type
        self.mode = mode
        self.n_frames = n_frames
        self.to_logmel = LogMelSpectrogram(**self.LOGMEL_PARAMS)
        
        # Get audio filelist
        self.tracks = glob.glob(f'{path}/*/*.wav')

    
    def __getitem__(self, idx):

        sample = {}
        
        audio_path = self.tracks[idx]
        print(audio_path)
        tempo_path = f"./gtzan/annotations/tempo/gtzan_{os.path.basename(audio_path).split('.')[0]}_{os.path.basename(audio_path).split('.')[1]}.bpm"
        beat_path = f"./gtzan/annotations/beats/gtzan_{os.path.basename(audio_path).split('.')[0]}_{os.path.basename(audio_path).split('.')[1]}.beats"

        feat_path = audio_path.replace(".wav", "_logmel.pt")
        beat_path_processed = beat_path.replace(".beats", f"_beat_sequence_logmel.pt")

        # Load logmel spectrogram
        if os.path.exists(feat_path):
            feat = th.load(feat_path)
        else:
            wav, samplerate = librosa.load(audio_path, sr=44100, mono=True)
            assert samplerate == 44100 and len(wav.shape) == 1
            feat = self.to_logmel(th.from_numpy(wav).float())[..., :-1].T
            th.save(feat, feat_path)
        
        if self.mode == "train":
            start, frames = np.random.randint(0, feat.shape[0] - self.n_frames), self.n_frames
        else:
            start, frames = 0, feat.shape[0]
            
        # Load Tempo
        if "tempo" in self.beat_type:
            sample["tempo"] = np.loadtxt(tempo_path)

        # Load beat sequence
        if "beats" in self.beat_type:
            if os.path.exists(beat_path_processed):
                beats_sequence = th.load(beat_path_processed)
            else:
                beats = np.loadtxt(beat_path)[:, 0]
                beats_sequence = th.zeros(feat.shape[0], dtype=int)
                beats_indices = np.array(beats) / self.FRAME_UNIT
                beats_indices = np.round(beats_indices, 0).astype(int)
                beats_indices = [b for b in beats_indices if b < feat.shape[0]]
                beats_sequence[beats_indices] = 1
                th.save(beats_sequence, beat_path_processed)
                    
            beats_sequence = beats_sequence[start:start+frames]
            
        sample["beats"] = beats_sequence.float()
        
        feat = feat[start:start+frames]
        sample["audio_features"] = feat


        if self.mode in ["validation"]:
            sample["time_unit"] = self.FRAME_UNIT
            sample["audio_path"] = audio_path

        return sample

    def __len__(self):
        return len(self.tracks)
    
    
class GTZANDataModule(pl.LightningDataModule):
    def __init__(
        self, path, meta_path, beat_type, n_frames,
        batch_size, pin_memory, n_workers
    ):
        super().__init__()
        # Dataset params
        self.path = path
        self.meta_path = meta_path
        self.beat_type = beat_type
        self.n_frames = n_frames

        # Dataloader params
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.n_workers = n_workers

    def setup(self):
        self.train_set = GTZANDataset(
            self.path, self.meta_path, "train",
            self.beat_type, self.n_frames
        )
        self.val_set = GTZANDataset(
            self.path, self.meta_path, "validation",
            self.beat_type, self.n_frames
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            pin_memory=self.pin_memory, num_workers=self.n_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=1, shuffle=False,
            pin_memory=self.pin_memory, num_workers=1
        )