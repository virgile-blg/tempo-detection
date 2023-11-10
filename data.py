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
            self, audio_list=None, beat_type="tempo", tempo_gt="tempo_annotations", mode="train", n_frames=512):

        assert beat_type in ["tempo", "beats", "beats+tempo"]
        assert tempo_gt in ["tempo_annotations", "beat_annotations"]
        if tempo_gt == "beat_annotations" and "beats" not in beat_type:
            beat_type = "beats+tempo"
        self.beat_type = beat_type
        self.tempo_gt = tempo_gt
        self.mode = mode
        self.n_frames = n_frames
        self.to_logmel = LogMelSpectrogram(**self.LOGMEL_PARAMS)
        # Issue with one annotation missing, one corrupted audio, and one tempo > 300 bpm
        self.tracks = [i for i in audio_list if "reggae.00086" not in i and "jazz.00054" not in i and "jazz.00031" not in i]
    
    def __getitem__(self, index):

        sample = {}
        audio_path = self.tracks[index]
        
        tempo_path = f"./gtzan/annotations/tempo/gtzan_{os.path.basename(audio_path).split('.')[0]}_{os.path.basename(audio_path).split('.')[1]}.bpm"
        beat_path = f"./gtzan/annotations/beats/gtzan_{os.path.basename(audio_path).split('.')[0]}_{os.path.basename(audio_path).split('.')[1]}.beats"

        feat_path = audio_path.replace(".wav", "_logmel.pt")
        beat_path_processed = beat_path.replace(".beats", f"_beat_sequence_logmel.pt")

        # Load logmel spectrogram
        if os.path.exists(feat_path):
            feat = th.load(feat_path)
        else:
            try:
                wav, samplerate = librosa.load(audio_path, sr=44100, mono=True)
            except Exception as e:
                print(audio_path)
            assert samplerate == 44100 and len(wav.shape) == 1
            feat = self.to_logmel(th.from_numpy(wav).float())[..., :-1].T
            th.save(feat, feat_path)
        
        # get n_frames for training
        if self.mode == "train":
            start, frames = np.random.randint(0, feat.shape[0] - self.n_frames), self.n_frames
        else:
            start, frames = 0, feat.shape[0]
            
        # Add audio feature to sample
        feat_n_frames = feat[start:start+frames]
        sample["audio_features"] = feat_n_frames
            
        # Add beat sequence to sample
        if "beats" in self.beat_type:
            if os.path.exists(beat_path_processed):
                beats_sequence = th.load(beat_path_processed)
            else:
                beats = np.loadtxt(beat_path)
                if len(beats.shape) > 1:
                    beats = beats[:, 0]
                beats_sequence = th.zeros(feat.shape[0], dtype=int)
                beats_indices = np.array(beats) / self.FRAME_UNIT
                beats_indices = np.round(beats_indices, 0).astype(int)
                beats_indices = [b for b in beats_indices if b < feat.shape[0]]
                beats_sequence[beats_indices] = 1
                th.save(beats_sequence, beat_path_processed)
                    
            beats_sequence = beats_sequence[start:start+frames]
            
            sample["beats"] = beats_sequence.float()
            
        # Add Tempo to sample
        if "tempo" in self.beat_type:
            if self.tempo_gt == "tempo_annotations":
                tempo = np.loadtxt(tempo_path)
                sample["tempo"] = tempo
            else:
                # Derive tempo from beats annotations
                if th.nonzero(sample["beats"]).numel() == 0:
                    sample["tempo"] = -1
                else:
                    mean_beat_frames = th.nonzero(sample["beats"]).squeeze(-1).diff().float().mean()
                    if mean_beat_frames.isnan():
                        sample["tempo"] = -1
                    else:
                        sample["tempo"] = round(60 / (mean_beat_frames.item() * self.FRAME_UNIT))
        
        


        if self.mode in ["validation"]:
            sample["time_unit"] = self.FRAME_UNIT
            sample["audio_path"] = audio_path

        return sample

    def __len__(self):
        return len(self.tracks)
    
    
class GTZANDataModule(pl.LightningDataModule):
    def __init__(
        self, path, beat_type, n_frames,
        batch_size, valid_percent, pin_memory, n_workers
    ):
        super().__init__()
        # Dataset params
        self.path = path
        self.beat_type = beat_type
        self.n_frames = n_frames

        # Dataloader params
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.n_workers = n_workers
        
        # Get audio filelist
        self.tracks = glob.glob(f'{self.path}/*/*.wav')
        self.split = round(valid_percent*len(self.tracks))

    def setup(self, stage=None):
        
        self.train_set = GTZANDataset(
            self.tracks[0:self.split], self.beat_type, "train",
            self.n_frames
        )
        
        self.val_set = GTZANDataset(
            self.tracks[self.split:], self.beat_type, "validation",
            self.n_frames
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