import torch
import madmom
import numpy as np
import torch.nn as nn
import omegaconf as om
import pytorch_lightning as pl
import torch.nn.functional as F

from metrics import tempo_acc_1
from bock_network import BeatBockNet
from utils import BCESequenceLoss, MelSpecAugment, NeighbourBalancingKernel, LogMelSpectrogram

class TempoBeatModel(pl.LightningModule):
    def __init__(self, hparams: om.DictConfig):
        super().__init__()
        self.hparams.update(hparams)
        if not isinstance(hparams, om.DictConfig):
            hparams = om.DictConfig(hparams)
        self.hparams.update(om.OmegaConf.to_container(hparams, resolve=True))
        
        self.model = BeatBockNet(**self.hparams['model'])
        
        if self.hparams['augment']:
            self.augment = MelSpecAugment()
        else:
            self.augment = torch.nn.Identity()
        
        self.tempo_neighbour_smooth = NeighbourBalancingKernel(weights=[0.25, 0.5, 1, 0.5, 0.25])
        self.beats_neighbour_smooth = NeighbourBalancingKernel(weights=[0.5, 1, 0.5])
        
        self.beat_type = self.hparams['data']['beat_type']
        self.beats_loss = BCESequenceLoss()
        self.tempo_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.tempo_lambda = 1

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.model.parameters(),
                                lr=self.hparams.optim["lr"],betas=self.hparams.optim["betas"],weight_decay=float(self.hparams.optim["weight_decay"]))
        
        self.optimizer = optimizer

        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, mode="max", patience=100, min_lr=0.00001)
        self.scheduler = scheduler

        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        loss = 0
        # Get audio & annotations from dataloaders
        audio_features = batch["audio_features"]
        # Augmentation if needed
        audio_features = self.augment(audio_features)
        
        tempo = batch["tempo"]
        # To One-hot
        tempo = F.one_hot(torch.round(tempo).long(), num_classes=300).float()
        # Neighbour smooth
        tempo = self.tempo_neighbour_smooth(tempo)
        
        beats = batch["beats"]
        # Neighbour smooth
        beats = self.beats_neighbour_smooth(beats)
        
        # Forward pass
        tempo_hat, beats_hat = self.forward(audio_features)
        
        # Compute losses
        loss_tempo = self.tempo_loss(tempo_hat, tempo)
        loss_beats = self.beats_loss(beats_hat, beats)
        loss += self.tempo_lambda * loss_tempo + loss_beats
        
        # Log losses
        self.log('train_tempo', torch.mean(loss_tempo), on_step=False, on_epoch=True, logger=True)
        self.log('train_beats', torch.mean(loss_beats), on_step=False, on_epoch=True, logger=True)
        self.log('train_loss', torch.mean(loss), on_step=False, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):

        val_loss = 0

        audio_features = batch["audio_features"]
        
        tempo = batch["tempo"]
        # To One-hot
        tempo = F.one_hot(torch.round(tempo).long(), num_classes=300).float()

        # Add neighbour balancing kernel
        tempo = self.tempo_neighbour_smooth(tempo)

        beats = batch["beats"]
        beats = self.beats_neighbour_smooth(beats)

        # Forward pass
        tempo_hat, beats_hat = self.forward(audio_features)

        # Compute losses
        val_loss_tempo = self.tempo_loss(tempo_hat, tempo)
        val_loss_beats = self.beats_loss(beats_hat, beats)
        val_loss += self.tempo_lambda * 0 + val_loss_beats
        
        # Log losses
        self.log('val_tempo', torch.mean(val_loss_tempo), on_step=False, on_epoch=True, logger=True)
        self.log('val_beats', torch.mean(val_loss_beats), on_step=False, on_epoch=True, logger=True)
        self.log('val_loss', torch.mean(val_loss), on_step=False, on_epoch=True, logger=True)
        
        # Tempo metrics
        tempo_acc = tempo_acc_1(torch.argmax(tempo_hat), torch.argmax(tempo))
        self.log('tempo_acc_1', tempo_acc, on_step=False, on_epoch=True, logger=True)
        
        return torch.mean(val_loss)

    @torch.inference_mode()
    def get_tempo(self, audio, chunk_frames_postprocess=-1):
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
        to_logmel = LogMelSpectrogram(**LOGMEL_PARAMS).to(audio.device)
        
        beat_postprocessor = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55.0, max_bpm=215.0, fps=1./FRAME_UNIT, transition_lambda=100, threshold=0.05, correct=True
        )

        def _chunked_postprocess(beats, chunk_frames=3000):
            if chunk_frames < 0:
                return beat_postprocessor(beats)
            
            beats_postprocessed = []
            for frame_idx, frame in enumerate(range(0, len(beats), chunk_frames)):
                beats_chunk = beats[frame:frame + chunk_frames]
                beats_chunk = beat_postprocessor(beats_chunk)
                offset = frame_idx * chunk_frames * FRAME_UNIT
                beats_chunk = beats_chunk + offset
                beats_postprocessed.extend(beats_chunk)
            return beats_postprocessed

        def _chunked_inference(feat: torch.Tensor, chunk_frames=3000):
            if chunk_frames < 0:
                return self.forward(feat)

            tempos, beats = [], []

            for frame in range(0, feat.shape[1], chunk_frames):
                feat_chunk = feat[:, frame:frame + chunk_frames]
                tempo_chunk, beats_chunk = self.forward(feat_chunk)
                tempos.append(torch.argmax(tempo_chunk))
                beats.append(beats_chunk)
            beats = torch.cat(beats, dim=1)
            return tempos, beats

        feat = to_logmel(audio)[..., :-1].transpose(-1, -2)

        tempos, beats = _chunked_inference(feat)
        tempo = np.mean([i.item() for i in tempos])

        beats_act = torch.sigmoid(beats).squeeze()
        beats = _chunked_postprocess(beats_act.detach().squeeze().cpu().numpy(), chunk_frames_postprocess)
        
        # Get tempo from interbeats
        mean_beat_frames = np.diff(beats.astype(np.float32)).mean()
        if np.isnan(mean_beat_frames):
            tempo_ib = -1
        else:
            tempo_ib = round(60 / (mean_beat_frames.item()), 2)

        return {"beats": beats, "tempo": tempo, "tempo_from_interbeats": tempo_ib}