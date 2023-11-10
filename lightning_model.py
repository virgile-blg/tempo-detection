import torch
import torch.nn as nn
import omegaconf as om
import pytorch_lightning as pl

from bock_network import BeatBockNet
from utils import BCESequenceLoss, MelSpecAugment
from metrics import tempo_acc_1

class TempoBeatModel(pl.LightningModule):
    def __init__(self, hparams: om.DictConfig):
        super().__init__()
        self.hparams.update(hparams)
        if not isinstance(hparams, om.DictConfig):
            hparams = om.DictConfig(hparams)
        self.hparams.update(om.OmegaConf.to_container(hparams, resolve=True))
        
        self.model = BeatBockNet(**self.hparams['model'])
        
        self.augment = torch.nn.Identity
        # self.augment = MelSpecAugment
        
        self.beats_loss = BCESequenceLoss
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
        beats = batch["beats"]
        tempo = batch["tempo"]

        # Augmentation if needed
        audio_features = self.augment(audio_features)
        beats = self.fp(beats)
        
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
        beats = batch["beats"]
        tempo = batch["tempo"]
        # beats_raw = batch["beats_raw"]
        # time_unit = batch["time_unit"].item()

        # Forward pass
        tempo_hat, beats_hat = self.forward(audio_features)
        
        # Compute losses
        val_loss_tempo = self.tempo_loss(tempo_hat, tempo)
        val_loss_beats = self.beats_loss(beats_hat, beats)
        val_loss += self.tempo_lambda * val_loss_tempo + val_loss_beats
        
        # Log losses
        self.log('val_tempo', torch.mean(val_loss_tempo), on_step=False, on_epoch=True, logger=True)
        self.log('val_beats', torch.mean(val_loss_beats), on_step=False, on_epoch=True, logger=True)
        self.log('val_loss', torch.mean(val_loss), on_step=False, on_epoch=True, logger=True)
        
        # Tempo metrics
        tempo_acc = tempo_acc_1(tempo_hat, tempo)
        self.log('tempo_acc_1', tempo_acc, on_step=False, on_epoch=True, logger=True)
        
        
        # TODO beats metrics computation
        # beats_hat = torch.sigmoid(beats_hat).squeeze()
        # beats_hat_pp = self.beat_postprocessor(beats_hat.detach().squeeze().cpu().numpy())
        # beat_indices = (beats_hat_pp / time_unit).astype(int)
        
        # beats_hat_nonpp = torch.where(beats_hat > 0.5, 1, 0).nonzero().squeeze().cpu().numpy() * time_unit
        
        # beat_metrics_pp = {k + "_pp": v for k, v in beats_score(beats_hat_pp, beats_raw).items()}
        # beat_metrics_nonpp = {k + "_nonpp": v for k, v in beats_score(beats_hat_nonpp, beats_raw).items()}
        
        return torch.mean(val_loss)
