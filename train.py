import os
import yaml
import argparse
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from lightning_model import *
from data import *


pl.seed_everything()

def main(hparams_file):

    cfg = yaml.load(open(args.hparams), Loader=yaml.FullLoader)

    # Get checkpoint folder
    ckpt_folder = os.path.join('./checkpoints',Path(args.hparams).stem)
    os.makedirs(ckpt_folder, exist_ok=True)
    with open(os.path.join(ckpt_folder, 'hparams.yml'), 'w') as file:
        yaml.dump(cfg, file)

    # Load model
    model = TempoBeatModel(cfg)
    # Load data
    datamodule = GTZANDataModule(**cfg['data'])
    # TB Log
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=os.path.splitext(os.path.basename(args.hparams))[0])
    # Callbacks
    checkpoint_callback = ModelCheckpoint(**cfg['model_checkpoint'], dirpath=ckpt_folder)
    early_stopping_calllback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=500)
    # Trainer
    trainer = pl.Trainer(**cfg['trainer'],
                        logger=logger,
                        callbacks=[checkpoint_callback, early_stopping_calllback])
    # Train
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)
