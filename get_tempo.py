import os
import yaml
import torch
import librosa
import argparse

from lightning_model import TempoBeatModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('audio_path',
                        type = str,
                        help = "input audio file")
    
    parser.add_argument("-ckpt",
                        "--ckpt_folder",
                        default='./checkpoints/tempo_derivated_from_beats',
                        type = str,
                        help = 'model ckpt folder')

    args = parser.parse_args()
    
    
# Load best checkpoint
ckpt_folder = args.ckpt_folder
ckpt_path = os.path.join(ckpt_folder, 'last.ckpt')
hparams_path = os.path.join(ckpt_folder, 'hparams.yml')
cfg = yaml.load(open(hparams_path), Loader=yaml.FullLoader)

# Instanciate PL model
model = TempoBeatModel(cfg)

# Load trained weights
state_dict = torch.load(ckpt_path)['state_dict']
model.load_state_dict(state_dict=state_dict)
model.eval()

# Load audio
wav, samplerate = librosa.load(args.audio_path, sr=44100, mono=True)
wav = torch.Tensor(wav)

# Infer
results = model.get_tempo(wav.unsqueeze(0))

# Print resutls 
print("Song tempo (through model's tempo classifier branch): ", results['tempo'], "BPM")
print("Song tempo (through model's beat detection branch): ", results['tempo_from_interbeats'], "BPM")
