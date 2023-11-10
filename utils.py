import torch.nn as nn
import random as rd
import torch as th
import torch.nn.functional as F
import torchaudio as ta
import torchaudio.transforms as tat
import math


class CrossEntropySequenceLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        # input: batch * seq * C, target: batch * seq
        input = input.reshape(-1, input.shape[-1])
        target = target.reshape(-1)

        return super().forward(input, target)


class BCESequenceLoss(nn.BCEWithLogitsLoss):
    def forward(self, input: th.Tensor, target: th.Tensor):
        # input: batch * seq * C, target: batch * seq
        input = input.reshape(-1, input.shape[-1])  
        target = target.reshape(-1, target.shape[-1])
        return super().forward(input, target)


class LogMelSpectrogram(nn.Module):
    def __init__(
            self,
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            f_min,
            f_max,
            norm,
            is_log,
            eps=1e-6,
            center=True
        ):
        super().__init__()
        assert norm in [None, "slaney"]

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.window_length = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.norm = norm
        self.eps = eps
        self.is_log = is_log

        self.to_melspec = ta.transforms.MelSpectrogram(
            sample_rate, n_fft, hop_length=hop_length, n_mels=n_mels,
            f_min=f_min, f_max=f_max, power=1, center=center,
            norm=norm, mel_scale="htk" if norm is None else "slaney"
        )

    def forward(self, waveform):
        mel = self.to_melspec(waveform)
        if self.is_log:
            mel = th.log(mel + self.eps)
        return mel


class MelPitchShift(nn.Module):
    def __init__(self, bin_range=[-8, 8], dim=-2, eps=1e-6):
        super().__init__()
        self.bin_range = bin_range
        self.dim = dim
        self.eps = math.log(eps)

        assert self.dim == -2, "Only supports dim=-2 for now."
    
    def forward(self, input):
        shift = rd.randint(*self.bin_range)
        input = th.roll(input, shift, dims=self.dim)
        if shift > 0:
            input[..., :shift, :] = self.eps
        elif shift < 0:
            input[..., shift:, :] = self.eps
        return input
    

class MelFreqMask(nn.Module):
    def __init__(self, nbins=32, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.transform = tat.FrequencyMasking(nbins)
    
    def forward(self, input):
        if rd.random() > self.threshold:
            input = self.transform(input)
        return input


class MelSpecAugment(nn.Module):
    def __init__(self, ps_params=None, fm_params=None):
        super().__init__()

        self.ps = MelPitchShift() if ps_params is None else MelPitchShift(**ps_params)
        self.fm = MelFreqMask() if fm_params is None else MelFreqMask(**fm_params)

    def forward(self, input):
        return self.ps(self.fm(input))


class NeighbourBalancingKernel(nn.Module):
    def __init__(self, weights=[0.5, 1, 0.5]):
        super().__init__()
        self.window = nn.Parameter(th.tensor([[weights]]), requires_grad=False)
        self.padding = len(weights) // 2

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x.unsqueeze(1)
        x = F.conv1d(x, self.window, padding=self.padding)
        x = x.squeeze(1)
        return x