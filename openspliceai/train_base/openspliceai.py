"""
Filename: train.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Train the OpenSpliceAI model.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)

    def forward(self, x, y):
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        return x + out, y


class Cropping1D(nn.Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        return x[:, :, self.cropping[0]:-self.cropping[1]] if self.cropping[1] > 0 else x[:, :, self.cropping[0]:]


class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y


class StructureEncoder(nn.Module):
    """Parallel encoder for the RNA-structure channels (dot-bracket + G-U wobble).

    Feeding structure through the width-1 ``initial_conv`` blends it into the
    shared trunk at a single position, so the model cannot build multi-nucleotide
    structural features (stem-loops) before sequence and structure are mixed.
    This branch instead gives structure its own capacity and *wide, dilated*
    convolutions, producing local structural features (~341 nt receptive field,
    which covers the span most splice-site base pairs occupy) that are then fused
    into the sequence trunk. Kept length-preserving so fusion is a plain add.
    """
    # (kernel_width, dilation) per dilated residual block.
    BLOCKS = [(11, 1), (21, 2), (31, 4)]

    def __init__(self, in_channels, out_channels, hidden=64):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden, 1)
        self.units = nn.ModuleList(
            [ResidualUnit(hidden, w, ar) for (w, ar) in self.BLOCKS]
        )
        self.project = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, x):
        x = self.lift(x)
        for unit in self.units:
            x, _ = unit(x, 0)
        return self.project(x)


class SpliceAI(nn.Module):
    def __init__(self, L, W, AR, apply_softmax=True, in_channels=4):
        super(SpliceAI, self).__init__()
        self.apply_softmax = apply_softmax  # new parameter to control softmax usage
        self.in_channels = in_channels
        # Structure-branch model: when extra (structure) channels are present,
        # route the 4 sequence channels through the trunk and the remaining
        # structure channels through a dedicated parallel encoder.
        self.use_structure = in_channels > 4
        seq_channels = 4 if self.use_structure else in_channels
        self.initial_conv = nn.Conv1d(seq_channels, L, 1)
        if self.use_structure:
            self.structure_encoder = StructureEncoder(in_channels - seq_channels, L)
            self.ablate_structure = False  # runtime toggle for zero-out ablation
        self.initial_skip = Skip(L)
        self.residual_units = nn.ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, w, r))
            if (i+1) % 4 == 0:
                self.residual_units.append(Skip(L))
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))

    def forward(self, x):
        if self.use_structure:
            x_struct = x[:, 4:, :]
            x = self.initial_conv(x[:, :4, :])
            if not self.ablate_structure:
                x = x + self.structure_encoder(x_struct)
        else:
            x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            x, skip = m(x, skip)
        final_x = self.crop(skip)
        out = self.final_conv(final_x)
        if self.apply_softmax:
            return F.softmax(out, dim=1)
        else:
            return out


def infer_in_channels(state_dict):
    """Recover the total input-channel count from a checkpoint, supporting both
    the plain model (channels live in ``initial_conv``) and the structure-branch
    model (4 sequence channels in ``initial_conv`` + structure channels in
    ``structure_encoder.lift``)."""
    seq = state_dict['initial_conv.weight'].shape[1]
    key = 'structure_encoder.lift.weight'
    if key in state_dict:
        return seq + state_dict[key].shape[1]
    return seq
