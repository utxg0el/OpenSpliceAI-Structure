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


class SpliceAI(nn.Module):
    def __init__(self, L, W, AR, apply_softmax=True):
        super(SpliceAI, self).__init__()
        self.apply_softmax = apply_softmax  # new parameter to control softmax usage
        self.initial_conv = nn.Conv1d(4, L, 1)
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


class GatedStructureHead(nn.Module):
    """Structure branch (zero-init correction) + sequence-confidence gate (see GatedSpliceAI).
    struct_in: (B, n_struct, SL) cropped accessibility channels; seq_logit: (B, 3, SL)."""
    def __init__(self, n_struct, hidden=16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_struct, hidden, 1)
        self.act = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(hidden, 3, 1)        # correction logits
        self.gate = nn.Conv1d(3, 1, 1)              # gate from the sequence logits
        # zero-init => correction == 0 and gate bias == 0 (sigmoid(0)=0.5) at start => exact no-op
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, struct_in, seq_logit):
        corr = self.conv2(self.act(self.conv1(struct_in)))   # (B, 3, SL)
        g = torch.sigmoid(self.gate(seq_logit))              # (B, 1, SL)
        return g * corr, g, corr


class GatedSpliceAI(nn.Module):
    """4ch SpliceAI backbone + gated late-fusion structure correction.

    The 8 accessibility channels do NOT mix into the sequence pathway (unlike concatenation at the input
    conv). Instead: a sequence backbone (a stock 4ch SpliceAI, weights loadable from a vanilla checkpoint)
    produces the splice logits; a small structure branch produces a per-position 3-class CORRECTION that is
    ZERO-INITIALISED (so the model starts exactly as the 4ch model and can only earn structure usage); a GATE
    derived from the sequence logits scales the correction (low where sequence is confident).

        seq_logit   = backbone(x[:, :n_seq])
        struct_corr = struct_head(crop(x[:, n_seq:]))   # == 0 at init
        g           = sigmoid(gate(seq_logit))          # (B,1,SL) in [0,1]
        out         = softmax(seq_logit + g * struct_corr)

    Input x: (B, in_channels, SL+CL), channels [0:n_seq]=sequence one-hot, [n_seq:]=accessibility.
    Set apply_softmax=False to return logits for the training loss."""
    def __init__(self, L, W, AR, apply_softmax=True, in_channels=12, n_seq_channels=4):
        super().__init__()
        assert in_channels > n_seq_channels, "GatedSpliceAI needs structure channels (in_channels > n_seq_channels)"
        self.apply_softmax = apply_softmax
        self.n_seq = n_seq_channels
        self.n_struct = in_channels - n_seq_channels
        self.backbone = SpliceAI(L, W, AR, apply_softmax=False)   # state_dict matches a vanilla 4ch checkpoint
        self.struct_head = GatedStructureHead(self.n_struct)
        self.last_gate = None
        self.last_corr = None

    def load_backbone(self, state_dict, strict=True):
        """Load a 4ch vanilla SpliceAI checkpoint into the sequence backbone."""
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        seq_logit = self.backbone(x[:, :self.n_seq, :])          # (B, 3, SL)
        struct_in = self.backbone.crop(x[:, self.n_seq:, :])     # (B, n_struct, SL) — same crop as backbone
        gated_corr, g, corr = self.struct_head(struct_in, seq_logit)
        self.last_gate = g
        self.last_corr = corr
        out = seq_logit + gated_corr
        return F.softmax(out, dim=1) if self.apply_softmax else out

    def struct_l2(self):
        """Mean |correction| (optional regulariser keeping structure near no-op)."""
        return self.last_corr.abs().mean() if self.last_corr is not None else torch.tensor(0.0)
