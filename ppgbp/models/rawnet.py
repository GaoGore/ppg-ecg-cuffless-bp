"""Raw-waveform network: the BeatNet v2 subject model (bag of rounds, attention
pooling, demographics fusion, quantile + auxiliary + round heads, same losses)
with a different input and encoder.

Input per round: a random 10-second window of the 16 band-passed PPG pulses
and the ECG at 125 Hz or 250 Hz (robust-normalised per channel, cache.build_raw), plus
first differences. Several beats per window let the encoder see beat-to-beat
dynamics that averaged templates remove. At inference every round is covered
by non-overlapping windows and the predictions are averaged. Windows are only
shifted in whole, never stretched, so ECG-to-pulse timing is intact.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import cache
from ..tables import DEMO
from .beatnet import ALPHAS, device
from .beatnet2 import BeatNet2Model, ResBlockK


class RawStore:
    _inst: dict = {}

    @classmethod
    def get(cls, fs=cache.RAW_FS):
        if fs not in cls._inst:
            cls._inst[fs] = cls(fs)
        return cls._inst[fs]

    def __init__(self, fs=cache.RAW_FS):
        px, pm = cache.raw_paths(fs)
        self.x = torch.from_numpy(np.load(px))                                          # (N, 17, 60*fs) float16
        self.mask = torch.from_numpy(np.load(pm).astype(np.float32))


class RawNet(nn.Module):
    def __init__(self, n_scal, n_demo, width=64, n_q=len(ALPHAS), drop=0.2, kernel=7, n_blocks=5, fs=cache.RAW_FS):
        super().__init__()
        # higher sampling rates keep the first layer's finer timing but downsample harder in the stem, so the
        # rest of the encoder runs at the same time resolution (and cost) as the 125 Hz model
        up = fs // cache.RAW_FS
        k0 = 7 if up == 1 else 4 * up + 3
        self.stem = nn.Sequential(nn.Conv1d(34, width, k0, stride=2 * up, padding=k0 // 2, bias=False), nn.BatchNorm1d(width), nn.GELU())
        mult = [1, 2, 2, 4, 4, 4][:n_blocks]
        blocks, c_in = [], width
        for m in mult:
            blocks.append(ResBlockK(c_in, width * m, 2, kernel))
            c_in = width * m
        self.blocks = nn.Sequential(*blocks)
        self.t_att = nn.Conv1d(c_in, 1, 1)                                             # attention over time
        self.scal = nn.Sequential(nn.Linear(n_scal, 32), nn.GELU())
        self.seg = nn.Sequential(nn.Linear(2 * c_in + 32, 128), nn.GELU(), nn.Dropout(drop))
        self.att = nn.Linear(128, 1)
        self.aux = nn.Linear(128, 4)
        self.demo = nn.Sequential(nn.Linear(n_demo, 32), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(128 + 32, 128), nn.GELU(), nn.Dropout(drop))
        self.q_base = nn.Linear(128, 2)
        self.q_inc = nn.Linear(128, 2 * (n_q - 1))
        self.seg_head = nn.Sequential(nn.Linear(128 + 32, 64), nn.GELU(), nn.Linear(64, 2))
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64))
        self.n_q = n_q

    def encoder_parameters(self):
        return [p for m in (self.stem, self.blocks, self.t_att, self.scal, self.seg) for p in m.parameters()]

    def encode(self, x, mask, scal):
        x = x * mask[:, :, None]
        d = torch.diff(x, dim=2, prepend=x[:, :, :1]) * 4
        h = self.blocks(self.stem(torch.cat([x, d], dim=1)))
        a = torch.softmax(self.t_att(h), dim=2)
        h = torch.cat([(h * a).sum(dim=2), h.amax(dim=2)], dim=1)
        return self.seg(torch.cat([h, self.scal(scal)], dim=1))

    def forward(self, x, mask, scal, bag_mask, demo):
        B, S = bag_mask.shape
        e = self.encode(x.flatten(0, 1), mask.flatten(0, 1), scal.flatten(0, 1)).view(B, S, -1)
        a = torch.softmax(self.att(e).squeeze(-1).masked_fill(bag_mask == 0, -1e4), dim=1)
        z = (a[:, :, None] * e).sum(dim=1)
        aux = self.aux(z)
        dm = self.demo(demo)
        h = self.fuse(torch.cat([z, dm], dim=1))
        base = self.q_base(h)
        inc = F.softplus(self.q_inc(h)).view(B, 2, self.n_q - 1)
        q = torch.cat([base[:, :, None], base[:, :, None] + torch.cumsum(inc, dim=2)], dim=2)
        seg = self.seg_head(torch.cat([e, dm[:, None, :].expand(B, S, -1)], dim=2))
        return q, aux, seg


class RawNetModel(BeatNet2Model):
    """BeatNet2Model with raw 10-s windows instead of beat templates."""

    def __init__(self, seed=0, crop_s=10.0, width=64, n_blocks=5, epochs=40, batch=24, fs=cache.RAW_FS, **kw):
        super().__init__(seed=seed, epochs=epochs, width=width, n_blocks=n_blocks, batch=batch, **kw)
        self.fs = fs
        self.raw = RawStore.get(fs)
        self.crop = int(crop_s * fs)
        self.n_windows = self.raw.x.shape[2] // self.crop

    def _make_net(self):
        return RawNet(self.st.n_scal, len(DEMO), self.width, drop=self.drop, kernel=self.kernel, n_blocks=self.n_blocks, fs=self.fs)

    def _bag(self, sids, rng, train, window=None):
        R, bm = self._rows(sids)
        Rc = torch.from_numpy(np.where(bm, R, 0))
        L = self.raw.x.shape[2]
        if train:
            starts = rng.integers(0, L - self.crop + 1, size=R.shape)
        else:
            starts = np.full(R.shape, (window % self.n_windows) * self.crop)
        idx = torch.from_numpy(starts)[..., None] + torch.arange(self.crop)               # (B, S, crop)
        x = self.raw.x[Rc]                                                               # (B, S, 17, L)
        x = torch.gather(x, 3, idx[:, :, None, :].expand(-1, -1, 17, -1)).float()
        bmt = torch.from_numpy(bm.astype(np.float32))
        mk = self.raw.mask[Rc] * bmt[..., None]
        if train:
            mk = mk * (torch.from_numpy(rng.random(mk.shape)).float() > 0.1)
            x = x * (1 + 0.1 * torch.randn(x.shape[0], x.shape[1], 17, 1))
        sc = self.st.scal[Rc] * bmt[..., None]
        return x, mk, sc, bmt, Rc

    @torch.no_grad()
    def predict_all(self, X):
        dev = device()
        self.net.eval()
        sids = np.asarray(X.index)
        demo = self._demo(X)
        qs, auxs, segs = [], [], []
        for w in range(self.n_windows):
            qq, aa, ss = [], [], []
            for b0 in range(0, len(sids), 32):
                x, mk, sc, bm, _ = self._bag(sids[b0:b0 + 32], None, False, window=w)
                q, aux, seg = self.net(x.to(dev), mk.to(dev), sc.to(dev), bm.to(dev), demo[b0:b0 + 32].to(dev))
                qq.append(q.cpu().numpy()); aa.append(aux.cpu().numpy()); ss.append(seg.cpu().numpy())
            qs.append(np.concatenate(qq)); auxs.append(np.concatenate(aa)); segs.append(np.concatenate(ss))
        q = np.mean(qs, 0) * self.y_sd[None, :, None] + self.y_mu[None, :, None]
        seg = np.mean(segs, 0) * self.y_sd + self.y_mu
        return np.sort(q, axis=2), np.mean(auxs, 0), seg
