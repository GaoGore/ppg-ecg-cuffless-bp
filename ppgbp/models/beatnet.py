"""D4: multi-instance beat-template network.

Each subject is a bag of segments; each segment contributes one beat template
(17 channels x 300 samples at 250 Hz, R-aligned, randomly drawn from 16
random-beat-subset averages + the full average, which acts as augmentation).
A 1D residual CNN encodes templates; segment scalars (log perfusion per
channel, heart rate, ECG flag) are appended; attention pooling over the bag
gives a subject embedding, fused with demographics to predict SBP/DBP
quantiles (monotone) plus an auxiliary age prediction from signals only.
Timing is never shifted in augmentation: R-aligned timing carries the pulse
arrival information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .. import cache
from ..tables import DEMO

ALPHAS = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Store:
    """All segments' templates and scalar inputs, indexed by subject id."""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        meta, feats = cache.load()
        tmpl, masks = cache.load_templates(mmap=False)
        self.tmpl = torch.from_numpy(np.ascontiguousarray(tmpl))          # (N, K+1, 17, L) float16
        self.masks = torch.from_numpy(masks.astype(np.float32))           # (N, 17)
        sc_cols = [c for c in feats.columns if c.endswith("__pi_log") and c.startswith("PPG_")]
        sc_cols += ["rhy__hr", "ecg_ok", "all__t_d2", "all__t_ms"]
        S = feats[sc_cols].astype(float)
        mu, sd = S.mean(), S.std().replace(0, 1)
        Z = ((S - mu) / sd)
        nan = Z.isna().values.astype(np.float32)
        self.scal = torch.from_numpy(np.c_[Z.fillna(0).values.astype(np.float32), nan[:, :16]])
        self.rows = meta.groupby("sid").indices                          # sid -> array of rows
        self.round = torch.from_numpy(meta["round"].values.astype(np.float32))
        self.n_scal = self.scal.shape[1]
        sids = sorted(self.rows)
        self.sid_index = {s: i for i, s in enumerate(sids)}
        self.max_seg = max(len(v) for v in self.rows.values())
        self.pad = np.full((len(sids), self.max_seg), -1, dtype=np.int64)  # padded segment rows
        for s, i in self.sid_index.items():
            self.pad[i, :len(self.rows[s])] = self.rows[s]


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sk = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
                   if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        return F.gelu(self.b2(self.c2(h)) + self.sk(x))


class Net(nn.Module):
    def __init__(self, n_scal, n_demo, width=48, n_q=len(ALPHAS), drop=0.2):
        super().__init__()
        cin = 17 * 2 + 17                                   # templates, derivatives, mask planes
        self.stem = nn.Sequential(nn.Conv1d(cin, width, 9, padding=4, bias=False), nn.BatchNorm1d(width), nn.GELU())
        self.blocks = nn.Sequential(ResBlock(width, width, 2), ResBlock(width, 2 * width, 2),
                                    ResBlock(2 * width, 2 * width, 2), ResBlock(2 * width, 4 * width, 2))
        emb = 4 * width * 2
        self.scal = nn.Sequential(nn.Linear(n_scal, 32), nn.GELU())
        self.seg = nn.Sequential(nn.Linear(emb + 32, 128), nn.GELU(), nn.Dropout(drop))
        self.att = nn.Linear(128, 1)
        self.age_head = nn.Linear(128, 1)
        self.demo = nn.Sequential(nn.Linear(n_demo, 32), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(128 + 32, 128), nn.GELU(), nn.Dropout(drop))
        self.q_base = nn.Linear(128, 2)
        self.q_inc = nn.Linear(128, 2 * (n_q - 1))
        self.n_q = n_q

    def encode(self, x, mask, scal):
        # x: (B*S, 17, L)  mask: (B*S, 17)
        d = torch.diff(x, dim=2, prepend=x[:, :, :1]) * 10
        m = mask[:, :, None].expand_as(x)
        h = self.stem(torch.cat([x * m, d * m, m], dim=1))
        h = self.blocks(h)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.seg(torch.cat([h, self.scal(scal)], dim=1))

    def forward(self, x, mask, scal, bag_mask, demo):
        B, S = bag_mask.shape
        e = self.encode(x.flatten(0, 1), mask.flatten(0, 1), scal.flatten(0, 1)).view(B, S, -1)
        a = self.att(e).squeeze(-1).masked_fill(bag_mask == 0, -1e4)
        a = torch.softmax(a, dim=1)
        z = (a[:, :, None] * e).sum(dim=1)
        age = self.age_head(z).squeeze(-1)
        h = self.fuse(torch.cat([z, self.demo(demo)], dim=1))
        base = self.q_base(h)                                          # (B, 2)
        inc = F.softplus(self.q_inc(h)).view(B, 2, self.n_q - 1)
        q = torch.cat([base[:, :, None], base[:, :, None] + torch.cumsum(inc, dim=2)], dim=2)
        return q, age


def pinball(q, y, alphas):
    # q: (B, 2, Q), y: (B, 2)
    diff = y[:, :, None] - q
    return torch.maximum(alphas * diff, (alphas - 1) * diff).mean(dim=2)


class BeatNetModel:
    """fit(X, Y, w) / predict(X) / predict_quantiles(X); X is the subject table
    (index = subject id) and must contain the DEMO columns."""

    def __init__(self, seed=0, epochs=60, lr=2e-3, width=48, batch=32, age_w=0.3, n_eval=8, wd=1e-3):
        self.seed, self.epochs, self.lr, self.width, self.batch = seed, epochs, lr, width, batch
        self.age_w, self.n_eval, self.wd = age_w, n_eval, wd
        self.st = Store.get()
        self.alphas = ALPHAS

    def _bag(self, sids, rng, train):
        st = self.st
        R = st.pad[[st.sid_index[s] for s in sids]]                       # (B, S) rows, -1 = padding
        bm = R >= 0
        Rc = torch.from_numpy(np.where(bm, R, 0))
        K1 = st.tmpl.shape[1]
        ks = torch.from_numpy(rng.integers(0 if train else 1, K1, size=R.shape))
        x = st.tmpl[Rc, ks].float()                                       # (B, S, 17, L)
        bmt = torch.from_numpy(bm.astype(np.float32))
        mk = st.masks[Rc] * bmt[..., None]
        if train:
            mk = mk * (torch.from_numpy(rng.random(mk.shape)).float() > 0.1)
            x = x * (1 + 0.05 * torch.randn(x.shape[0], x.shape[1], 17, 1))
        sc = st.scal[Rc] * bmt[..., None]
        return x, mk, sc, bmt

    def _demo(self, X):
        D = X[DEMO].astype(float).values
        return torch.from_numpy(np.nan_to_num((D - self.d_mu) / self.d_sd).astype(np.float32))

    def fit(self, X, Y, w=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = device()
        D = X[DEMO].astype(float).values
        self.d_mu, self.d_sd = np.nanmean(D, 0), np.nanstd(D, 0) + 1e-6
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0)
        self.a_mu, self.a_sd = X["age"].mean(), X["age"].std()
        sids = np.asarray(X.index)
        demo = self._demo(X)
        Yn = torch.from_numpy(((Y - self.y_mu) / self.y_sd).astype(np.float32))
        age = torch.from_numpy(((X["age"].values - self.a_mu) / self.a_sd).astype(np.float32))
        wt = torch.ones(len(sids)) if w is None else torch.from_numpy(np.asarray(w, np.float32))
        self.net = Net(self.st.n_scal, len(DEMO), self.width).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(sids) / self.batch))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.15)
        al = torch.tensor(self.alphas, dtype=torch.float32, device=dev)
        tw = torch.tensor([0.6, 0.4], device=dev)
        self.net.train()
        for ep in range(self.epochs):
            perm = rng.permutation(len(sids))
            for b0 in range(0, len(perm), self.batch):
                idx = perm[b0:b0 + self.batch]
                x, mk, sc, bm = self._bag(sids[idx], rng, True)
                q, a = self.net(x.to(dev), mk.to(dev), sc.to(dev), bm.to(dev), demo[idx].to(dev))
                lq = (pinball(q, Yn[idx].to(dev), al) * tw).sum(dim=1)
                la = F.smooth_l1_loss(a, age[idx].to(dev), reduction="none")
                ww = wt[idx].to(dev)
                loss = ((lq + self.age_w * la) * ww).sum() / ww.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                opt.step()
                sch.step()
        return self

    @torch.no_grad()
    def predict_all(self, X):
        dev = device()
        self.net.eval()
        sids = np.asarray(X.index)
        demo = self._demo(X)
        rng = np.random.default_rng(12345)
        qs, ages = [], []
        for _ in range(self.n_eval):
            qq, aa = [], []
            for b0 in range(0, len(sids), 64):
                x, mk, sc, bm = self._bag(sids[b0:b0 + 64], rng, False)
                q, a = self.net(x.to(dev), mk.to(dev), sc.to(dev), bm.to(dev), demo[b0:b0 + 64].to(dev))
                qq.append(q.cpu().numpy())
                aa.append(a.cpu().numpy())
            qs.append(np.concatenate(qq))
            ages.append(np.concatenate(aa))
        q = np.mean(qs, axis=0) * self.y_sd[None, :, None] + self.y_mu[None, :, None]
        age = np.mean(ages, axis=0) * self.a_sd + self.a_mu
        return np.sort(q, axis=2), age

    def predict_quantiles(self, X):
        return self.predict_all(X)[0]

    def predict(self, X):
        return self.predict_quantiles(X)[:, :, int(np.argmin(np.abs(self.alphas - 0.5)))]
