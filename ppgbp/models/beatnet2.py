"""D4 v2: beat-template network with round-level supervision, auxiliary
targets and subject-contrastive pretraining.

Changes over beatnet.py (same inputs, encoder and bag/attention design):
* Round-level head: every round's own BP label supervises that round's
  embedding (3,620 labels instead of 900 subject means). At inference it gives
  per-round deviations from the subject mean.
* Auxiliary targets predicted from the signal-only subject embedding: age,
  BMI, sex and antihypertensive medication. Medication pushes the encoder
  towards the vascular signatures of treated hypertension.
* Optional contrastive pretraining (InfoNCE) on the training fold's own
  signals: two templates from the same subject (different rounds and/or
  random beat subsets) are positives, other subjects in the batch negatives.
  Per-segment scalars are zeroed during pretraining so matching must rely on
  waveform shape and timing. No labels and no test signals are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .. import cache
from ..tables import DEMO
from .beatnet import ALPHAS, ResBlock, Store, device, pinball


class ResBlockK(nn.Module):
    """Residual block with configurable kernel size."""

    def __init__(self, cin, cout, stride, k=7):
        super().__init__()
        p = k // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=p, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sk = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
                   if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        return F.gelu(self.b2(self.c2(h)) + self.sk(x))


class Net2(nn.Module):
    def __init__(self, n_scal, n_demo, width=48, n_q=len(ALPHAS), drop=0.2, kernel=7, n_blocks=4, pool="att"):
        super().__init__()
        cin = 17 * 3                                        # templates, derivatives, mask planes
        self.stem = nn.Sequential(nn.Conv1d(cin, width, 9, padding=4, bias=False), nn.BatchNorm1d(width), nn.GELU())
        mult = [1, 2, 2, 4, 4, 4][:n_blocks]
        chans = [width * m for m in mult]
        blocks, c_in = [], width
        for c in chans:
            blocks.append(ResBlockK(c_in, c, 2, kernel))
            c_in = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = pool
        emb = chans[-1] * 2
        self.scal = nn.Sequential(nn.Linear(n_scal, 32), nn.GELU())
        self.seg = nn.Sequential(nn.Linear(emb + 32, 128), nn.GELU(), nn.Dropout(drop))
        self.att = nn.Linear(128, 1)
        self.aux = nn.Linear(128, 4)                        # age, BMI (regression); male, medication (logits)
        self.demo = nn.Sequential(nn.Linear(n_demo, 32), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(128 + 32, 128), nn.GELU(), nn.Dropout(drop))
        self.q_base = nn.Linear(128, 2)
        self.q_inc = nn.Linear(128, 2 * (n_q - 1))
        self.seg_head = nn.Sequential(nn.Linear(128 + 32, 64), nn.GELU(), nn.Linear(64, 2))
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64))
        self.n_q = n_q

    def encoder_parameters(self):
        return [p for m in (self.stem, self.blocks, self.scal, self.seg) for p in m.parameters()]

    def encode(self, x, mask, scal):
        d = torch.diff(x, dim=2, prepend=x[:, :, :1]) * 10
        m = mask[:, :, None].expand_as(x)
        h = self.blocks(self.stem(torch.cat([x * m, d * m, m], dim=1)))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.seg(torch.cat([h, self.scal(scal)], dim=1))

    def forward(self, x, mask, scal, bag_mask, demo):
        B, S = bag_mask.shape
        e = self.encode(x.flatten(0, 1), mask.flatten(0, 1), scal.flatten(0, 1)).view(B, S, -1)
        if self.pool == "mean":
            a = bag_mask / bag_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            a = torch.softmax(self.att(e).squeeze(-1).masked_fill(bag_mask == 0, -1e4), dim=1)
        z = (a[:, :, None] * e).sum(dim=1)
        aux = self.aux(z)
        dm = self.demo(demo)
        h = self.fuse(torch.cat([z, dm], dim=1))
        base = self.q_base(h)
        inc = F.softplus(self.q_inc(h)).view(B, 2, self.n_q - 1)
        q = torch.cat([base[:, :, None], base[:, :, None] + torch.cumsum(inc, dim=2)], dim=2)
        seg = self.seg_head(torch.cat([e, dm[:, None, :].expand(B, S, -1)], dim=2))   # (B, S, 2)
        return q, aux, seg


class BeatNet2Model:
    """Same interface as BeatNetModel, plus predict_seg(X) -> per-round deviations."""

    def __init__(self, seed=0, epochs=60, lr=2e-3, width=48, batch=32, n_eval=8, wd=1e-3,
                 seg_w=0.5, aux_w=0.3, ssl_epochs=0, ssl_lr=1e-3, ssl_batch=64, ssl_temp=0.1, enc_lr_mult=1.0,
                 drop=0.2, kernel=7, n_blocks=4, pool="att"):
        self.__dict__.update(dict(seed=seed, epochs=epochs, lr=lr, width=width, batch=batch, n_eval=n_eval, wd=wd,
                                  seg_w=seg_w, aux_w=aux_w, ssl_epochs=ssl_epochs, ssl_lr=ssl_lr,
                                  ssl_batch=ssl_batch, ssl_temp=ssl_temp, enc_lr_mult=enc_lr_mult,
                                  drop=drop, kernel=kernel, n_blocks=n_blocks, pool=pool))
        self.st = Store.get()
        self.alphas = ALPHAS
        if not hasattr(self.st, "yseg"):
            meta, _ = cache.load()
            self.st.yseg = torch.from_numpy(meta[["sbp", "dbp"]].values.astype(np.float32))
            self.st.seg_ids = meta.seg_id.values

    # ---------------------------------------------------------------- data
    def _rows(self, sids):
        R = self.st.pad[[self.st.sid_index[s] for s in sids]]
        return R, R >= 0

    def _bag(self, sids, rng, train):
        st = self.st
        R, bm = self._rows(sids)
        Rc = torch.from_numpy(np.where(bm, R, 0))
        ks = torch.from_numpy(rng.integers(0 if train else 1, st.tmpl.shape[1], size=R.shape))
        x = st.tmpl[Rc, ks].float()
        bmt = torch.from_numpy(bm.astype(np.float32))
        mk = st.masks[Rc] * bmt[..., None]
        if train:
            mk = mk * (torch.from_numpy(rng.random(mk.shape)).float() > 0.1)
            x = x * (1 + 0.05 * torch.randn(x.shape[0], x.shape[1], 17, 1))
        sc = st.scal[Rc] * bmt[..., None]
        return x, mk, sc, bmt, Rc

    def _view(self, sids, rng):
        """One random template from a random round of each subject, for pretraining."""
        st = self.st
        R, bm = self._rows(sids)
        pick = np.array([rng.choice(np.flatnonzero(b)) for b in bm])
        rows = torch.from_numpy(R[np.arange(len(sids)), pick])
        ks = torch.from_numpy(rng.integers(0, st.tmpl.shape[1], size=len(sids)))
        x = st.tmpl[rows, ks].float() * (1 + 0.05 * torch.randn(len(sids), 17, 1))
        mk = st.masks[rows] * (torch.from_numpy(rng.random((len(sids), 17))).float() > 0.2)
        return x, mk

    def _demo(self, X):
        D = X[DEMO].astype(float).values
        return torch.from_numpy(np.nan_to_num((D - self.d_mu) / self.d_sd).astype(np.float32))

    # ---------------------------------------------------------------- training
    def _make_net(self):
        return Net2(self.st.n_scal, len(DEMO), self.width, drop=self.drop, kernel=self.kernel,
                    n_blocks=self.n_blocks, pool=self.pool)

    def _pretrain(self, sids, rng, dev):
        net = self.net
        params = net.encoder_parameters() + list(net.proj.parameters())
        opt = torch.optim.AdamW(params, lr=self.ssl_lr, weight_decay=self.wd)
        steps = self.ssl_epochs * int(np.ceil(len(sids) / self.ssl_batch))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.ssl_lr, total_steps=steps, pct_start=0.1)
        zeros = torch.zeros(self.ssl_batch, self.st.n_scal, device=dev)
        net.train()
        for _ in range(self.ssl_epochs):
            perm = rng.permutation(len(sids))
            for b0 in range(0, len(perm), self.ssl_batch):
                b = sids[perm[b0:b0 + self.ssl_batch]]
                if len(b) < 8:
                    continue
                (x1, m1), (x2, m2) = self._view(b, rng), self._view(b, rng)
                z1 = F.normalize(net.proj(net.encode(x1.to(dev), m1.to(dev), zeros[:len(b)])), dim=1)
                z2 = F.normalize(net.proj(net.encode(x2.to(dev), m2.to(dev), zeros[:len(b)])), dim=1)
                logits = z1 @ z2.T / self.ssl_temp
                target = torch.arange(len(b), device=dev)
                loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 2.0)
                opt.step()
                sch.step()

    def fit(self, X, Y, w=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = device()
        D = X[DEMO].astype(float).values
        self.d_mu, self.d_sd = np.nanmean(D, 0), np.nanstd(D, 0) + 1e-6
        self.y_mu, self.y_sd = Y.mean(0).astype(np.float32), Y.std(0).astype(np.float32)
        sids = np.asarray(X.index)
        demo = self._demo(X)
        Yn = torch.from_numpy(((Y - self.y_mu) / self.y_sd).astype(np.float32))
        bmi = X["bmi"].astype(float).values
        aux_reg = np.c_[(X["age"].values - X["age"].mean()) / X["age"].std(), (bmi - np.nanmean(bmi)) / np.nanstd(bmi)]
        aux_reg = torch.from_numpy(np.nan_to_num(aux_reg).astype(np.float32))
        aux_bin = torch.from_numpy(X[["male", "med_his"]].astype(float).values.astype(np.float32))
        wt = torch.ones(len(sids)) if w is None else torch.from_numpy(np.asarray(w, np.float32))
        y_mu_t, y_sd_t = torch.from_numpy(self.y_mu), torch.from_numpy(self.y_sd)
        self.net = self._make_net().to(dev)
        if self.ssl_epochs > 0:
            self._pretrain(sids, rng, dev)
        enc = set(id(p) for p in self.net.encoder_parameters())
        groups = [{"params": self.net.encoder_parameters(), "lr": self.lr * self.enc_lr_mult},
                  {"params": [p for p in self.net.parameters() if id(p) not in enc], "lr": self.lr}]
        opt = torch.optim.AdamW(groups, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(sids) / self.batch))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.15)
        al = torch.tensor(self.alphas, dtype=torch.float32, device=dev)
        tw = torch.tensor([0.6, 0.4], device=dev)
        self.net.train()
        for _ in range(self.epochs):
            perm = rng.permutation(len(sids))
            for b0 in range(0, len(perm), self.batch):
                idx = perm[b0:b0 + self.batch]
                x, mk, sc, bm, Rc = self._bag(sids[idx], rng, True)
                q, aux, seg = self.net(x.to(dev), mk.to(dev), sc.to(dev), bm.to(dev), demo[idx].to(dev))
                lq = (pinball(q, Yn[idx].to(dev), al) * tw).sum(dim=1)
                ys = ((self.st.yseg[Rc] - y_mu_t) / y_sd_t).to(dev)                     # (B, S, 2)
                bmd = bm.to(dev)
                ls = (F.smooth_l1_loss(seg, torch.nan_to_num(ys), reduction="none") * tw).sum(dim=2)
                ls = (ls * bmd).sum(dim=1) / bmd.sum(dim=1).clamp(min=1)
                la = F.smooth_l1_loss(aux[:, :2], aux_reg[idx].to(dev), reduction="none").sum(dim=1) + \
                    F.binary_cross_entropy_with_logits(aux[:, 2:], aux_bin[idx].to(dev), reduction="none").sum(dim=1)
                ww = wt[idx].to(dev)
                loss = ((lq + self.seg_w * ls + self.aux_w * la) * ww).sum() / ww.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                opt.step()
                sch.step()
        return self

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_all(self, X):
        dev = device()
        self.net.eval()
        sids = np.asarray(X.index)
        demo = self._demo(X)
        rng = np.random.default_rng(12345)
        qs, auxs, segs = [], [], []
        for _ in range(self.n_eval):
            qq, aa, ss = [], [], []
            for b0 in range(0, len(sids), 64):
                x, mk, sc, bm, _ = self._bag(sids[b0:b0 + 64], rng, False)
                q, aux, seg = self.net(x.to(dev), mk.to(dev), sc.to(dev), bm.to(dev), demo[b0:b0 + 64].to(dev))
                qq.append(q.cpu().numpy()); aa.append(aux.cpu().numpy()); ss.append(seg.cpu().numpy())
            qs.append(np.concatenate(qq)); auxs.append(np.concatenate(aa)); segs.append(np.concatenate(ss))
        q = np.mean(qs, 0) * self.y_sd[None, :, None] + self.y_mu[None, :, None]
        seg = np.mean(segs, 0) * self.y_sd + self.y_mu                                  # (n, S, 2) mmHg
        return np.sort(q, axis=2), np.mean(auxs, 0), seg

    def predict_quantiles(self, X):
        return self.predict_all(X)[0]

    def predict(self, X):
        return self.predict_quantiles(X)[:, :, int(np.argmin(np.abs(self.alphas - 0.5)))]

    def predict_seg(self, X) -> pd.DataFrame:
        """Per-round deviation of the round head from its subject's mean, indexed by seg_id."""
        _, _, seg = self.predict_all(X)
        R, bm = self._rows(np.asarray(X.index))
        out_ids, out = [], []
        for i in range(len(R)):
            v = seg[i][bm[i]]
            out.append(v - v.mean(axis=0))
            out_ids += list(self.st.seg_ids[R[i][bm[i]]])
        return pd.DataFrame(np.concatenate(out), index=out_ids, columns=["d_sbp", "d_dbp"])
