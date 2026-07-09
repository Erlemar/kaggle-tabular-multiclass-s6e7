"""
train_mlp.py — PyTorch MLP (categorical embeddings + standardized numerics) for tabular multiclass.

The most decorrelated blend member. Copy to experiments/mlp_v<N>/train.py and adapt.
Preprocessing (impute + standardize) is fit on the TRAIN FOLD ONLY — fitting on all rows leaks
the val distribution. Obeys the common.py contract.

    python templates/train_mlp.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C

TAG = "mlp_v1"
N_SEEDS = 1
MAX_EPOCHS = 60
PATIENCE = 8
BATCH = 512
LR = 2e-3
WEIGHT_DECAY = 1e-5
HIDDEN = (256, 128)
DROPOUT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_design():
    """Return (num_tr, num_te, cat_tr, cat_te, cardinalities). Categorical codes are factorized on
    train+test (unsupervised, no target -> no leak); code 0 reserved for NaN/unseen."""
    train, test = C.load_train(), C.load_test()
    ftr, fte = C.ART / "features_train.parquet", C.ART / "features_test.parquet"
    if ftr.exists() and fte.exists():
        Xtr = train[[C.ID]].merge(pd.read_parquet(ftr), on=C.ID, how="left").drop(columns=[C.ID])
        Xte = test[[C.ID]].merge(pd.read_parquet(fte), on=C.ID, how="left").drop(columns=[C.ID])
    else:
        print("!! features parquet not found — using raw columns. Build features first for a real run.")
        feat = [c for c in train.columns if c not in (C.ID, C.TARGET) and c in test.columns]
        Xtr, Xte = train[feat].copy(), test[feat].copy()

    cat_cols = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]

    cards = []
    cat_tr = np.zeros((len(Xtr), len(cat_cols)), dtype=np.int64)
    cat_te = np.zeros((len(Xte), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        codes, uniques = pd.factorize(pd.concat([Xtr[c], Xte[c]], ignore_index=True))
        codes = codes + 1  # 0 reserved for NaN/unseen
        cat_tr[:, j] = codes[:len(Xtr)]
        cat_te[:, j] = codes[len(Xtr):]
        cards.append(len(uniques) + 1)

    num_tr = Xtr[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    num_te = Xte[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return num_tr, num_te, cat_tr, cat_te, cards


class MLP(nn.Module):
    def __init__(self, cards, n_num, n_classes):
        super().__init__()
        dims = [(c, min(50, (c + 1) // 2)) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in dims])
        self.emb_drop = nn.Dropout(0.1)
        in_dim = sum(d for _, d in dims) + n_num
        layers, prev = [], in_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        if len(self.embs):
            e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
            x = torch.cat([self.emb_drop(e), x_num], dim=1)
        else:
            x = x_num
        return self.net(x)


def _standardize(num, tr_mask):
    med = np.nanmedian(num[tr_mask], axis=0)
    num = np.where(np.isnan(num), med, num)
    mu = num[tr_mask].mean(0)
    sd = num[tr_mask].std(0) + 1e-6
    return ((num - mu) / sd).astype(np.float32)


def main():
    C.set_seed()
    train, test = C.load_train(), C.load_test()
    labels = C.class_labels(train)
    y_idx = C.encode_target(train[C.TARGET].to_numpy(), labels)
    folds = C.fold_array(train)
    num_tr, num_te, cat_tr, cat_te, cards = build_design()
    n, m, ncls = len(train), len(test), len(labels)
    print(f"{TAG}: {num_tr.shape[1]} numeric + {len(cards)} categorical, {ncls} classes, dev={DEVICE}")

    oof = np.zeros((n, ncls))
    test_pred = np.zeros((m, ncls))
    n_folds = int(C.CFG["n_folds"])
    lower_better = not C.greater_is_better()
    xcat_te = torch.tensor(cat_te, dtype=torch.long, device=DEVICE)

    for seed in range(N_SEEDS):
        torch.manual_seed(C.SEED + seed)
        for f in range(n_folds):
            tr, va = folds != f, folds == f
            if not va.any():
                continue
            num = _standardize(np.vstack([num_tr, num_te]), np.r_[tr, np.zeros(m, bool)])
            num_tr_s, num_te_s = num[:n], num[n:]

            ds = TensorDataset(torch.tensor(cat_tr[tr]), torch.tensor(num_tr_s[tr]),
                               torch.tensor(y_idx[tr]))
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
            Xc_va = torch.tensor(cat_tr[va], device=DEVICE)
            Xn_va = torch.tensor(num_tr_s[va], device=DEVICE)

            model = MLP(cards, num_tr.shape[1], ncls).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            lossf = nn.CrossEntropyLoss()
            best, best_state, bad = (np.inf if lower_better else -np.inf), None, 0

            for epoch in range(MAX_EPOCHS):
                model.train()
                for xc, xn, yy in dl:
                    opt.zero_grad()
                    out = model(xc.to(DEVICE), xn.to(DEVICE))
                    loss = lossf(out, yy.to(DEVICE))
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    p = torch.softmax(model(Xc_va, Xn_va), dim=1).cpu().numpy()
                val = C.score(y_idx[va], p)
                improved = (val < best) if lower_better else (val > best)
                if improved:
                    best, best_state, bad = val, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                oof[va] += torch.softmax(model(Xc_va, Xn_va), 1).cpu().numpy() / N_SEEDS
                xn_te = torch.tensor(num_te_s, device=DEVICE)
                test_pred += torch.softmax(model(xcat_te, xn_te), 1).cpu().numpy() / (N_SEEDS * n_folds)
            torch.save(best_state, C.ART / "models" / f"{TAG}_s{seed}_f{f}.pt")
            print(f"  fold {f} best {C.METRIC}={best:.5f}")

    rep = C.cv_report(y_idx, oof, folds)
    print(f"{TAG} OOF {C.METRIC}: {rep['overall']:.5f}  (folds {rep['mean']:.5f}±{rep['std']:.5f})")

    C.save_oof(TAG, train[C.ID].to_numpy(), oof, labels,
               meta={"family": "mlp", "n_numeric": int(num_tr.shape[1]),
                     "n_categorical": len(cards), "hidden": list(HIDDEN), "cv": rep})
    C.save_test(TAG, test[C.ID].to_numpy(), test_pred, labels)
    C.log_experiment(TAG, "mlp", rep, n_feat=num_tr.shape[1] + len(cards),
                     notes=f"emb+MLP{list(HIDDEN)}, {N_SEEDS} seed(s)")


if __name__ == "__main__":
    main()
