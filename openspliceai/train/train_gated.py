"""
Train a gated-structure OpenSpliceAI model (NEW FILE — no existing OSAI file edited).

Self-contained train loop that IMPORTS OSAI utilities (load_data_from_shard, clip_datapoints) + the new
GatedSpliceAI + weighted losses. Model selection by lowest VALIDATION loss (-> model_best.pt), matching OSAI.

Backbone == a 4ch vanilla SpliceAI (optionally warm-started from a 4ch checkpoint via --init-4ch); the gated
structure correction is zero-initialised so training starts exactly as the 4ch model and can only earn
structure usage. With --no-gated it trains a stock 4ch SpliceAI on channels[:4] (the MATCHED baseline).

Example (cluster):
  python -m openspliceai.train.train_gated --gated \
    --train-dataset .../MANE_12ch/dataset_train.h5 --val-dataset .../MANE_12ch/dataset_validation.h5 \
    --flanking-size 10000 --epochs 30 --scheduler CosineAnnealingWarmRestarts --random-seed 1 \
    --init-4ch .../vanilla/cosine_s1/.../model_best.pt --struct-l2 0.0 --out-dir runs_gated/s1
"""
import argparse, os, sys, time, numpy as np, torch, h5py
from openspliceai.train_base.openspliceai import SpliceAI, GatedSpliceAI
from openspliceai.train_base.utils import (load_data_from_shard, clip_datapoints,
                                           categorical_crossentropy_2d, focal_loss)
from openspliceai.constants import SL, CL_max

def arch(flank):
    if flank == 80:    W=[11]*4;                         AR=[1]*4
    elif flank == 400: W=[11]*8;                         AR=[1]*4+[4]*4
    elif flank == 2000:W=[11]*8+[21]*4;                  AR=[1]*4+[4]*4+[10]*4
    elif flank ==10000:W=[11]*8+[21]*4+[41]*4;           AR=[1]*4+[4]*4+[10]*4+[25]*4
    else: raise ValueError(flank)
    return 32, np.asarray(W), np.asarray(AR)

def make_model(a, device):
    L, W, AR = arch(a.flanking_size)
    if a.gated:
        m = GatedSpliceAI(L, W, AR, apply_softmax=True, in_channels=a.in_channels, n_seq_channels=a.n_seq_channels)
        if a.init_4ch:
            sd = torch.load(a.init_4ch, map_location="cpu"); sd = sd.get("model_state_dict", sd)
            print("warm-start backbone from 4ch:", m.load_backbone(sd, strict=False))
    else:
        m = SpliceAI(L, W, AR, apply_softmax=True)         # stock 4ch matched baseline
    CL = int(2*np.sum(AR*(W-1)))
    return m.to(device), CL

def lossfn(name):
    return focal_loss if name == "focal_loss" else categorical_crossentropy_2d

def run_epoch(model, h5f, idxs, a, CL, device, crit, opt=None, wh5=None):
    train = opt is not None; model.train(train); tot=0.0; n=0; gate_acc=0.0; gb=0
    for si in idxs:
        dl = load_data_from_shard(h5f, si, device, a.batch_size, {}, shuffle=train)
        Wt = torch.tensor(wh5[f"W{si}"][:], dtype=torch.float32) if wh5 is not None else None
        for bi, (X, Y) in enumerate(dl):
            X = X[:, :a.in_channels, :] if not a.gated else X
            if not a.gated: X = X[:, :a.n_seq_channels, :]
            X, Y = clip_datapoints(X.to(device), Y.to(device), CL, CL_max, 1)
            with torch.set_grad_enabled(train):
                yp = model(X)
                w = None
                if Wt is not None:
                    wb = Wt[bi*a.batch_size:bi*a.batch_size+X.shape[0]].to(device)
                    w = wb if wb.shape[0]==X.shape[0] else None
                loss = crit(Y, yp, weights=w)
                if a.struct_l2 and getattr(model, "last_corr", None) is not None:
                    loss = loss + a.struct_l2 * model.struct_l2()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())*X.shape[0]; n += X.shape[0]
            if getattr(model, "last_gate", None) is not None:
                gate_acc += float(model.last_gate.mean().detach()); gb += 1
    return tot/max(n,1), (gate_acc/gb if gb else float("nan"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dataset", required=True)
    p.add_argument("--val-dataset", default=None, help="validation h5; if omitted, hold out last ~10%% of train shards")
    p.add_argument("--flanking-size", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--scheduler", default="CosineAnnealingWarmRestarts")
    p.add_argument("--loss", default="cross_entropy_loss")
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--gated", action="store_true"); p.add_argument("--no-gated", dest="gated", action="store_false")
    p.add_argument("--in-channels", type=int, default=12); p.add_argument("--n-seq-channels", type=int, default=4)
    p.add_argument("--struct-l2", type=float, default=0.0)
    p.add_argument("--init-4ch", default=None)
    p.add_argument("--posweights", default=None, help="sidecar h5 of W{shard} (n,SL) per-position weights")
    p.add_argument("--batch-size", type=int, default=36)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    torch.manual_seed(a.random_seed); np.random.seed(a.random_seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, CL = make_model(a, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sch = (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=1, eta_min=1e-5)
           if a.scheduler == "CosineAnnealingWarmRestarts"
           else torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[a.epochs-i for i in range(1,6)], gamma=0.5))
    crit = lossfn(a.loss)
    tr = h5py.File(a.train_dataset, "r")
    all_tr = sorted(int(k[1:]) for k in tr if k.startswith("X"))
    if a.val_dataset and a.val_dataset.lower() != "none":
        va = h5py.File(a.val_dataset, "r"); va_sep = True
        tr_idx = all_tr; va_idx = sorted(int(k[1:]) for k in va if k.startswith("X"))
    else:                                  # no separate val file -> hold out the last ~10% of train shards
        k = max(1, len(all_tr) // 10); tr_idx, va_idx = all_tr[:-k], all_tr[-k:]; va = tr; va_sep = False
        print(f"no --val-dataset -> validation = last {len(va_idx)}/{len(all_tr)} train shards", flush=True)
    wh5 = h5py.File(a.posweights, "r") if a.posweights else None
    print(f"device={device} gated={a.gated} CL={CL} train_shards={len(tr_idx)} val_shards={len(va_idx)}")
    best = np.inf
    for ep in range(a.epochs):
        t0=time.time()
        np.random.shuffle(tr_idx)
        trl, trg = run_epoch(model, tr, tr_idx, a, CL, device, crit, opt=opt, wh5=wh5)
        val, vg = run_epoch(model, va, va_idx, a, CL, device, crit, opt=None, wh5=None)
        sch.step()
        tag = "  *best" if val < best else ""
        if val < best:
            best = val; torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "val_loss": val,
                                     "gated": a.gated, "args": vars(a)}, os.path.join(a.out_dir, "model_best.pt"))
        print(f"ep{ep:02d} train_loss={trl:.4f} val_loss={val:.4f} gate={trg:.3f} {time.time()-t0:.0f}s{tag}", flush=True)
    tr.close();  va_sep and va.close();  wh5 and wh5.close()
    print("done. best val_loss=%.4f -> %s/model_best.pt" % (best, a.out_dir))

if __name__ == "__main__":
    main()
