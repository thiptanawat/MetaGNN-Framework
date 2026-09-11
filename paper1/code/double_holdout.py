#!/usr/bin/env python3
"""Full-scale double hold-out evaluation (patients x reactions) on the rebuilt TCGA-CRC 624 features.
Designed for one GPU (RTX 5090 / H100). Restart-safe: every (model, lambda, pfold, rfold) cell writes its own
result file and is skipped when present.

Usage (from the data record v2 root, i.e. the directory containing crc_624/):
  python double_holdout.py --data crc_624 --out results_double_holdout --device cuda \
      --models gnn_B:0.0 gnn_B:0.2 gnn_A:0.0 mlp:0.0 mlp_emb:0.0 --pfolds 5 --rfolds 3
Each cell: train on train-patients x train-reactions (BCE on train reactions only + lambda * mass-balance),
select on val-patients x train-reactions AUROC (patience 15, max 80 epochs), evaluate on test-patients x
held-out reactions with MC-Dropout T=30, save per-patient predictions (scores, sigma) and the diagnostics.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', default='results_double_holdout'); ap.add_argument('--device', default='cuda')
ap.add_argument('--models', nargs='+', default=['gnn_B:0.0', 'gnn_B:0.2', 'gnn_A:0.0', 'mlp:0.0', 'mlp_emb:0.0'])
ap.add_argument('--pfolds', type=int, default=5); ap.add_argument('--rfolds', type=int, default=3); ap.add_argument('--epochs', type=int, default=80)
ap.add_argument('--patience', type=int, default=15); ap.add_argument('--T', type=int, default=30); ap.add_argument('--seed', type=int, default=2024)
ap.add_argument('--n_patients', type=int, default=0, help='0 = all patients; smaller for smoke tests')
ap.add_argument('--preset', choices=['core', 'full'], default=None, help='core = gnn_B:0.0 mlp:0.0 mlp_emb:0.0 (answers the main question); full = adds gnn_B:0.2 gnn_A:0.0')
ap.add_argument('--preflight', action='store_true', help='check environment + data, time a few steps, print an ETA, then exit')
A = ap.parse_args()
if A.preset == 'core': A.models = ['gnn_B:0.0', 'mlp:0.0', 'mlp_emb:0.0']
elif A.preset == 'full': A.models = ['gnn_B:0.0', 'gnn_B:0.2', 'gnn_A:0.0', 'mlp:0.0', 'mlp_emb:0.0']
os.makedirs(A.out, exist_ok=True); dev = torch.device(A.device if torch.cuda.is_available() or A.device == 'cpu' else 'cpu')
N_RXN = 10600; FEAT = os.path.join(A.data, 'reaction_features')
ARCH = {'A': dict(hidden=128, layers=2, heads=4), 'B': dict(hidden=256, layers=3, heads=8)}

class ReactionMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2, emb=0):
        super().__init__(); self.emb = nn.Embedding(N_RXN, emb) if emb else None
        d = in_dim + emb; self.net = nn.Sequential(nn.Linear(d, hidden), nn.ELU(), T.MCDropout(dropout), nn.Linear(hidden, hidden // 2), nn.ELU(), T.MCDropout(dropout), nn.Linear(hidden // 2, 1), nn.Sigmoid())
    def forward(self, x_dict, edge_index_dict=None):
        x = x_dict['reaction']
        if self.emb is not None: x = torch.cat([x, self.emb.weight], 1)
        return self.net(x).squeeze(-1)
def fwd(model, b): return model(x_dict={'reaction': b['reaction'].x, 'metabolite': b['metabolite'].x}, edge_index_dict={r: b[r].edge_index for r in b.edge_types})
def ds(ids): torch.manual_seed(A.seed); return T.UnifiedDataset(A.data, ids, 'cohort_220', bipartite_only=False, feature_override_dir=FEAT)
def build(name, sample):
    kind = name.split('_')[0]
    if kind == 'gnn':
        a = ARCH[name.split('_')[1]]
        return T.MetaGNN(rxn_in_dim=sample['reaction'].x.shape[1], met_in_dim=sample['metabolite'].x.shape[1], hidden_dim=a['hidden'], n_layers=a['layers'], heads=a['heads'], dropout=0.2, edge_types=list(sample.edge_types))
    return ReactionMLP(sample['reaction'].x.shape[1], emb=(64 if name == 'mlp_emb' else 0))
def auc_masked(model, dset, mask):
    model.eval(); ps = []; ys = []
    with torch.no_grad():
        for i in range(len(dset)): b = dset[i].to(dev); ps.append(fwd(model, b).cpu().numpy()[mask]); ys.append(b['reaction'].y.cpu().numpy()[mask])
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

meta = pd.read_csv(os.path.join(A.data, 'clinical_metadata.tsv'), sep='\t'); ids = meta['tcga_barcode'].tolist(); proj = meta['project'].fillna('U').tolist()
if A.n_patients: ids, _, proj, _ = train_test_split(ids, proj, train_size=A.n_patients, random_state=A.seed, stratify=proj)
pk = StratifiedKFold(A.pfolds, shuffle=True, random_state=A.seed); pfolds = []
for tr, te in pk.split(ids, proj):
    tr_ids = [ids[i] for i in tr]; trn, val = train_test_split(tr_ids, test_size=0.15, random_state=A.seed); pfolds.append(dict(train=trn, val=val, test=[ids[i] for i in te]))
T_START = time.time()
import h5py
with h5py.File(os.path.join(A.data, 'recon3d_stoich.h5')) as h: S = torch.tensor(h['S'][:], dtype=torch.float32).to(dev)
y_all = np.asarray(torch.load(os.path.join(A.data, 'activity_pseudolabels.pt'), map_location='cpu', weights_only=False)).astype(int)
probe = ds(pfolds[0]['train'][:20]); has = np.stack([probe[i]['reaction'].x[:, 0].numpy() for i in range(len(probe))]).max(0) > 0
rk = StratifiedKFold(A.rfolds, shuffle=True, random_state=A.seed); rfolds = list(rk.split(np.zeros(N_RXN), 2 * y_all + has.astype(int)))

if A.preflight:
    print(f'device            : {dev} ({torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"})')
    print(f'patients          : {len(ids)} -> per fold ~{len(pfolds[0]["train"])} train / {len(pfolds[0]["val"])} val / {len(pfolds[0]["test"])} test')
    print(f'reactions         : {N_RXN}, {A.rfolds} folds -> ~{len(rfolds[0][1])} held out per fold')
    print(f'labels active     : {y_all.mean():.4f}; expression-bearing reactions: {int(has.sum())}')
    bench = ds(pfolds[0]['train'][:6]); b0 = bench[0].to(dev); bce0 = nn.BCELoss()
    for spec in A.models:
        name, lam = spec.split(':'); lam = float(lam)
        m = build(name, bench[0]).to(dev); o = torch.optim.AdamW(m.parameters(), lr=1e-3); ts = []
        for k in range(6):
            if dev.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time(); o.zero_grad(); s_ = fwd(m, b0); l = bce0(s_, b0['reaction'].y.float())
            if lam > 0: l = l + lam * (S @ s_).pow(2).mean()
            l.backward(); o.step()
            if dev.type == 'cuda': torch.cuda.synchronize()
            ts.append(time.time() - t0)
        step = float(np.median(ts[2:])); ntr = len(pfolds[0]['train']); nva = len(pfolds[0]['val']); nte = len(pfolds[0]['test'])
        ep_assumed = min(A.epochs, 40)
        per_cell = ep_assumed * (ntr * step + nva * step / 3) + nte * A.T * step / 3
        cells = A.pfolds * A.rfolds
        print(f'  {name:8s} lam={lam}: {sum(p.numel() for p in m.parameters()):>9,} params | {step*1000:6.1f} ms/patient-step | ~{per_cell/60:5.1f} min/cell x {cells} cells = {per_cell*cells/3600:5.2f} h')
        del m, o
    print('\nTotal above assumes ~40 epochs before early stopping; restart-safe, so you can stop and resume anytime.')
    sys.exit(0)

for spec in A.models:
    name, lam = spec.split(':'); lam = float(lam)
    for pf, sp in enumerate(pfolds):
        tr_ds = ds(sp['train']); va_ds = ds(sp['val']); te_ds = ds(sp['test'])
        for rf, (rtr, rte) in enumerate(rfolds):
            tag = f'{name}_mb{lam}_p{pf}_r{rf}'; rfile = os.path.join(A.out, tag + '.json')
            if os.path.exists(rfile): continue
            mtr = torch.zeros(N_RXN, dtype=torch.bool); mtr[rtr] = True; mte = ~mtr; mtr_d = mtr.to(dev)
            torch.manual_seed(A.seed + pf * 100 + rf); model = build(name, tr_ds[0]).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5); bce = nn.BCELoss()
            best, best_state, pat, t0 = -1, None, 0, time.time()
            for ep in range(1, A.epochs + 1):
                model.train(); g = torch.Generator().manual_seed(A.seed + pf * 1000 + rf * 100 + ep)
                for i in torch.randperm(len(tr_ds), generator=g).tolist():
                    b = tr_ds[i].to(dev); opt.zero_grad(); s = fwd(model, b); loss = bce(s[mtr_d], b['reaction'].y[mtr_d].float())
                    if lam > 0: loss = loss + lam * (S @ s).pow(2).mean()
                    loss.backward(); opt.step()
                v = auc_masked(model, va_ds, mtr.numpy())
                if v > best + 1e-5: best, pat, best_state = v, 0, {k: x.detach().clone() for k, x in model.state_dict().items()}
                else: pat += 1
                if pat >= A.patience: break
            model.load_state_dict(best_state); model.train()  # dropout active for MC
            P, U = [], []
            with torch.no_grad():
                for i in range(len(te_ds)):
                    b = te_ds[i].to(dev); o = torch.stack([fwd(model, b) for _ in range(A.T)]); P.append(o.mean(0).cpu().numpy()); U.append(o.std(0).cpu().numpy())
            P = np.stack(P); U = np.stack(U); mt = mte.numpy(); yv = y_all
            Xte = np.stack([te_ds[i]['reaction'].x[:, 0].numpy() for i in range(len(te_ds))]); ind = (Xte.max(0) > 0).astype(float)
            res = dict(model=name, lambda_mb=lam, pfold=pf, rfold=rf, n_train=len(sp['train']), n_test=len(sp['test']), n_heldout_rxn=int(mt.sum()), epochs=ep, best_val=round(best, 4), minutes=round((time.time() - t0) / 60, 1),
                       heldout_AUROC_perpatient=round(float(np.mean([roc_auc_score(yv[mt], P[i, mt]) for i in range(len(P))])), 4),
                       heldout_AUROC_pooled=round(roc_auc_score(np.tile(yv[mt], len(P)), P[:, mt].reshape(-1)), 4),
                       heldout_exprbearing_AUROC=round(float(np.mean([roc_auc_score(yv[mt & (ind > 0)], P[i, mt & (ind > 0)]) for i in range(len(P))])), 4),
                       trainrxn_AUROC_perpatient=round(float(np.mean([roc_auc_score(yv[~mt], P[i, ~mt]) for i in range(len(P))])), 4),
                       baseline_rawexpr=round(float(np.mean([roc_auc_score(yv[mt], Xte[i, mt]) for i in range(len(P))])), 4),
                       baseline_indicator=round(float(roc_auc_score(yv[mt], ind[mt])), 4),
                       rho=round(float(P.std(0).mean() / (U.mean() / np.sqrt(A.T))), 4), interpatient_r_median=round(float(np.median(np.corrcoef(P)[np.triu_indices(len(P), 1)])), 4))
            np.savez_compressed(os.path.join(A.out, tag + '_preds.npz'), patient_ids=np.array(sp['test']), scores=P.astype(np.float32), uncertainties=U.astype(np.float32), heldout_mask=mt)
            json.dump(res, open(rfile, 'w'), indent=1); print(json.dumps(res), flush=True)
            done = len([f for f in os.listdir(A.out) if f.endswith('.json')]); total = len(A.models) * A.pfolds * A.rfolds
            el = (time.time() - T_START) / 3600
            print(f'  [{done}/{total} cells | {el:.2f} h elapsed | ETA {el/done*(total-done):.2f} h remaining]', flush=True)
print('ALL CELLS DONE')
