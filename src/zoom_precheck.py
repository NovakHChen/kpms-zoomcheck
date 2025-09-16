# (same script I sent you; trimmed to fit here)
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def load_dlc_h5(path, wanted_parts):
    df = pd.read_hdf(path)
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels >= 3:
        T = len(df); K = len(wanted_parts)
        arr = np.full((T, K, 3), np.nan, float)
        for k, bp in enumerate(wanted_parts):
            for j, c in enumerate(("x","y","likelihood")):
                hits = [col for col in df.columns if (len(col)>=3 and col[1]==bp and col[2]==c)]
                if hits: arr[:,k,j] = df[hits[0]].to_numpy()
        return arr
    # fallback for flat exports
    T = len(df); K = len(wanted_parts); arr = np.full((T,K,3), np.nan, float)
    cols = df.columns
    for k,bp in enumerate(wanted_parts):
        for j,c in enumerate(("x","y","likelihood")):
            if isinstance(cols, pd.MultiIndex) and cols.nlevels==2:
                key = (bp,c)
                if key in cols: arr[:,k,j] = df[key].to_numpy()
            else:
                for nm in (f"{bp}_{c}", f"{bp}{c}", f"{bp}:{c}", f"{bp}-{c}"):
                    if nm in cols: arr[:,k,j] = df[nm].to_numpy(); break
    return arr

def body_length(arr, i_head, i_tail, pcut):
    d = np.linalg.norm(arr[:, i_head, :2] - arr[:, i_tail, :2], axis=1)
    lh, lt = arr[:, i_head, 2], arr[:, i_tail, 2]
    good = (lh >= pcut) & (lt >= pcut) & np.isfinite(d)
    d[~good] = np.nan
    return d

def windowize(X, win, step):
    T = X.shape[0]
    starts = list(range(0, max(0, T - win + 1), step))
    if not starts: return np.empty((0,win,X.shape[1],2))
    return np.stack([X[s:s+win, :, :] for s in starts], axis=0)

def center_features(W):
    m = np.nanmean(W, axis=2, keepdims=True)
    Xc = W - m
    fill = np.nanmean(Xc, axis=(2,3), keepdims=True)
    Xc = np.where(np.isfinite(Xc), Xc, fill)
    return Xc.reshape(len(W), -1)

def main(args):
    parts = args.parts
    idx = {bp:i for i,bp in enumerate(parts)}
    head, tail = args.anterior, args.posterior

    data = {}
    for p in args.h5:
        data[Path(p).name] = load_dlc_h5(p, parts)

    # per-video summary
    rows = []
    for name, arr in data.items():
        d = body_length(arr, idx[head], idx[tail], args.pcutoff)
        rows.append(dict(
            video=name,
            frames=int(arr.shape[0]),
            valid_frames=int(np.isfinite(d).sum()),
            bodylen_median_px=float(np.nanmedian(d)),
            bodylen_MAD_px=float(np.nanmedian(np.abs(d - np.nanmedian(d)))),
            bodylen_CV=float(np.nanstd(d)/(np.nanmean(d)+1e-9))
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv("zoom_precheck_summary.csv", index=False)

    # KS distance between first two vids
    vids = list(data.keys())
    d1 = body_length(data[vids[0]], idx[head], idx[tail], args.pcutoff)
    d2 = body_length(data[vids[1]], idx[head], idx[tail], args.pcutoff)
    d1, d2 = d1[np.isfinite(d1)], d2[np.isfinite(d2)]
    ks_stat, ks_p = ks_2samp(d1, d2) if (len(d1)>0 and len(d2)>0) else (np.nan, np.nan)

    # zoom classifier
    Xs, ys = [], []
    for gi, (name, arr) in enumerate(data.items()):
        XY = arr[:, :, :2]
        W = windowize(XY, args.win, args.step)
        Xf = center_features(W)
        Xs.append(Xf); ys.append(np.full(len(Xf), gi, int))
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)

    accs, aucs, cms = [], [], []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        yhat = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
        try:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:,1]))
        except Exception:
            pass
        cms.append(confusion_matrix(y[te], yhat))
    cm = np.sum(np.stack(cms), axis=0)

    # save confusion
    plt.figure(figsize=(3,3))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Zoom confusion (pre)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout(); plt.savefig("zoom_confusion_pre.png", dpi=200); plt.close()

    print("\n=== DLC Zoom Precheck (before any normalization) ===")
    print(summary.to_string(index=False))
    print(f"\nKS(body-length px) {vids[0]} vs {vids[1]}: stat={ks_stat:.3f}, p={ks_p:.2e}")
    print(f"Zoom-classifier accuracy (5-fold CV): {np.mean(accs):.3f} (chance ≈ 0.50)")
    print(f"Zoom-classifier ROC AUC           : {np.mean(aucs) if aucs else float('nan'):.3f}")
    print("Saved: zoom_confusion_pre.png, zoom_precheck_summary.csv")

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", nargs="+", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--anterior", required=True)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--pcutoff", type=float, default=0.6)
    ap.add_argument("--win", type=int, default=60)
    ap.add_argument("--step", type=int, default=30)
    args = ap.parse_args()
    main(args)
