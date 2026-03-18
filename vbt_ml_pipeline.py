#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vbt_ml_pipeline.py — ML lifecycle pipeline (numpy only).

Commands:
  build-dataset   build-dataset --db squat_gym.db --out ml_data/
  train-fatigue   train-fatigue --data ml_data/ --out models/
  train-technique train-technique --data ml_data/ --out models/
  evaluate        evaluate --db squat_gym.db
  report          report --db squat_gym.db --out ml_data/metrics.json
"""
from __future__ import annotations
import argparse, json, logging, os, sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("vbt_ml_pipeline")
FAT_DIM, TECH_DIM = 8, 16


def build_dataset(db_path: str, out_dir: str) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT session_id,set_number,user_name,v_mean,rom,rom_completion_pct,"
        "velocity_loss,calib_is_fallback,pose_issues "
        "FROM reps WHERE v_mean IS NOT NULL AND v_mean>0.02 "
        "ORDER BY session_id,set_number,ts"
    )
    groups: dict = {}
    for r in cur.fetchall():
        groups.setdefault((r["session_id"] or "unk", r["set_number"] or 0), []).append(dict(r))
    X_fat, y_fat, meta_fat, trust_fat = [], [], [], []
    for (sid, sno), reps in groups.items():
        if len(reps) < 1: continue  # accept single-rep sets for bootstrap
        vels = [float(r["v_mean"]) for r in reps]
        best = max(vels)
        if best <= 0: continue
        seq = []
        for r in reps:
            v = float(r["v_mean"]); loss = (best-v)/best*100 if best>0 else 0
            seq.append([v, v, float(r["rom"] or 0), loss, 0.5, 0.0,
                        0.3 if r["pose_issues"] else 0.0,
                        1.0-int(r["calib_is_fallback"] or 0)])
        fat_lbl = float(np.clip((best-vels[-1])/best, 0, 1))
        stop_lbl = 1 if (best-vels[-1])/best > 0.20 else 0
        X_fat.append(seq); y_fat.append([fat_lbl, float(stop_lbl)])
        meta_fat.append({"session_id": sid, "set_number": sno})
        trust_fat.append(0.7 if any(r.get("calib_is_fallback") for r in reps) else 1.0)
    cur.execute(
        "SELECT v_mean,left_knee_angle,right_knee_angle,trunk_angle,rom "
        "FROM reps WHERE v_mean>0.02 AND left_knee_angle IS NOT NULL"
    )
    X_tech = []
    for r in cur.fetchall():
        v=float(r["v_mean"] or 0); lk=float(r["left_knee_angle"] or 90)
        rk=float(r["right_knee_angle"] or 90); tr=float(r["trunk_angle"] or 15)
        rm=float(r["rom"] or 0.3)
        X_tech.append(([v]*4+[lk,rk,(lk+rk)/2,abs(lk-rk)]+[tr]*4+[rm]*4)[:TECH_DIM])
    conn.close()
    with open(os.path.join(out_dir, "fatigue_sequences.json"), "w") as f:
        json.dump({"X": X_fat, "y": y_fat, "meta": meta_fat, "trust": trust_fat}, f)
    np.save(os.path.join(out_dir, "technique_X.npy"),
            np.array(X_tech, dtype=np.float32) if X_tech else np.zeros((0,TECH_DIM), np.float32))
    stats = {"fatigue_sets": len(X_fat), "technique_reps": len(X_tech),
             "db": db_path, "built_at": datetime.now().isoformat()}
    with open(os.path.join(out_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Dataset: %d sets, %d reps -> %s", len(X_fat), len(X_tech), out_dir)
    return stats


def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x.astype(float),-20,20)))
def _relu(x): return np.maximum(0.0, x)

def _split(X, y, frac=0.2, seed=42):
    rng=np.random.default_rng(seed); idx=rng.permutation(len(X)); sp=max(1,int(len(X)*frac))
    vi,ti=idx[:sp],idx[sp:]
    return [X[i] for i in ti],[X[i] for i in vi],[y[i] for i in ti],[y[i] for i in vi]

def _bce(p,t): e=1e-7; return float(-np.mean(t*np.log(p+e)+(1-t)*np.log(1-p+e)))

def _norm_fat(x):
    x=x.copy()
    for c,s in enumerate([1.5,2.0,0.6,50.0,3.0,20.0]):
        x[:,c]=np.clip(x[:,c]/s,0,1)
    return x


def _fat_w(seed=0):
    rng=np.random.default_rng(seed); w=lambda s: rng.normal(0,0.1,s).astype(np.float32)
    return {"conv_w":w((3,FAT_DIM,16)),"conv_b":np.zeros(16,np.float32),
            "fc1_w":w((16,8)),"fc1_b":np.zeros(8,np.float32),
            "fc2_w":w((8,2)),"fc2_b":np.zeros(2,np.float32)}

def _fat_fwd(x, wts):
    K=wts["conv_w"].shape[0]; T=x.shape[0]
    if T<K: x=np.pad(x,((K-T,0),(0,0)),mode="edge"); T=x.shape[0]
    co=np.array([np.einsum("ki,kio->o",x[t:t+K],wts["conv_w"])+wts["conv_b"]
                 for t in range(T-K+1)],dtype=np.float32)
    pool=_relu(co).mean(0); h1p=pool@wts["fc1_w"]+wts["fc1_b"]; h1=_relu(h1p)
    return _sig(h1@wts["fc2_w"]+wts["fc2_b"]),(pool,h1p,h1)


def train_fatigue_net(data_dir: str, out_dir: str, epochs: int=40, lr: float=0.01) -> dict:
    fp=os.path.join(data_dir,"fatigue_sequences.json")
    if not os.path.exists(fp): log.error("Missing %s",fp); return {}
    with open(fp) as f: data=json.load(f)
    X_all,y_all,trust=data["X"],data["y"],data.get("trust",[])
    n_total = len(X_all)
    # High-trust sets: calib_is_fallback==False for all reps (trust==1.0)
    n_high_trust = sum(1 for t in trust if t and float(t) >= 1.0) if trust else n_total
    n_usable = n_total

    # ── Tiered training strategy ──
    if n_usable < 10:
        msg = (f"数据不足: 仅有 {n_usable} sets (需要 ≥10，推荐 ≥50)。"
               " 继续使用规则回退，待更多训练数据后重试。")
        log.warning(msg)
        return {"error": "insufficient_data", "n": n_usable,
                "message": msg, "training_tier": "none"}

    if n_usable >= 50:
        training_tier = "production"
        log.info("Training tier: PRODUCTION (%d sets)", n_usable)
    else:
        training_tier = "bootstrap"
        log.info("Training tier: BOOTSTRAP (%d sets, <50 — low confidence)", n_usable)

    if len(X_all) < 4: return {"error":"insufficient_data","n":len(X_all)}
    X_tr,X_val,y_tr,y_val=_split(X_all,y_all)
    wts=_fat_w(); best=float("inf"); bwts={k:v.copy() for k,v in wts.items()}
    for ep in range(epochs):
        for i in np.random.permutation(len(X_tr)):
            x=_norm_fat(np.array(X_tr[i],dtype=np.float32))
            tgt=np.array(y_tr[i],dtype=np.float32)
            pred,(pool,h1p,h1)=_fat_fwd(x,wts)
            d=(pred-tgt)/max(1,len(tgt))
            wts["fc2_b"]-=lr*d; wts["fc2_w"]-=lr*np.outer(h1,d)
            dh1=(d@wts["fc2_w"].T)*(h1p>0)
            wts["fc1_b"]-=lr*dh1; wts["fc1_w"]-=lr*np.outer(pool,dh1)
        vl=float(np.mean([_bce(_fat_fwd(_norm_fat(np.array(X_val[i],dtype=np.float32)),wts)[0],
                               np.array(y_val[i],dtype=np.float32)) for i in range(len(X_val))]))
        if vl<best: best=vl; bwts={k:v.copy() for k,v in wts.items()}
        if (ep+1)%10==0: log.info("fat %d/%d val_bce=%.4f",ep+1,epochs,vl)
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    # Versioned filename: avoid overwriting previous models
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    versioned_name = f"set_fatigue_net_{ts}.npz"
    op = os.path.join(out_dir, versioned_name)
    np.savez(op,**bwts)
    # Also write canonical symlink/copy for backward compat
    canonical = os.path.join(out_dir, "set_fatigue_net.npz")
    import shutil; shutil.copy2(op, canonical)
    # Write metadata
    meta = {
        "model_name": "SetFatigueNet", "version": ts,
        "training_tier": training_tier, "val_bce": float(best),
        "n_train": len(X_tr), "n_val": len(X_val),
        "n_total_sets": n_total, "n_high_trust_sets": n_high_trust,
        "versioned_path": op, "canonical_path": canonical,
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, f"set_fatigue_net_{ts}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("SetFatigueNet [%s] -> %s (val_bce=%.4f)", training_tier, op, best)
    log.info("  samples: total=%d high_trust=%d train=%d val=%d",
             n_total, n_high_trust, len(X_tr), len(X_val))
    return meta


def _tech_w(seed=1):
    rng=np.random.default_rng(seed); w=lambda s: rng.normal(0,0.1,s).astype(np.float32)
    return {"enc1_w":w((TECH_DIM,8)),"enc1_b":np.zeros(8,np.float32),
            "enc2_w":w((8,4)),"enc2_b":np.zeros(4,np.float32),
            "dec1_w":w((4,8)),"dec1_b":np.zeros(8,np.float32),
            "dec2_w":w((8,TECH_DIM)),"dec2_b":np.zeros(TECH_DIM,np.float32)}

def _tech_fwd(x,wts):
    h=_relu(x@wts["enc1_w"]+wts["enc1_b"]); z=_relu(h@wts["enc2_w"]+wts["enc2_b"])
    return _relu(z@wts["dec1_w"]+wts["dec1_b"])@wts["dec2_w"]+wts["dec2_b"]

def train_technique_net(data_dir: str, out_dir: str, epochs: int=60, lr: float=0.005) -> dict:
    fp=os.path.join(data_dir,"technique_X.npy")
    if not os.path.exists(fp): log.error("Missing %s",fp); return {}
    Xr=np.load(fp)
    n_total = len(Xr)

    # ── Tiered training strategy ──
    if n_total < 15:
        msg = (f"数据不足: 仅有 {n_total} reps (需要 ≥15，推荐 ≥50)。"
               " 继续使用规则回退。")
        log.warning(msg)
        return {"error": "insufficient_data", "n": n_total,
                "message": msg, "training_tier": "none"}

    if n_total >= 50:
        training_tier = "production"
        log.info("Technique tier: PRODUCTION (%d reps)", n_total)
    else:
        training_tier = "bootstrap"
        log.info("Technique tier: BOOTSTRAP (%d reps, <50)", n_total)

    if len(Xr)<4: return {"error":"insufficient_data","n":len(Xr)}
    mu,sd=Xr.mean(0),Xr.std(0)+1e-8
    X=((Xr-mu)/sd).astype(np.float32); sp=max(1,int(len(X)*0.2)); Xv,Xt=X[:sp],X[sp:]
    wts=_tech_w(); best=float("inf"); bwts={k:v.copy() for k,v in wts.items()}
    for ep in range(epochs):
        for i in np.random.permutation(len(Xt)):
            x=Xt[i]; d=_tech_fwd(x,wts)-x
            h2=_relu(_relu(_relu(x@wts["enc1_w"]+wts["enc1_b"])@wts["enc2_w"]+wts["enc2_b"])@wts["dec1_w"]+wts["dec1_b"])
            wts["dec2_b"]-=lr*d; wts["dec2_w"]-=lr*np.outer(h2,d)
        vl=float(np.mean([(Xv[i]-_tech_fwd(Xv[i],wts))**2 for i in range(len(Xv))]))
        if vl<best: best=vl; bwts={k:v.copy() for k,v in wts.items()}
        if (ep+1)%20==0: log.info("tech %d/%d val_mse=%.6f",ep+1,epochs,vl)
    errs=[float(np.mean((Xt[i]-_tech_fwd(Xt[i],bwts))**2)) for i in range(len(Xt))]
    bl=float(np.percentile(errs,90)) if errs else 1.0
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    versioned_name = f"technique_anomaly_net_{ts}.npz"
    op = os.path.join(out_dir, versioned_name)
    np.savez(op,baseline_error=np.float32(bl),**bwts)
    # Canonical copy for backward compat
    canonical = os.path.join(out_dir, "technique_anomaly_net.npz")
    import shutil; shutil.copy2(op, canonical)
    meta = {
        "model_name": "TechniqueAnomalyNet", "version": ts,
        "training_tier": training_tier, "val_mse": float(best),
        "baseline_error": bl, "n_train": len(Xt), "n_val": len(Xv),
        "n_total_reps": n_total,
        "versioned_path": op, "canonical_path": canonical,
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, f"technique_anomaly_net_{ts}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("TechniqueAnomalyNet [%s] -> %s (val_mse=%.6f, baseline=%.4f)",
             training_tier, op, best, bl)
    return meta


def evaluate(db_path: str) -> dict:
    if not os.path.exists(db_path): return {"error":"db_not_found"}
    conn=sqlite3.connect(db_path); cur=conn.cursor()
    m: dict = {"db":db_path,"evaluated_at":datetime.now().isoformat()}
    def _q(sql, default=None):
        try: cur.execute(sql); r=cur.fetchone(); return r[0] if r and r[0] is not None else default
        except Exception: return default
    total = _q("SELECT COUNT(*) FROM reps", 1) or 1
    fb    = _q("SELECT COUNT(*) FROM reps WHERE calib_is_fallback=1", 0)
    m["calib_fallback_rate"] = round((fb or 0)/total, 4)
    try:
        cur.execute("SELECT COUNT(*) FROM prediction_logs WHERE used_model=1")
        dl_count = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM prediction_logs")
        all_pred = cur.fetchone()[0] or 1
        m["dl_usage_rate"] = round(dl_count/all_pred, 4)
        m["fallback_rate"]  = round(1.0 - dl_count/all_pred, 4)
    except Exception:
        m["dl_usage_rate"] = None; m["fallback_rate"] = None
    try:
        cur.execute(
            "SELECT COUNT(*) FROM user_feedback f "
            "JOIN prediction_logs p ON f.set_id=p.set_id "
            "WHERE f.rpe IS NOT NULL AND p.stop_probability IS NOT NULL")
        labelled = cur.fetchone()[0] or 0
        m["labelled_feedback_count"] = labelled
        if labelled >= 5:
            cur.execute(
                "SELECT p.stop_probability, CASE WHEN f.rpe>=9 THEN 1 ELSE 0 END "
                "FROM user_feedback f "
                "JOIN prediction_logs p ON f.set_id=p.set_id "
                "WHERE f.rpe IS NOT NULL AND p.stop_probability IS NOT NULL")
            rows = cur.fetchall()
            probs = np.array([r[0] for r in rows], dtype=float)
            labels = np.array([r[1] for r in rows], dtype=float)
            preds_bin = (probs >= 0.5).astype(float)
            tp = float(np.sum((preds_bin == 1) & (labels == 1)))
            fp = float(np.sum((preds_bin == 1) & (labels == 0)))
            fn = float(np.sum((preds_bin == 0) & (labels == 1)))
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            m["stop_precision"] = round(prec, 4)
            m["stop_recall"]    = round(rec, 4)
    except Exception:
        m["labelled_feedback_count"] = 0
    conn.close()
    return m


def report(db_path: str, out_path: str) -> dict:
    m = evaluate(db_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    log.info("Metrics report -> %s", out_path)
    return m


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="VBT ML Pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    bd = sub.add_parser("build-dataset")
    bd.add_argument("--db",  default="squat_gym.db")
    bd.add_argument("--out", default="ml_data")

    tf = sub.add_parser("train-fatigue")
    tf.add_argument("--data",   default="ml_data")
    tf.add_argument("--out",    default="models")
    tf.add_argument("--epochs", type=int, default=40)
    tf.add_argument("--lr",     type=float, default=0.01)

    tt = sub.add_parser("train-technique")
    tt.add_argument("--data",   default="ml_data")
    tt.add_argument("--out",    default="models")
    tt.add_argument("--epochs", type=int, default=60)
    tt.add_argument("--lr",     type=float, default=0.005)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--db", default="squat_gym.db")

    rp = sub.add_parser("report")
    rp.add_argument("--db",  default="squat_gym.db")
    rp.add_argument("--out", default="ml_data/metrics.json")

    args = p.parse_args()
    if args.cmd == "build-dataset":
        build_dataset(args.db, args.out)
    elif args.cmd == "train-fatigue":
        train_fatigue_net(args.data, args.out, args.epochs, args.lr)
    elif args.cmd == "train-technique":
        train_technique_net(args.data, args.out, args.epochs, args.lr)
    elif args.cmd == "evaluate":
        m = evaluate(args.db)
        print(json.dumps(m, indent=2))
    elif args.cmd == "report":
        report(args.db, args.out)


if __name__ == "__main__":
    main()