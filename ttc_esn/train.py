"""Train and evaluate the ESN TTC predictor on features from ttc_esn.extract.

Evaluation is leave-one-sequence-out cross-validation (every sequence is a
test set once; another training sequence provides early stopping). Compared
methods, all evaluated at each horizon h on the same frames:
  * heuristic - legacy fused TTC(t) - h (constant closing speed),
  * expansion - 1 / expansion_rate(t) - h (least-squares looming, pure physics),
  * mlp       - the readout without reservoir (no temporal memory; ablation),
  * esn       - reservoir + MLP readout.
After CV, a final model is fit on all sequences for the median best epoch count.

    python -m ttc_esn.train --features outputs/features --out outputs/models
"""

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from collision_avoidance.telemetry import start_run

from .features import FEATURE_NAMES
from .model import EsnConfig, EsnTtcModel

MIN_TTC_S, MAX_TTC_S = 0.1, 30.0


# ----------------------------------------------------------------------------- data
def load_sequences(features_dir: Path) -> dict[str, dict]:
    return {p.stem: dict(np.load(p)) for p in sorted(features_dir.glob("*.npz"))}


def segments(seq: dict) -> list[slice]:
    """Contiguous runs of frames where the same target track has features."""
    present = ~np.isnan(seq["features"][:, 0])
    ids = np.where(present, seq["track_ids"], -1)
    runs, start = [], None
    for t in range(len(ids) + 1):
        boundary = t == len(ids) or not present[t] or (start is not None and ids[t] != ids[start])
        if start is not None and boundary:
            runs.append(slice(start, t))
            start = None
        if t < len(ids) and present[t] and start is None:
            start = t
    return runs


def rows(model: EsnTtcModel, seqs: dict[str, dict], names: list[str], washout: int) -> dict[str, np.ndarray]:
    """Stack readout inputs, labels and baseline inputs for the given sequences."""
    out = {k: [] for k in ("x", "y", "heuristic", "expansion", "sequence")}
    exp_idx = FEATURE_NAMES.index("expansion_rate")
    for name in names:
        seq = seqs[name]
        for seg in segments(seq):
            design = model.design_matrix(seq["features"][seg])
            keep = np.arange(design.shape[0]) >= washout
            labels = seq["labels"][seg][keep]
            out["x"].append(design[keep])
            out["y"].append(labels)
            out["heuristic"].append(seq["heuristic_ttc"][seg][keep])
            out["expansion"].append(seq["features"][seg][keep, exp_idx])
            out["sequence"].append(np.full(keep.sum(), name))
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}


def label_mask(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & (y > MIN_TTC_S) & (y < MAX_TTC_S)


# ----------------------------------------------------------------------------- training
def fit_readout(model: EsnTtcModel, train: dict, val: dict | None, epochs: int, lr: float, weight_decay: float,
                patience: int) -> dict:
    x = torch.from_numpy(train["x"])
    y = torch.from_numpy(np.log(np.clip(np.nan_to_num(train["y"], nan=1.0), MIN_TTC_S, None)).astype(np.float32))
    m = torch.from_numpy(label_mask(train["y"]).astype(np.float32))
    if val is not None:
        xv = torch.from_numpy(val["x"])
        yv = torch.from_numpy(np.log(np.clip(np.nan_to_num(val["y"], nan=1.0), MIN_TTC_S, None)).astype(np.float32))
        mv = torch.from_numpy(label_mask(val["y"]).astype(np.float32))

    def masked_mse(pred, target, mask):
        return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)

    opt = torch.optim.AdamW(model.readout.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": []}
    best = (float("inf"), 0, None)
    for epoch in range(epochs):
        model.readout.train()
        opt.zero_grad()
        loss = masked_mse(model.readout(x), y, m)
        loss.backward()
        opt.step()
        history["train"].append(float(loss))
        if val is not None:
            model.readout.eval()
            with torch.no_grad():
                val_loss = float(masked_mse(model.readout(xv), yv, mv))
            history["val"].append(val_loss)
            if val_loss < best[0]:
                best = (val_loss, epoch, {k: v.clone() for k, v in model.readout.state_dict().items()})
            elif epoch - best[1] >= patience:
                break
    if val is not None:
        model.readout.load_state_dict(best[2])
    history["best_epoch"] = best[1] if val is not None else epochs - 1
    return history


# ----------------------------------------------------------------------------- evaluation
def predictions(model: EsnTtcModel, data: dict, horizons: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore"):
        heuristic = data["heuristic"][:, None] - horizons[None]
        expansion = np.where(data["expansion"] > 0, 1.0 / data["expansion"], np.nan)[:, None] - horizons[None]
    return {"learned": np.exp(model.predict_log_ttc(data["x"])), "heuristic": heuristic, "expansion": expansion}


def metrics(pred: np.ndarray, y: np.ndarray, horizons: np.ndarray, prefix: str) -> dict[str, float]:
    """Relative TTC error (EvTTC metric) per horizon; invalid predictions count as 100 % error."""
    out = {}
    for j, h in enumerate(horizons):
        mask = label_mask(y[:, j])
        p, t = pred[mask, j], y[mask, j]
        rel = np.where(np.isfinite(p) & (p > 0), np.abs(p - t) / t, 1.0)
        tag = f"{prefix}/h{int(round(h * 1000))}ms"
        out[f"{tag}/median_rel_err"] = float(np.median(rel)) if rel.size else float("nan")
        out[f"{tag}/mean_rel_err"] = float(np.mean(rel)) if rel.size else float("nan")
        out[f"{tag}/within_20pct"] = float(np.mean(rel < 0.2)) if rel.size else float("nan")
        out[f"{tag}/n"] = int(mask.sum())
    return out


# ----------------------------------------------------------------------------- experiment
def cross_validate(seqs, names, cfg, use_reservoir, args, run) -> tuple[dict, list[int]]:
    horizons = np.array(next(iter(seqs.values()))["horizons_s"], dtype=np.float64)
    pooled = {"y": [], "learned": [], "heuristic": [], "expansion": []}
    best_epochs, fold_rows = [], []
    for k, test_name in enumerate(names):
        train_names = [n for n in names if n != test_name]
        val_name = train_names[k % len(train_names)]
        fit_names = [n for n in train_names if n != val_name]
        model = EsnTtcModel(len(FEATURE_NAMES), tuple(horizons), cfg, use_reservoir)
        model.fit_scaler(np.concatenate([seqs[n]["features"] for n in fit_names]))
        train, val, test = (rows(model, seqs, ns, args.washout) for ns in (fit_names, [val_name], [test_name]))
        history = fit_readout(model, train, val, args.epochs, args.lr, args.weight_decay, args.patience)
        best_epochs.append(history["best_epoch"] + 1)
        preds = predictions(model, test, horizons)
        for key in pooled:
            pooled[key].append(test["y"] if key == "y" else preds[key])
        fold = metrics(preds["learned"], test["y"], horizons, "fold")
        fold_rows.append([k, test_name, val_name, history["best_epoch"] + 1, len(test["y"]),
                          fold["fold/h0ms/median_rel_err"], fold["fold/h300ms/median_rel_err"]])
        print(f"fold {k:2d} test={test_name:20s} val={val_name:20s} best_epoch={history['best_epoch'] + 1:4d} "
              f"median_rel_err h0={fold['fold/h0ms/median_rel_err']:.3f} h300={fold['fold/h300ms/median_rel_err']:.3f}")
        if run is not None:
            import wandb
            epochs = list(range(1, len(history["train"]) + 1))
            run.log({f"curves/fold{k}": wandb.plot.line_series(xs=epochs, ys=[history["train"], history["val"]],
                                                                keys=["train", "val"], title=f"fold {k} ({test_name}) log-TTC MSE",
                                                                xname="epoch")})
    y = np.concatenate(pooled["y"])
    summary = {}
    for method in ("learned", "heuristic", "expansion"):
        summary.update(metrics(np.concatenate(pooled[method]), y, horizons, f"cv/{method}"))
    if run is not None:
        import wandb
        run.log({"folds": wandb.Table(columns=["fold", "test", "val", "best_epoch", "n_test", "median_rel_err_h0", "median_rel_err_h300"],
                                      data=fold_rows)})
    return summary, best_epochs


def data_fingerprint(features_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(features_dir.glob("*.npz")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("outputs/features"))
    p.add_argument("--out", type=Path, default=Path("outputs/models"))
    p.add_argument("--variant", choices=["esn", "mlp"], default="esn", help="mlp = no reservoir (ablation)")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--washout", type=int, default=5, help="first frames of each track excluded from loss/metrics")
    p.add_argument("--reservoir-size", type=int, default=EsnConfig.reservoir_size)
    p.add_argument("--leak-rate", type=float, default=EsnConfig.leak_rate)
    p.add_argument("--spectral-radius", type=float, default=EsnConfig.spectral_radius)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    seqs = load_sequences(args.features)
    names = list(seqs)
    use_reservoir = args.variant == "esn"
    cfg = EsnConfig(reservoir_size=args.reservoir_size, leak_rate=args.leak_rate,
                    spectral_radius=args.spectral_radius, seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {**vars(args), **asdict(cfg), "features": FEATURE_NAMES, "sequences": names,
              "data_fingerprint": data_fingerprint(args.features)}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}

    cv_run = start_run("cross-validation", config, tags=["esn", args.variant, "cv"], name=f"cv-{args.variant}")
    summary, best_epochs = cross_validate(seqs, names, cfg, use_reservoir, args, cv_run)
    for key in sorted(k for k in summary if "median_rel_err" in k):
        print(f"{key:48s} {summary[key]:.3f}")
    if cv_run is not None:
        cv_run.summary.update({**summary, "best_epochs": best_epochs})
        cv_run.finish()

    # Final fit on every sequence, for the typical CV stopping point.
    final_epochs = int(np.median(best_epochs))
    final_run = start_run("train", {**config, "final_epochs": final_epochs}, tags=["esn", args.variant, "final"],
                          name=f"final-{args.variant}")
    horizons = next(iter(seqs.values()))["horizons_s"]
    model = EsnTtcModel(len(FEATURE_NAMES), tuple(float(h) for h in horizons), cfg, use_reservoir)
    model.fit_scaler(np.concatenate([seqs[n]["features"] for n in names]))
    history = fit_readout(model, rows(model, seqs, names, args.washout), None, final_epochs, args.lr, args.weight_decay, args.patience)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.variant}_ttc.pt"
    model.save(path)
    print(f"final model ({final_epochs} epochs, train loss {history['train'][-1]:.4f}) -> {path}")
    if final_run is not None:
        for epoch, loss in enumerate(history["train"], 1):
            final_run.log({"train/log_ttc_mse": loss, "epoch": epoch})
        final_run.summary.update({"model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12]})
        final_run.finish()


if __name__ == "__main__":
    main()
