"""Make YOLOv9t DPU-ready: SiLU -> Hardswish, prune channels, fine-tune (designed for a Colab GPU).

    python scripts/prune_yolo.py --data coco.yaml --fraction 0.25 --epochs 20 --ratio 0.3
    python scripts/prune_yolo.py --data coco8.yaml --epochs 1 --device cpu    # smoke test

Steps, each logged to W&B: evaluate the original model, swap SiLU for
Hardswish (the DPU cannot run SiLU), prune channels
(collision_avoidance.yolo_pruning), evaluate without fine-tuning, fine-tune
with Ultralytics' trainer (kept on the modified architecture), evaluate again.
With both changes the Vitis AI inspector places every operator on the DPU.
The fine-tuned checkpoint is written to --out.
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO, settings  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

from collision_avoidance import yolo_pruning  # noqa: E402
from collision_avoidance.telemetry import start_run  # noqa: E402


class PrunedModelTrainer(DetectionTrainer):
    """Ultralytics rebuilds models from YAML; hand it the pruned network instead."""

    pruned_model = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        return self.pruned_model


def evaluate(model_or_path, data: str, imgsz: int, device, batch: int) -> dict[str, float]:
    # val() fuses layers and leaves inference tensors behind, so always evaluate a copy.
    yolo = YOLO(model_or_path) if isinstance(model_or_path, (str, Path)) else copy.deepcopy(model_or_path)
    m = yolo.val(data=data, imgsz=imgsz, device=device, batch=batch, plots=False, verbose=False).box
    return {"map50_95": float(m.map), "map50": float(m.map50)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(ROOT / "yolov9t.pt"))
    p.add_argument("--data", default="coco.yaml")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of the training set used for fine-tuning")
    p.add_argument("--ratio", type=float, default=0.3, help="channel pruning ratio (0 = activation swap only)")
    p.add_argument("--act", choices=["hardswish", "silu"], default="hardswish", help="activation for the DPU model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--optimizer", default="SGD", help="explicit: Ultralytics' 'auto' ignores lr0 on short runs")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--nbs", type=int, default=64, help="nominal batch for gradient accumulation (lower it for tiny smoke-test sets)")
    p.add_argument("--device", default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default=str(ROOT / "outputs" / "prune"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "models" / "yolov9t_pruned.pt"))
    args = p.parse_args(argv)
    settings.update({"wandb": False})  # we log to our own run below

    run = start_run("prune", vars(args), tags=["yolov9t", "pruning", f"ratio{args.ratio}"])
    log = (lambda d: run.log(d)) if run is not None else (lambda d: None)

    original = YOLO(args.weights)
    base_cost = yolo_pruning.complexity(original.model.float(), args.imgsz)
    base_metrics = evaluate(original, args.data, args.imgsz, args.device, args.batch)
    print("original:", base_metrics, base_cost)

    model = YOLO(args.weights).model.float()
    if args.act == "hardswish":
        model = yolo_pruning.replace_silu(model)
    pruned = yolo_pruning.prune(model, args.ratio, args.imgsz) if args.ratio > 0 else model
    pruned_cost = yolo_pruning.complexity(pruned, args.imgsz)
    pruned_yolo = YOLO(args.weights)
    pruned_yolo.model = pruned
    pruned_metrics = evaluate(pruned_yolo, args.data, args.imgsz, args.device, args.batch)
    print(f"modified ({args.act}, ratio {args.ratio}), before fine-tuning:", pruned_metrics, pruned_cost)

    PrunedModelTrainer.pruned_model = pruned.train()
    trainer = PrunedModelTrainer(overrides=dict(
        model=args.weights, data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        optimizer=args.optimizer, lr0=args.lr0, warmup_epochs=args.warmup_epochs, nbs=args.nbs, fraction=args.fraction, device=args.device, workers=args.workers,
        project=args.project, name="finetune", exist_ok=True, plots=False,
    ))
    trainer.add_callback("on_fit_epoch_end", lambda t: log({
        "epoch": t.epoch + 1,
        **{f"train/{k}": float(v) for k, v in zip(t.loss_names, t.tloss)},
        **{f"val/{k.split('/')[-1].replace('(B)', '')}": float(v) for k, v in t.metrics.items()},
    }))
    trainer.train()

    best = Path(trainer.best)
    final_metrics = evaluate(str(best), args.data, args.imgsz, args.device, args.batch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    summary = {
        **{f"original/{k}": v for k, v in {**base_metrics, **base_cost}.items()},
        **{f"pruned_no_ft/{k}": v for k, v in {**pruned_metrics, **pruned_cost}.items()},
        **{f"pruned_ft/{k}": v for k, v in final_metrics.items()},
        "gflops_reduction": 1 - pruned_cost["gflops"] / base_cost["gflops"],
        "output": str(out),
    }
    print(summary)
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
