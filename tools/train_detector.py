"""Fine-tune a pretrained detector and evaluate it on held-out sessions."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catcare.config import LOCAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=LOCAL / "dataset" / "data.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=LOCAL / "training")
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit("Prepare the labeled dataset before training.")
    (LOCAL / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(LOCAL / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "weights_dir": str(LOCAL / "models"),
                     "datasets_dir": str(LOCAL / "dataset"), "runs_dir": str(LOCAL / "training")})
    source = LOCAL / "models" / args.model if args.model in ("yolo26n.pt", "yolo26s.pt") else Path(args.model)
    source.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(source))
    def progress(phase, **values):
        if args.progress_file is None:
            return
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.progress_file.with_suffix(".tmp")
        temporary.write_text(json.dumps({"phase": phase, "epochs": args.epochs, **values}), encoding="utf-8")
        temporary.replace(args.progress_file)

    model.add_callback("on_train_start", lambda trainer: progress("training", epoch=0))
    model.add_callback("on_train_epoch_end", lambda trainer: progress("training", epoch=trainer.epoch + 1))
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(args.project),
                name="cat-detector", patience=10, seed=42, workers=0)
    best = Path(model.trainer.best)
    progress("evaluating", epoch=model.trainer.epoch + 1)
    metrics = YOLO(str(best)).val(data=str(args.data.resolve()), split="test",
                                imgsz=args.imgsz, device=args.device, batch=args.batch, workers=0)
    report = {"weights": str(best), "evaluation_split": "test", "mAP50": float(metrics.box.map50),
              "mAP50_95": float(metrics.box.map), "precision": float(metrics.box.mp),
              "recall": float(metrics.box.mr)}
    (best.parent.parent / "held_out_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress("completed", epoch=model.trainer.epoch + 1, **report)
    print(json.dumps(report, indent=2))
    print("Choose these weights using Browse in the monitor. Compare real visits before replacing your baseline.")


if __name__ == "__main__":
    main()
