import argparse
import inspect
import json
import logging
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    logging as hf_transformers_logging,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    image_path: str
    label_name: str


def derive_label_name(label_json: Dict) -> str:
    cases = label_json.get("meta", {}).get("annotation_case", []) or []
    cleaned = sorted({str(c).strip().lower() for c in cases if str(c).strip()})
    if not cleaned:
        info_text = str(label_json.get("image_data", {}).get("information", "")).strip().lower()
        return info_text if info_text else "unknown"
    if len(cleaned) == 1:
        return cleaned[0]
    return " + ".join(cleaned)


def build_samples(source_root: Path, label_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    missing_images = 0

    for json_path in label_root.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        image_data = obj.get("image_data", {})
        file_name = image_data.get("file_name")
        image_format = image_data.get("format", "jpg")
        if not file_name:
            continue

        rel_parent = json_path.parent.relative_to(label_root)
        image_path = source_root / rel_parent / f"{file_name}.{image_format}"
        if not image_path.exists():
            missing_images += 1
            continue

        label_name = derive_label_name(obj)
        samples.append(Sample(str(image_path), label_name))

    if not samples:
        raise ValueError("No valid (image, label) pairs found. Check source/label folder paths.")

    if missing_images > 0:
        print(f"[WARN] Missing image files for {missing_images} labels. They were skipped.")

    return samples


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        rows: List[Tuple[str, int]],
        image_processor,
        image_size: int,
    ) -> None:
        self.rows = rows
        self.image_processor = image_processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, label_id = self.rows[idx]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        encoded = self.image_processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def freeze_backbone_params(model) -> None:
    # Keep only classification head trainable for faster CPU training.
    for _, param in model.named_parameters():
        param.requires_grad = False

    head_keywords = ("classifier", "score", "head")
    for name, param in model.named_parameters():
        if any(key in name for key in head_keywords):
            param.requires_grad = True


def limit_rows(rows: List[Tuple[str, int]], limit: int, seed: int) -> List[Tuple[str, int]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    copied = rows[:]
    rng.shuffle(copied)
    return copied[:limit]


def save_training_plots(log_history: List[Dict], output_dir: Path) -> Path:
    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_accs = [], [], []

    for row in log_history:
        if "loss" in row and "eval_loss" not in row:
            train_steps.append(row.get("step", 0))
            train_losses.append(row["loss"])
        if "eval_loss" in row:
            eval_steps.append(row.get("step", 0))
            eval_losses.append(row["eval_loss"])
            eval_accs.append(row.get("eval_accuracy"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_steps, train_losses, marker="o", label="train_loss")
    axes[0].plot(eval_steps, eval_losses, marker="s", label="eval_loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    valid_eval = [(s, a) for s, a in zip(eval_steps, eval_accs) if a is not None]
    if valid_eval:
        xs, ys = zip(*valid_eval)
        axes[1].plot(xs, ys, marker="^", color="#0E7490", label="eval_accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def save_result_tables(
    trainer: Trainer,
    eval_dataset: Dataset,
    id_to_label: Dict[int, str],
    output_dir: Path,
) -> Tuple[Path, Path]:
    pred_output = trainer.predict(eval_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=-1)
    true_ids = pred_output.label_ids
    label_ids = sorted(id_to_label)

    report = classification_report(
        true_ids,
        pred_ids,
        labels=label_ids,
        target_names=[id_to_label[i] for i in label_ids],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_csv = output_dir / "classification_report.csv"
    report_df.to_csv(report_csv, encoding="utf-8-sig")

    summary_rows = [
        {"metric": "eval_loss", "value": float(pred_output.metrics.get("test_loss", np.nan))},
        {"metric": "eval_accuracy", "value": float(pred_output.metrics.get("test_accuracy", np.nan))},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== Classification Report (head) ===")
    print(report_df.head(10).to_string())

    return report_csv, summary_csv


def verify_saved_model_artifacts(model_dir: Path) -> Dict[str, str]:
    artifact_candidates = {
        "config": [model_dir / "config.json"],
        "weights": [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"],
        "processor": [model_dir / "preprocessor_config.json", model_dir / "feature_extractor_config.json"],
    }

    resolved: Dict[str, str] = {}
    missing: list[str] = []

    for artifact_name, candidates in artifact_candidates.items():
        existing = next((path for path in candidates if path.exists()), None)
        if existing is None:
            missing.append(artifact_name)
        else:
            resolved[artifact_name] = str(existing)

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Saved model verification failed. Missing artifacts in {model_dir}: {missing_text}"
        )

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ViT classifier from source/label folders.")
    parser.add_argument("--source-dir", default="01.원천데이터", help="Root folder of source images")
    parser.add_argument("--label-dir", default="02.라벨링데이터", help="Root folder of label JSON files")
    parser.add_argument("--model-name", default="google/vit-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=0, help="Use 0 for all samples")
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true", help="Train classifier head only")
    parser.add_argument("--cpu-optimized", action="store_true", help="Apply safer defaults for CPU-only training")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on eval metric")
    parser.add_argument("--early-stopping-patience", type=int, default=2, help="Patience epochs for early stopping")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="Optional Hugging Face token")
    parser.add_argument("--show-warnings", action="store_true", help="Show library warnings/logs")
    parser.add_argument("--train-samples", type=int, default=0, help="Limit number of training samples after split")
    parser.add_argument("--val-samples", type=int, default=0, help="Limit number of validation samples after split")
    parser.add_argument(
        "--tiny-test",
        action="store_true",
        help="Fast smoke test mode: use 10 train + 10 validation samples and 1 epoch",
    )
    parser.add_argument("--output-dir", default="outputs_vit")
    args = parser.parse_args()

    if not args.show_warnings:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        warnings.filterwarnings("ignore", message=".*pin_memory.*")
        warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        hf_transformers_logging.set_verbosity_error()

    if args.cpu_optimized:
        if args.model_name == "google/vit-base-patch16-224":
            args.model_name = "facebook/deit-tiny-patch16-224"
        args.batch_size = min(args.batch_size, 4)
        args.image_size = min(args.image_size, 192)
        args.freeze_backbone = True
        print("[INFO] CPU optimized mode enabled")

    if args.tiny_test:
        args.train_samples = 10
        args.val_samples = 10
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        print("[INFO] Tiny test mode enabled: train=10, val=10, epochs=1")

    set_seed(args.seed)

    source_root = Path(args.source_dir)
    label_root = Path(args.label_dir)
    if not source_root.exists() or not label_root.exists():
        raise FileNotFoundError("source-dir or label-dir does not exist.")

    samples = build_samples(source_root=source_root, label_root=label_root)

    if args.max_samples > 0:
        random.shuffle(samples)
        samples = samples[: args.max_samples]

    labels = [s.label_name for s in samples]
    label_names = sorted(set(labels))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    id_to_label = {i: name for name, i in label_to_id.items()}

    print(f"[INFO] total samples: {len(samples)}")
    print(f"[INFO] label count: {len(label_names)}")
    print(f"[INFO] labels: {label_names}")

    rows = [(s.image_path, label_to_id[s.label_name]) for s in samples]
    y = [row[1] for row in rows]

    can_stratify = len(set(y)) > 1 and min(np.bincount(np.array(y))) >= 2
    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y if can_stratify else None,
    )

    train_rows = limit_rows(train_rows, args.train_samples, args.seed)
    val_rows = limit_rows(val_rows, args.val_samples, args.seed + 1)

    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError("train/validation rows became empty. Increase sample limits.")

    print(f"[INFO] split rows -> train: {len(train_rows)}, val: {len(val_rows)}")

    token = args.hf_token if args.hf_token else None

    image_processor = AutoImageProcessor.from_pretrained(args.model_name, token=token)
    train_dataset = ImageClassificationDataset(train_rows, image_processor, args.image_size)
    val_dataset = ImageClassificationDataset(val_rows, image_processor, args.image_size)

    config = AutoConfig.from_pretrained(args.model_name, token=token)
    config.num_labels = len(label_names)
    config.id2label = id_to_label
    config.label2id = label_to_id

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        config=config,
        token=token,
        ignore_mismatched_sizes=True,
    )

    if args.freeze_backbone:
        freeze_backbone_params(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Freeze backbone enabled: trainable params={trainable:,} / total={total:,}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        if args.early_stopping
        else None,
    }

    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = image_processor
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = image_processor

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("\n=== Final Eval Metrics ===")
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")

    model_dir = output_dir / "best_model"
    trainer.save_model(str(model_dir))
    image_processor.save_pretrained(str(model_dir))
    saved_artifacts = verify_saved_model_artifacts(model_dir)

    history_path = output_dir / "training_log.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(history_path, index=False, encoding="utf-8-sig")

    plot_path = save_training_plots(trainer.state.log_history, output_dir)
    report_csv, summary_csv = save_result_tables(trainer, val_dataset, id_to_label, output_dir)

    print("\n=== Artifacts ===")
    print(f"- Model: {model_dir}")
    print(f"- Model config: {saved_artifacts['config']}")
    print(f"- Model weights: {saved_artifacts['weights']}")
    print(f"- Model processor: {saved_artifacts['processor']}")
    print(f"- Training log: {history_path}")
    print(f"- Curves image: {plot_path}")
    print(f"- Classification report table: {report_csv}")
    print(f"- Evaluation summary table: {summary_csv}")


if __name__ == "__main__":
    main()
