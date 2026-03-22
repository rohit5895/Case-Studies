"""
TrOCR Fine-tuning Script for SageMaker
=======================================
Supports both single-GPU and multi-GPU (DDP via torchrun).

Data format expected in S3:
  s3://<your-bucket>/data/
    train/
      images/          <- all training .png/.jpg files
      labels.csv       <- columns: file_name, text
    val/
      images/
      labels.csv

SageMaker will mount these to:
  /opt/ml/input/data/training/
  /opt/ml/input/data/validation/

Usage (SDK) — multi-GPU with ml.g5.12xlarge (4x A10G):
  from sagemaker.huggingface import HuggingFace
  estimator = HuggingFace(
      entry_point="train.py",
      instance_type="ml.g5.12xlarge",
      instance_count=1,
      role=role,
      transformers_version="4.36",
      pytorch_version="2.1",
      py_version="py310",
      distribution={"torch_distributed": {"enabled": True}},
      hyperparameters={
          "epochs": 10,
          "batch_size": 32,        # per-GPU batch; effective = 32 * 4 GPUs = 128
          "learning_rate": 5e-5,
          "warmup_steps": 200,
          "eval_every_n_epochs": 2,
          "model_name": "microsoft/trocr-large-stage1",
          "gradient_accumulation_steps": 1,
      },
      use_spot_instances=True,
      max_wait=86400,
      max_run=86400,
      checkpoint_s3_uri="s3://<your-bucket>/checkpoints",
  )
  estimator.fit({
      "training":   "s3://<your-bucket>/Train",
      "validation": "s3://<your-bucket>/Val",
  })
"""

import argparse
import os
import json
import logging

import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)
from evaluate import load as load_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────

def setup_distributed():
    """Initialize process group if running under torchrun/torch.distributed.launch."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return local_rank, world_size


def is_main_process(world_size):
    if world_size == 1:
        return True
    return dist.get_rank() == 0


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class CellOCRDataset(Dataset):
    """Dataset for cropped table-cell images with OCR text labels.

    Args:
        root_dir: Path to the directory containing image files.
        df: DataFrame with columns 'file_name' and 'text'.
        processor: TrOCRProcessor for image preprocessing and tokenization.
        max_target_length: Maximum token length for label sequences.
    """

    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length

        # Pre-tokenize all labels once at init instead of per __getitem__ call
        pad_id = processor.tokenizer.pad_token_id
        encodings = processor.tokenizer(
            self.df["text"].tolist(),
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
        ).input_ids
        self.labels = torch.stack([
            torch.tensor([t if t != pad_id else -100 for t in ids])
            for ids in encodings
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["file_name"][idx]
        image_path = os.path.join(self.root_dir, file_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}. Returning blank image.")
            image = Image.new("RGB", (384, 384), color=255)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": self.labels[idx],
        }


# ─────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────

def load_model(model_name, processor, device):
    """Load TrOCR and apply the sinusoidal embedding device fix.

    TrOCRSinusoidalPositionalEmbedding stores its weights as a plain
    Python attribute (not a registered buffer), so .to(device) silently
    skips them, causing a device mismatch at runtime. This function
    patches the weights onto the correct device after loading.
    """
    logger.info(f"Loading model: {model_name}")
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    # Fix: move sinusoidal positional embedding weights to the correct device
    embed_pos = model.decoder.model.decoder.embed_positions
    embed_pos.weights = embed_pos.get_embedding(
        embed_pos.weights.shape[0],
        embed_pos.embedding_dim,
        embed_pos.padding_idx,
    ).to(device)
    logger.info(f"embed_positions.weights device: {embed_pos.weights.device}")

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 64
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.num_beams = 1

    return model


def decode_batch(pred_ids, label_ids, processor):
    """Decode predicted and label token IDs to strings for CER computation."""
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = label_ids.clone()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return pred_str, label_str


def unwrap(model):
    """Return the underlying model, unwrapping DDP if needed."""
    return model.module if isinstance(model, DDP) else model


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    main = is_main_process(world_size)

    if main:
        logger.info(f"World size: {world_size} | Device: {device}")

    # ── Data paths ───────────────────────────────────────────────────────
    train_image_dir = os.path.join(args.training_dir, "images")
    val_image_dir   = os.path.join(args.validation_dir, "images")
    train_csv = os.path.join(args.training_dir, "labels.csv")
    val_csv   = os.path.join(args.validation_dir, "labels.csv")

    for csv_path in (train_csv, val_csv):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    if main:
        logger.info(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # ── Processor & datasets ─────────────────────────────────────────────
    processor = TrOCRProcessor.from_pretrained(args.model_name, use_fast=True)

    train_dataset = CellOCRDataset(train_image_dir, train_df, processor, args.max_target_length)
    val_dataset   = CellOCRDataset(val_image_dir,   val_df,   processor, args.max_target_length)

    # DistributedSampler shards data across GPUs; each GPU sees a unique subset
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=dist.get_rank() if world_size > 1 else 0,
                                       shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size,
                                       rank=dist.get_rank() if world_size > 1 else 0,
                                       shuffle=False)

    num_workers = min(8, (os.cpu_count() or 4) // world_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = load_model(args.model_name, processor, device)
    # Freeze the ViT pooler — it's never used in seq2seq forward/backward,
    # which causes DDP to complain about unused parameters.
    for param in model.encoder.pooler.parameters():
        param.requires_grad = False

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # num_training_steps counts optimizer steps (after grad accumulation), per rank
    steps_per_epoch = len(train_dataloader)
    num_training_steps = (steps_per_epoch // args.gradient_accumulation_steps) * args.epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler("cuda")
    cer_metric = load_metric("cer")
    best_cer = float("inf")
    start_epoch = 0
    start_step = 0

    # ── Resume from checkpoint ───────────────────────────────────────────
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    if os.path.isfile(checkpoint_path):
        if main:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        unwrap(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        saved_step = ckpt.get("step", -1)
        best_cer = ckpt["best_cer"]
        # step=-1 means the checkpoint was saved at end-of-epoch; start the next epoch from step 0
        if saved_step == -1:
            start_epoch += 1
            start_step = 0
        else:
            start_step = saved_step + 1
        if main:
            logger.info(f"Resumed at epoch {start_epoch}, step {start_step}, best CER: {best_cer}")
    else:
        if main:
            logger.info("No checkpoint found, starting from scratch.")

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # Tell sampler which epoch we're on so shuffle is deterministic but different each epoch
        train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            # Fast-forward already-completed steps when resuming mid-epoch
            if epoch == start_epoch and step < start_step:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * args.gradient_accumulation_steps

            if main and step % 100 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(train_dataloader)} "
                    f"| Loss: {loss.item() * args.gradient_accumulation_steps:.4f}"
                )

            # ── Intra-epoch checkpoint (rank 0 only) ─────────────────────
            if main and args.save_every_n_steps > 0 and (step + 1) % args.save_every_n_steps == 0:
                _save_checkpoint(checkpoint_path, args, epoch, step,
                                 model, optimizer, lr_scheduler, scaler, best_cer)
                logger.info(f"Intra-epoch checkpoint saved at epoch {epoch} step {step}")

        if main:
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} complete | Avg train loss: {avg_train_loss:.4f}")

        # ── Validation (all ranks run inference; rank 0 aggregates) ──────
        if (epoch + 1) % args.eval_every_n_epochs == 0:
            model.eval()
            all_preds, all_refs = [], []
            with torch.no_grad():
                for batch in val_dataloader:
                    outputs = unwrap(model).generate(
                        batch["pixel_values"].to(device),
                        num_beams=1,
                        max_new_tokens=64,
                    )
                    pred_str, label_str = decode_batch(outputs, batch["labels"], processor)
                    all_preds.extend(pred_str)
                    all_refs.extend(label_str)

            # Gather predictions from all ranks onto rank 0
            if world_size > 1:
                gathered_preds = [None] * world_size
                gathered_refs  = [None] * world_size
                dist.all_gather_object(gathered_preds, all_preds)
                dist.all_gather_object(gathered_refs,  all_refs)
                all_preds = [p for rank_preds in gathered_preds for p in rank_preds]
                all_refs  = [r for rank_refs  in gathered_refs  for r in rank_refs]

            if main:
                avg_cer = cer_metric.compute(predictions=all_preds, references=all_refs)
                logger.info(f"Epoch {epoch} | Validation CER: {avg_cer:.4f}")

                if avg_cer < best_cer:
                    best_cer = avg_cer
                    best_model_path = os.path.join(args.model_dir, "best")
                    unwrap(model).save_pretrained(best_model_path)
                    processor.save_pretrained(best_model_path)
                    logger.info(f"New best model saved (CER: {best_cer:.4f}) -> {best_model_path}")

        # ── End-of-epoch checkpoint (rank 0 only) ────────────────────────
        if main:
            _save_checkpoint(checkpoint_path, args, epoch, -1,
                             model, optimizer, lr_scheduler, scaler, best_cer)
            logger.info(f"Checkpoint saved at epoch {epoch} -> {checkpoint_path}")

        start_step = 0  # reset for subsequent epochs
        barrier(world_size)

    # ── Final model ──────────────────────────────────────────────────────
    if main:
        final_model_path = os.path.join(args.model_dir, "final")
        unwrap(model).save_pretrained(final_model_path)
        processor.save_pretrained(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        metadata = {
            "best_val_cer": best_cer,
            "epochs_trained": args.epochs,
            "model_name": args.model_name,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "world_size": world_size,
        }
        with open(os.path.join(args.model_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training complete. Best CER: {best_cer:.4f}")

    if world_size > 1:
        dist.destroy_process_group()


def _save_checkpoint(path, args, epoch, step, model, optimizer, lr_scheduler, scaler, best_cer):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_cer": best_cer,
        },
        path,
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs",                       type=int,   default=10)
    parser.add_argument("--batch_size",                   type=int,   default=32)
    parser.add_argument("--learning_rate",                type=float, default=5e-5)
    parser.add_argument("--max_target_length",            type=int,   default=128)
    parser.add_argument("--warmup_steps",                 type=int,   default=200)
    parser.add_argument("--eval_every_n_epochs",          type=int,   default=2)
    parser.add_argument("--gradient_accumulation_steps",  type=int,   default=1)
    parser.add_argument("--save_every_n_steps",           type=int,   default=200)
    parser.add_argument("--max_grad_norm",                type=float, default=1.0)
    parser.add_argument("--model_name",                   type=str,   default="microsoft/trocr-large-stage1")

    parser.add_argument("--model_dir",      type=str, default=os.environ.get("SM_MODEL_DIR",          "/opt/ml/model"))
    parser.add_argument("--training_dir",   type=str, default=os.environ.get("SM_CHANNEL_TRAINING",   "/opt/ml/input/data/training"))
    parser.add_argument("--validation_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--checkpoint_dir", type=str, default=os.environ.get("SM_CHECKPOINT_DIR",     "/opt/ml/checkpoints"))

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info(f"Training args: {vars(args)}")

    train(args)
