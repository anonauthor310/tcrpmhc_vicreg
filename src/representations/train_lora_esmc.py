#!/usr/bin/env python3
"""Train the three LoRA ESMC-300M adapters used by the LoRA input family.

This is the masked-language-model procedure that writes

    tcr_encoder_checkpoint.pth
    peptide_encoder_checkpoint.pth
    hla_encoder_checkpoint.pth

which ``export_lora_esmc`` then loads. Adapters are not shipped.

Each modality (TCR, peptide, MHC) is a separate LoRA-adapted ``esmc_300m``
encoder. Base weights stay frozen; only LoRA A/B parameters are trained.
Masked LM uses 15% valid amino-acid positions (at least 2, at most 45), of
which 20% are random amino-acid replacements and the rest are ``<mask>``.

Defaults match the paper adapters: seed 31, 3 epochs, batch size 8,
AdamW lr 1e-4, weight decay 0.01, LoRA r=8 / alpha=32 / dropout=0.05 on
``out_proj`` and ``layernorm_qkv.1``. Training sequences come from the frozen
paper train split.

    PYTHONPATH=. python -m src.representations.train_lora_esmc
"""

from __future__ import annotations

import argparse
import gc
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from src.representations.lora import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    LORA_TARGET_MODULES,
)

REPO = Path(__file__).resolve().parents[2]
AA_IDS = [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_train_sequences(csv_path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)
    required = ("TCR_full", "Peptide", "HLA_sequence")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return {
        "tcrs_data": df["TCR_full"].astype(str).tolist(),
        "peptides_data": df["Peptide"].astype(str).tolist(),
        "hlas_data": df["HLA_sequence"].astype(str).tolist(),
        "n_rows": int(len(df)),
    }


class EncodedSeqDataset(Dataset):
    def __init__(self, sequences: Sequence[str], enc: Dict[str, Any]):
        self.sequences = list(sequences)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "sequence": self.sequences[idx],
            "input_ids": torch.as_tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.as_tensor(self.attention_mask[idx], dtype=torch.long),
        }


class MLMProteinCollator:
    def __init__(
        self,
        *,
        cls_id: int,
        eos_id: int,
        pad_id: int,
        mask_id: int,
        amino_acids: Sequence[int],
        tokenizer,
        p: float = 0.15,
        min_per_seq: int = 2,
        max_per_seq: int = 45,
        aa_frac: float = 0.20,
    ):
        self.CLS = cls_id
        self.EOS = eos_id
        self.PAD = pad_id
        self.MASK = mask_id
        self.aa = torch.as_tensor(list(amino_acids), dtype=torch.long)
        self.tokenizer = tokenizer
        self.p = p
        self.min_per_seq = min_per_seq
        self.max_per_seq = max_per_seq
        self.aa_frac = aa_frac

    @torch.no_grad()
    def mask_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        device = input_ids.device
        aa = self.aa.to(device)
        valid_mask = (
            attention_mask.bool()
            & (input_ids != self.PAD)
            & (input_ids != self.CLS)
            & (input_ids != self.EOS)
        )
        masked_input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        B = input_ids.shape[0]
        for i in range(B):
            vmask = valid_mask[i]
            if not vmask.any():
                continue
            valid_idx = vmask.nonzero(as_tuple=False).squeeze(1)
            L_valid = valid_idx.numel()
            n = torch.floor(self.p * torch.tensor(L_valid, device=device, dtype=torch.float32)).to(
                torch.int64
            )
            n = torch.clamp(n, min=self.min_per_seq, max=min(self.max_per_seq, L_valid))
            if n.item() == 0:
                continue
            chosen = valid_idx[torch.randperm(L_valid, device=device)[:n]]
            n_amino = torch.floor(self.aa_frac * n).to(torch.int64)
            if n.item() >= 2:
                n_amino = torch.clamp(n_amino, min=1)
            n_mask = n - n_amino
            order = torch.randperm(n.item(), device=device)
            mask_pos = chosen[order[:n_mask]]
            amino_pos = chosen[order[n_mask:]]
            labels[i, chosen] = input_ids[i, chosen]
            if n_mask.item() > 0:
                masked_input_ids[i, mask_pos] = self.MASK
            if n_amino.item() > 0:
                r_idx = torch.randint(high=aa.numel(), size=(n_amino.item(),), device=device)
                masked_input_ids[i, amino_pos] = aa[r_idx]
        return masked_input_ids, labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
        attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
        masked_input_ids, labels = self.mask_batch(input_ids, attention_mask)
        return {
            "masked_input_ids": masked_input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def build_modality_loaders(
    tokenizer,
    sequences: Dict[str, List[str]],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> Dict[str, DataLoader]:
    tok = tokenizer
    collator = MLMProteinCollator(
        cls_id=tok.cls_token_id,
        eos_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        mask_id=tok.mask_token_id,
        amino_acids=AA_IDS,
        tokenizer=tok,
    )
    loaders = {}
    for key, col in (("tcr", "tcrs_data"), ("pep", "peptides_data"), ("hla", "hlas_data")):
        enc = tokenizer(sequences[col], return_tensors="pt", padding=True)
        ds = EncodedSeqDataset(sequences[col], enc)
        loaders[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=collator,
        )
    return loaders


def optimizer_to_cpu(optim: torch.optim.Optimizer) -> None:
    for st in optim.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.detach().to("cpu")


def unfreeze_lora(model) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True


def train_encoder(
    *,
    modality: str,
    adapter_name: str,
    state_key: str,
    loader: DataLoader,
    checkpoint_path: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    from esm.models.esmc import ESMC
    from peft import LoraConfig
    from peft.tuners.lora import LoraModel

    print(f"\n=== {modality} LoRA MLM ({epochs} epochs) ===", flush=True)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
    )
    model = LoraModel(ESMC.from_pretrained("esmc_300m"), lora_cfg, adapter_name=adapter_name)
    unfreeze_lora(model)
    model.to(device)
    model.train()
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    use_amp = device.type == "cuda"
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["masked_input_ids"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device)
            optim.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out = model(input_ids)
                logits = out.sequence_logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            loss.backward()
            optim.step()
            running_loss += float(loss.item())
            n_batches += 1
            del out, logits, loss, input_ids, labels, batch
            if device.type == "cuda":
                torch.cuda.synchronize()
        avg = running_loss / max(1, n_batches)
        epoch_losses.append(avg)
        print(f"Epoch {epoch + 1}/{epochs} - {modality} MLM loss: {avg:.4f}", flush=True)

    optimizer_to_cpu(optim)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            state_key: model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "epoch_losses": epoch_losses,
        },
        checkpoint_path,
    )
    print(f"Wrote {checkpoint_path}", flush=True)
    model.to("cpu")
    del optim, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", default=str(REPO / "data/processed/train.csv.gz"))
    p.add_argument("--checkpoint-dir", default=str(REPO / "models/checkpoints"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("CUDA not available; falling back to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    sequences = load_train_sequences(args.train_csv)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("LoRA ESMC adapter MLM training", flush=True)
    print(f"Train CSV: {args.train_csv} ({sequences['n_rows']} rows)", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    from esm.models.esmc import ESMC

    tokenizer = ESMC.from_pretrained("esmc_300m").tokenizer
    loaders = build_modality_loaders(
        tokenizer,
        sequences,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    del tokenizer
    gc.collect()

    jobs = (
        ("TCR", "tcr", "tcr_model_state_dict", "tcr", "tcr_encoder_checkpoint.pth"),
        ("peptide", "pep", "peptide_model_state_dict", "pep", "peptide_encoder_checkpoint.pth"),
        ("MHC", "hla", "hla_model_state_dict", "hla", "hla_encoder_checkpoint.pth"),
    )
    for modality, adapter, state_key, loader_key, filename in jobs:
        train_encoder(
            modality=modality,
            adapter_name=adapter,
            state_key=state_key,
            loader=loaders[loader_key],
            checkpoint_path=checkpoint_dir / filename,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    print("\nAdapter training complete.", flush=True)
    print("Next: PYTHONPATH=. python -m src.representations.export_lora_esmc", flush=True)


if __name__ == "__main__":
    main()
