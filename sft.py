# sft.py
from typing import Dict, Any, Optional, Tuple
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_loss(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_sft(
    train_dataset,
    val_dataset,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str = "google/flan-t5-small",
    batch_size: int = 4,
    num_epochs: int = 1,
    lr: float = 5e-5,
    warmup_ratio: float = 0.1,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[T5ForConditionalGeneration, Dict[str, Any]]:
    device = _get_device(device)
    print(f"[SFT] Using device: {device}")

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    train_loader, val_loader = _make_dataloaders(train_dataset, val_dataset, batch_size)

    num_training_steps = num_epochs * len(train_loader)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    history: Dict[str, Any] = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                avg_step_loss = running_loss / (step + 1)
                print(
                    f"[SFT] Epoch {epoch+1}/{num_epochs} "
                    f"Step {step+1}/{len(train_loader)} "
                    f"loss={avg_step_loss:.4f}"
                )

        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate_loss(model, val_loader, device)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"[SFT] Epoch {epoch+1} done. "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}"
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[SFT] Saving model + tokenizer to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return model, history