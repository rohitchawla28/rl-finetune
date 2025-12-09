# train/sft.py
from typing import Dict, Any, Tuple
import math, os
from dataclasses import fields, is_dataclass

import torch
from transformers import (
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

def _filter_kwargs_for_dataclass(cls, **kwargs):
    """Keep only kwargs that are valid fields for the given dataclass (version-safe)."""
    if not is_dataclass(cls):
        return kwargs
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in kwargs.items() if k in allowed}

def run_sft(
    train_dataset,
    val_dataset,
    tokenizer,
    model_name: str = "google/flan-t5-small",
    output_dir: str = "./checkpoints/flan_t5_sft_cnn_dm",
    batch_size: int = 4,
    num_epochs: int = 1,
    lr: float = 5e-5,
    warmup_ratio: float = 0.1,
) -> Tuple[T5ForConditionalGeneration, Dict[str, Any]]:
    # Model + collator
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Compute warmup steps in case this version lacks warmup_ratio
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    warmup_steps = max(1, int(warmup_ratio * steps_per_epoch * num_epochs))

    # Build a superset of arguments, then filter for this transformers version
    args_superset = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        # new-ish APIs (kept if supported)
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=warmup_ratio,
        report_to="none",
        predict_with_generate=True,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        # old-compatible fallback fields
        warmup_steps=warmup_steps,
    )
    args_clean = _filter_kwargs_for_dataclass(Seq2SeqTrainingArguments, **args_superset)
    training_args = Seq2SeqTrainingArguments(**args_clean)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Always try an eval once (covers very old versions lacking eval scheduling)
    try:
        eval_metrics = trainer.evaluate()
    except Exception:
        eval_metrics = {}

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    history = {"eval_metrics": eval_metrics}
    if hasattr(trainer.state, "log_history"):
        history["train_log_history"] = trainer.state.log_history
    return model, history