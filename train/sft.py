# train/sft.py
from typing import Dict, Any, Tuple
import math, os
from transformers import (
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch

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
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    steps_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    warmup_steps = max(1, int(warmup_ratio * steps_per_epoch * num_epochs))

    # OLD-CODE-COMPATIBLE: no evaluation_strategy/save_strategy/warmup_ratio
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=25,
        warmup_steps=warmup_steps,
        predict_with_generate=False,   # old eval style (you eval separately)
        fp16=False,                    # keep stable; flip to True later if you want
        report_to="none",
        remove_unused_columns=False,   # preserve fields exactly like old code paths
        save_steps=10_000_000,         # effectively disable mid-run saves
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # one explicit eval pass (loss-only, matches “old code” behavior)
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