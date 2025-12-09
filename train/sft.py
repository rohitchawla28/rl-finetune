# src/train/sft.py
from typing import Dict, Any, Tuple
import math, os, torch
from transformers import (
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

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

    # Try modern API first; fall back if needed (keeps you compatible across versions)
    try:
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            predict_with_generate=True,
            logging_steps=25,
            warmup_ratio=warmup_ratio,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
    except TypeError:
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            logging_steps=25,
            warmup_steps=warmup_steps,
            fp16=torch.cuda.is_available(),
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
    # Always evaluate once (works on old/new transformers)
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