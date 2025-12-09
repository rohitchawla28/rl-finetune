# train/sft.py
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

    # ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # compute warmup_steps (works even if warmup_ratio isn't supported)
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    warmup_steps = max(1, int(warmup_ratio * steps_per_epoch * num_epochs))

    # MINIMAL ARGS: no evaluation_strategy, no save_strategy, no warmup_ratio
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=25,
        warmup_steps=warmup_steps,
        predict_with_generate=False,   # stays compatible across versions
        fp16=torch.cuda.is_available(),
        report_to="none",              # harmless if ignored
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # run one eval pass explicitly (covers very old Trainer APIs)
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