# src/train/sft.py
from typing import Dict, Any, Tuple
from transformers import (T5ForConditionalGeneration, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
import torch, os

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
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, {"train_loss": trainer.state.log_history}