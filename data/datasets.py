# src/data/datasets.py
from typing import Tuple
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, PreTrainedTokenizerBase

def load_cnn_dm_raw(n_train: int = 500, n_val: int = 100, seed: int = 42) -> Tuple[Dataset, Dataset]:
    ds = load_dataset("cnn_dailymail", "3.0.0")
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    val = ds["validation"].shuffle(seed=seed).select(range(n_val))
    return train, val

def get_tokenizer(name: str = "google/flan-t5-small") -> PreTrainedTokenizerBase:
    tok = T5Tokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def preprocess_cnn_dm(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_len: int = 512,
    max_target_len: int = 128,
):
    prefix = "summarize: "

    def _prep(batch):
        inputs = [prefix + a for a in batch["article"]]
        enc = tokenizer(inputs, max_length=max_input_len, truncation=True, padding="max_length")
        # modern way to tokenize targets (no as_target_tokenizer)
        labels = tokenizer(text_target=batch["highlights"],
                           max_length=max_target_len, truncation=True, padding="max_length")
        pad_id = tokenizer.pad_token_id
        enc["labels"] = [[(tid if tid != pad_id else -100) for tid in seq] for seq in labels["input_ids"]]
        return enc

    train_tok = train_ds.map(_prep, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(_prep,   batched=True, remove_columns=val_ds.column_names)
    cols = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format(type="torch", columns=cols)
    val_tok.set_format(type="torch", columns=cols)
    return train_tok, val_tok

def load_cnn_dm_tokenized(
    n_train: int = 500,
    n_val: int = 100,
    seed: int = 42,
    tokenizer_name: str = "google/flan-t5-small",
    max_input_len: int = 512,
    max_target_len: int = 128,
):
    tok = get_tokenizer(tokenizer_name)
    train_raw, val_raw = load_cnn_dm_raw(n_train=n_train, n_val=n_val, seed=seed)
    train_tok, val_tok = preprocess_cnn_dm(train_raw, val_raw, tokenizer=tok,
                                           max_input_len=max_input_len, max_target_len=max_target_len)
    return tok, train_tok, val_tok