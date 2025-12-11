"""
data.py

Helpers for loading and preprocessing summarization data
(CNN/DailyMail and XSum) for SFT / RL experiments.
"""

from typing import Tuple, Literal
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, PreTrainedTokenizerBase


def load_cnn_dm_raw(
    n_train: int = 500,
    n_val: int = 100,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train = dataset["train"].shuffle(seed=seed).select(range(n_train))
    val = dataset["validation"].shuffle(seed=seed).select(range(n_val))
    return train, val


def get_tokenizer(tokenizer_name: str = "google/flan-t5-small") -> PreTrainedTokenizerBase:
    return T5Tokenizer.from_pretrained(tokenizer_name)


def preprocess_cnn_dm(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_len: int = 512,
    max_target_len: int = 128,
) -> Tuple[Dataset, Dataset]:
    def build_prompt(article: str) -> str:
        # Optionally: return "Summarize the following article:\n\n" + article
        return article

    def preprocess_batch(batch):
        prompts = [build_prompt(a) for a in batch["article"]]
        model_inputs = tokenizer(
            prompts,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["highlights"],
                max_length=max_target_len,
                truncation=True,
                padding="max_length",
            )

        pad_id = tokenizer.pad_token_id
        processed_labels = []
        for seq in labels["input_ids"]:
            processed_seq = [(tid if tid != pad_id else -100) for tid in seq]
            processed_labels.append(processed_seq)

        model_inputs["labels"] = processed_labels
        return model_inputs

    train_tok = train_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tok = val_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=val_ds.column_names,
    )

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
) -> Tuple[PreTrainedTokenizerBase, Dataset, Dataset]:
    tokenizer = get_tokenizer(tokenizer_name)
    train_raw, val_raw = load_cnn_dm_raw(
        n_train=n_train,
        n_val=n_val,
        seed=seed,
    )
    train_tok, val_tok = preprocess_cnn_dm(
        train_raw,
        val_raw,
        tokenizer=tokenizer,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )
    return tokenizer, train_tok, val_tok


# XSum dataset support
def load_xsum_raw(
    n_train: int = 500,
    n_val: int = 100,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load XSum dataset (document/summary format)."""
    dataset = load_dataset("xsum")
    train = dataset["train"].shuffle(seed=seed).select(range(n_train))
    val = dataset["validation"].shuffle(seed=seed).select(range(n_val))
    return train, val


def preprocess_xsum(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_len: int = 512,
    max_target_len: int = 128,
) -> Tuple[Dataset, Dataset]:
    """Preprocess XSum dataset (uses 'document' and 'summary' fields)."""
    def build_prompt(document: str) -> str:
        return document

    def preprocess_batch(batch):
        prompts = [build_prompt(d) for d in batch["document"]]
        model_inputs = tokenizer(
            prompts,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                max_length=max_target_len,
                truncation=True,
                padding="max_length",
            )

        pad_id = tokenizer.pad_token_id
        processed_labels = []
        for seq in labels["input_ids"]:
            processed_seq = [(tid if tid != pad_id else -100) for tid in seq]
            processed_labels.append(processed_seq)

        model_inputs["labels"] = processed_labels
        return model_inputs

    train_tok = train_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tok = val_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    cols = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format(type="torch", columns=cols)
    val_tok.set_format(type="torch", columns=cols)

    return train_tok, val_tok


def load_xsum_tokenized(
    n_train: int = 500,
    n_val: int = 100,
    seed: int = 42,
    tokenizer_name: str = "google/flan-t5-small",
    max_input_len: int = 512,
    max_target_len: int = 128,
) -> Tuple[PreTrainedTokenizerBase, Dataset, Dataset]:
    """Load and tokenize XSum dataset."""
    tokenizer = get_tokenizer(tokenizer_name)
    train_raw, val_raw = load_xsum_raw(
        n_train=n_train,
        n_val=n_val,
        seed=seed,
    )
    train_tok, val_tok = preprocess_xsum(
        train_raw,
        val_raw,
        tokenizer=tokenizer,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )
    return tokenizer, train_tok, val_tok


def load_dataset_tokenized(
    dataset_name: Literal["cnn_dailymail", "xsum"],
    n_train: int = 500,
    n_val: int = 100,
    seed: int = 42,
    tokenizer_name: str = "google/flan-t5-small",
    max_input_len: int = 512,
    max_target_len: int = 128,
) -> Tuple[PreTrainedTokenizerBase, Dataset, Dataset]:
    """Unified interface to load either CNN/DM or XSum."""
    if dataset_name == "cnn_dailymail":
        return load_cnn_dm_tokenized(
            n_train=n_train,
            n_val=n_val,
            seed=seed,
            tokenizer_name=tokenizer_name,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
        )
    elif dataset_name == "xsum":
        return load_xsum_tokenized(
            n_train=n_train,
            n_val=n_val,
            seed=seed,
            tokenizer_name=tokenizer_name,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'cnn_dailymail' or 'xsum'")