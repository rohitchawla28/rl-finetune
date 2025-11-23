"""
data.py

Helpers for loading and preprocessing summarization data
(CNN/DailyMail) for SFT / RL experiments.
"""

from typing import Tuple

from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, PreTrainedTokenizerBase


def load_cnn_dm_raw(
    n_train: int = 500,
    n_val: int = 100,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Load a small subset of the CNN/DailyMail dataset.
    Returns train/val splits with 'article' and 'highlights' fields.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train = dataset["train"].shuffle(seed=seed).select(range(n_train))
    val = dataset["validation"].shuffle(seed=seed).select(range(n_val))

    return train, val


def get_tokenizer(tokenizer_name: str = "google/flan-t5-small") -> PreTrainedTokenizerBase:
    """Load a T5/FLAN-T5 tokenizer by name."""
    return T5Tokenizer.from_pretrained(tokenizer_name)


def preprocess_cnn_dm(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_len: int = 512,
    max_target_len: int = 128,
) -> Tuple[Dataset, Dataset]:
    """
    Tokenize CNN/DailyMail article/summary pairs for seq2seq training.

    Returns train/val datasets with:
      - input_ids
      - attention_mask
      - labels (pad tokens replaced by -100)
    """

    def build_prompt(article: str) -> str:
        # If we want to use FLAN-style instructions later, uncomment this:
        # return "summarize the following article:\n\n" + article
        return article

    def preprocess_batch(batch):
        # Encoder input: (optionally) instruction + article
        prompts = [build_prompt(a) for a in batch["article"]]
        model_inputs = tokenizer(
            prompts,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )

        # Decoder targets: reference summaries
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["highlights"],
                max_length=max_target_len,
                truncation=True,
                padding="max_length",
            )

        # Ignore padding tokens in the loss by setting them to -100
        pad_id = tokenizer.pad_token_id
        processed_labels = []
        for seq in labels["input_ids"]:
            processed_seq = [
                (tid if tid != pad_id else -100)
                for tid in seq
            ]
            processed_labels.append(processed_seq)

        model_inputs["labels"] = processed_labels
        return model_inputs

    # Apply preprocessing and drop original text columns
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

    # Make datasets return PyTorch tensors
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
    """
    Convenience wrapper:
    - sample a small CNN/DM subset
    - tokenize it
    - return (tokenizer, train_ds, val_ds)
    """
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