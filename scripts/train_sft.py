# train_sft.py
from src.data import load_cnn_dm_tokenized
from src.sft import run_sft


def main():
    tokenizer, train_ds, val_ds = load_cnn_dm_tokenized(
        n_train=500,
        n_val=100,
        tokenizer_name="google/flan-t5-small",
        max_input_len=512,
        max_target_len=128,
    )

    model, history = run_sft(
        train_dataset=train_ds,
        val_dataset=val_ds,
        tokenizer=tokenizer,
        model_name="google/flan-t5-small",
        batch_size=4,
        num_epochs=1,
        lr=5e-5,
        warmup_ratio=0.1,
        output_dir="./checkpoints/flan_t5_sft_cnn_dm",
    )

    print(history)


if __name__ == "__main__":
    main()