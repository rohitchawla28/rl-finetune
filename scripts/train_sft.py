# scripts/train_sft.py
from ..data.datasets import load_cnn_dm_tokenized
from ..train.sft import run_sft

def main():
    tok, train_ds, val_ds = load_cnn_dm_tokenized(n_train=500, n_val=100,
                                                  tokenizer_name="google/flan-t5-small",
                                                  max_input_len=512, max_target_len=128)
    model, history = run_sft(train_ds, val_ds, tok, model_name="google/flan-t5-small",
                             output_dir="./checkpoints/flan_t5_sft_cnn_dm",
                             batch_size=4, num_epochs=1, lr=5e-5, warmup_ratio=0.1)
    print(history)

if __name__ == "__main__":
    main()