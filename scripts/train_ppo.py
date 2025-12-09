# scripts/train_ppo.py
from typing import List, Dict
import os
import torch
from datasets import load_dataset
from transformers import T5Tokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead

from rewards import rouge_l_rewards  # PYTHONPATH=src

def build_ppo_dataset(
    n_train: int = 300,
    min_len: int = 200,
    max_len: int = 1200,
    use_instruction: bool = False,
) -> List[Dict[str, str]]:
    """
    Returns list of {"query": <input text>, "reference": <gold summary>} from CNN/DM train subset.
    """
    raw = load_dataset("cnn_dailymail", "3.0.0", split="train").select(range(n_train))

    def ok(ex):
        L = len(ex["article"])
        return (L >= min_len) and (L <= max_len)

    raw = raw.filter(ok)

    def to_pair(ex):
        article = ex["article"]
        if use_instruction:
            query = f"Summarize the following article.\n\n{article}\n\nSummary:"
        else:
            # match your SFT style (no instruction prefix)
            query = article
        return {"query": query, "reference": ex["highlights"]}

    return [to_pair(ex) for ex in raw]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Choose PPO starting point ===
    # Start from base FLAN-T5-small…
    model_name = "google/flan-t5-small"
    # …or start from your SFT checkpoint:
    # model_name = "./checkpoints/flan_t5_sft_cnn_dm"

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Seq2Seq policy with value head
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)

    # PPO config – keep small for Colab sanity
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=4,               # samples per PPO "batch"
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        target_kl=0.1,              # gentle KL control
        ppo_epochs=4,               # inner PPO epochs per batch
        seed=42,
    )

    # Create PPO trainer (ref model auto-cloned if None)
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=None,  # we'll feed batches manually
    )

    # Build small training list
    data = build_ppo_dataset(n_train=300, use_instruction=False)

    max_new_tokens = 64
    steps_per_epoch = 25  # number of PPO batches per epoch (tune as you like)
    epochs = 1            # bump for longer runs

    for epoch in range(epochs):
        print(f"\n=== PPO Epoch {epoch+1}/{epochs} ===")

        # simple streaming batches
        ptr = 0
        while ptr < len(data) and steps_per_epoch > 0:
            batch = data[ptr: ptr + ppo_config.batch_size]
            ptr += ppo_config.batch_size
            steps_per_epoch -= 1

            queries = [ex["query"] for ex in batch]
            refs    = [ex["reference"] for ex in batch]

            # Tokenize encoder inputs
            enc = tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Generate responses with current policy (batched)
            with torch.no_grad():
                gen = ppo_trainer.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
            response_tensors = list(gen)
            response_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

            # Compute rewards (ROUGE-L per-example)
            rewards = rouge_l_rewards(response_texts, refs)

            # PPO step (TRL expects lists of tensors)
            stats = ppo_trainer.step(
                query_tensors=list(enc["input_ids"]),
                response_tensors=response_tensors,
                rewards=rewards,
            )

            # Simple logging
            mean_reward = torch.stack(rewards).mean().item()
            print(
                f"PPO step: reward={mean_reward:.4f} "
                f"kl={stats.get('objective/kl', float('nan')):.4f} "
                f"loss={stats.get('ppo/mean_policy_loss', float('nan')):.4f}"
            )

    out_dir = "./checkpoints/flan_t5_ppo_cnn_dm"
    os.makedirs(out_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved PPO model to {out_dir}")

if __name__ == "__main__":
    main()