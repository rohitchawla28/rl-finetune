# src/ppo.py
from typing import List, Dict, Optional
import os
import dataclasses
import torch
from datasets import load_dataset
from transformers import T5Tokenizer

# --- TRL imports: prefer experimental, fallback to classic ---
try:
    from trl.experimental.ppo import PPOConfig as _PPOConfig, PPOTrainer as _PPOTrainer
    from trl import AutoModelForSeq2SeqLMWithValueHead
except Exception:
    from trl import PPOConfig as _PPOConfig, PPOTrainer as _PPOTrainer, AutoModelForSeq2SeqLMWithValueHead

# --- rewards import works both with PYTHONPATH=src and src. prefix ---
try:
    from rewards import rouge_l_rewards
except ModuleNotFoundError:
    from src.rewards import rouge_l_rewards


def build_ppo_dataset(
    n_train: int = 300,
    min_len: int = 200,
    max_len: int = 1200,
    use_instruction: bool = False,
) -> List[Dict[str, str]]:
    """
    Returns a list of {"query": <input text>, "reference": <gold summary>}
    from a subset of CNN/DailyMail train.
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
            query = article  # match your SFT/no-instruction style
        return {"query": query, "reference": ex["highlights"]}

    return [to_pair(ex) for ex in raw]


def _build_version_safe_ppo_config(
    *, lr: float, batch_size: int, ppo_epochs: int, target_kl: float, seed: int
) -> _PPOConfig:
    """
    Create a PPOConfig including only fields supported by the installed TRL version.
    """
    wanted = dict(
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,  # filtered out automatically if not supported
        ppo_epochs=ppo_epochs,
        target_kl=target_kl,
        seed=seed,
    )
    allowed = {f.name for f in dataclasses.fields(_PPOConfig)}
    safe_kwargs = {k: v for k, v in wanted.items() if k in allowed}
    return _PPOConfig(**safe_kwargs)


def _make_trainer(cfg: _PPOConfig, model, tokenizer):
    """
    Instantiate PPOTrainer with whatever signature this TRL build expects.
    Tries (config=...), then (ppo_config=...), then positional.
    """
    try:
        return _PPOTrainer(config=cfg, model=model, ref_model=None, tokenizer=tokenizer, dataset=None)
    except TypeError:
        try:
            return _PPOTrainer(ppo_config=cfg, model=model, ref_model=None, tokenizer=tokenizer, dataset=None)
        except TypeError:
            return _PPOTrainer(cfg, model, None, tokenizer=tokenizer, dataset=None)


def run_ppo(
    model_name: str = "google/flan-t5-small",           # or "./checkpoints/flan_t5_sft_cnn_dm"
    out_dir: str = "./checkpoints/flan_t5_ppo_cnn_dm",
    n_train: int = 300,
    steps_per_epoch: int = 25,
    epochs: int = 1,
    max_new_tokens: int = 64,
    length_penalty: float = 0.0,
    use_instruction: bool = False,
    lr: float = 1e-5,
    batch_size: int = 4,
    ppo_epochs: int = 4,
    target_kl: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
) -> str:
    """
    PPO fine-tuning for summarization with ROUGE-L rewards.
    Saves model + tokenizer to `out_dir` and returns that path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Tokenizer & value-headed policy
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)

    # Version-safe PPO config
    cfg = _build_version_safe_ppo_config(
        lr=lr, batch_size=batch_size, ppo_epochs=ppo_epochs, target_kl=target_kl, seed=seed
    )

    # Trainer (ref model auto-cloned if None)
    ppo_trainer = _make_trainer(cfg, model, tokenizer)

    # Data
    data = build_ppo_dataset(n_train=n_train, use_instruction=use_instruction)
    effective_bs = batch_size  # don't rely on cfg.batch_size existing in all TRL versions

    for ep in range(epochs):
        print(f"\n=== PPO Epoch {ep + 1}/{epochs} ===")
        ptr, left = 0, steps_per_epoch

        while ptr < len(data) and left > 0:
            batch = data[ptr: ptr + effective_bs]
            ptr += effective_bs
            left -= 1

            queries = [ex["query"] for ex in batch]
            refs    = [ex["reference"] for ex in batch]

            enc = tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                gen = ppo_trainer.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )

            response_tensors = list(gen)
            response_texts   = tokenizer.batch_decode(gen, skip_special_tokens=True)

            rewards = rouge_l_rewards(response_texts, refs, length_penalty=length_penalty)

            stats = ppo_trainer.step(
                query_tensors=list(enc["input_ids"]),
                response_tensors=response_tensors,
                rewards=rewards,
            )

            mean_reward = torch.stack(rewards).mean().item()
            print(
                f"reward={mean_reward:.4f} "
                f"kl={stats.get('objective/kl', float('nan')):.4f} "
                f"loss={stats.get('ppo/mean_policy_loss', float('nan')):.4f}"
            )

    os.makedirs(out_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved PPO model to {out_dir}")
    return out_dir