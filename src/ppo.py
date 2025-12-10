# src/ppo.py
"""
Stable & verbose PPO for CNN/DailyMail summarization (base or SFT checkpoint).

Tested with:
  - trl==0.10.1
  - transformers==4.57.x
  - torch==2.x

Key features:
  • Starts from base (flan-t5-small) or your SFT checkpoint
  • Safe generation defaults to reduce NaNs / mode collapse
  • Per-step debug: token lengths, rewards, KL, losses, sample I/O
  • Reward clamping + optional length penalty
  • Skips short final batch to keep PPO happy about batch sizes

Usage (notebook or script):
  from src.ppo import run_ppo
  out_dir = run_ppo(
      model_name="google/flan-t5-small",          # or "./checkpoints/flan_t5_sft_cnn_dm"
      out_dir="checkpoints/flan_t5_ppo_from_base",
      n_train=300, steps_per_epoch=30, epochs=1,
      batch_size=4, debug=True, debug_every=1, print_examples=1
  )
"""

from typing import List, Dict, Optional, Literal
import os, math, torch
from datasets import load_dataset
from transformers import T5Tokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead

# Allow both local and package import layouts
try:
    from rewards import rouge_l_rewards
except ModuleNotFoundError:
    from src.rewards import rouge_l_rewards


# ----------------------------
# Data builder
# ----------------------------
def build_ppo_dataset(
    n_train: int = 300,
    min_len: int = 200,
    max_len: int = 1200,
    use_instruction: bool = False,
) -> List[Dict[str, str]]:
    """
    Returns list of {"query": <encoder input>, "reference": <gold summary>}
    drawn from CNN/DailyMail train subset.
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
            query = article  # matches your SFT formatting
        return {"query": query, "reference": ex["highlights"]}

    return [to_pair(ex) for ex in raw]


# ----------------------------
# Helpers
# ----------------------------
def _device_str(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick(d: Dict, *keys, default=float("nan")):
    """Pick first present key from dict (robust across TRL stats variants)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _len_stats(lengths: List[int]) -> str:
    if not lengths:
        return "n=0"
    t = torch.tensor(lengths, dtype=torch.float32)
    return f"n={len(lengths)} min={int(t.min())} mean={t.mean():.1f} max={int(t.max())} std={t.std(unbiased=False):.1f}"


def _reward_stats(rewards: List[torch.Tensor]) -> str:
    if not rewards:
        return "n=0"
    t = torch.stack([r.float() for r in rewards])
    return f"n={len(rewards)} min={t.min().item():.4f} mean={t.mean().item():.4f} max={t.max().item():.4f} std={t.std(unbiased=False).item():.4f}"


def _first_n(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[:n] + " …"


def _token_lengths(ids: torch.Tensor, pad_id: int) -> List[int]:
    # ids: [B, T]
    with torch.no_grad():
        return [(row != pad_id).sum().item() for row in ids]


def _safe_isfinite(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        raise ValueError(f"Non-finite values detected in {name}")


def _kl_coef_from(trainer: PPOTrainer, stats: Dict) -> float:
    # Try stats first
    val = _pick(stats, "ppo/kl_coef", "train/kl_coef", default=float("nan"))
    if isinstance(val, float) and not math.isnan(val):
        return val
    # Some TRL versions expose a controller
    kl_ctl = getattr(trainer, "kl_ctl", None)
    if kl_ctl is not None:
        for attr in ("value", "kl_coef", "coeff", "coef"):
            v = getattr(kl_ctl, attr, None)
            if isinstance(v, float):
                return v
    return float("nan")


# ----------------------------
# Main PPO entrypoint
# ----------------------------
def run_ppo(
    model_name: str = "google/flan-t5-small",
    out_dir: str = "checkpoints/flan_t5_ppo_from_base",

    # data / batching
    n_train: int = 300,
    steps_per_epoch: int = 30,
    epochs: int = 1,
    batch_size: int = 4,
    remainder: Literal["skip", "wrap"] = "skip",  # keep PPO batch sizes consistent

    # generation (safer defaults)
    max_new_tokens: int = 56,
    min_new_tokens: int = 8,
    no_repeat_ngram_size: int = 4,
    top_k: int = 30,
    top_p: float = 0.85,
    temperature: float = 0.8,
    use_instruction: bool = False,

    # PPO stability knobs
    lr: float = 2e-6,
    ppo_epochs: int = 2,
    target_kl: float = 0.05,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    max_grad_norm: float = 1.0,
    whiten_rewards: bool = False,
    adap_kl_ctrl: bool = True,
    init_kl_coef: float = 0.2,

    # reward shaping
    length_penalty: float = 0.0,
    reward_clamp: float = 0.5,  # clamp to [-0.5, 0.5]

    # debug verbosity
    debug: bool = True,
    debug_every: int = 1,
    print_examples: int = 1,

    # misc
    seed: int = 42,
    device: Optional[str] = None,
) -> str:
    device = _device_str(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Tokenizer & policy with value head
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)

    # TRL config (0.10.1)
    cfg = PPOConfig(
        model_name=model_name,
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=ppo_epochs,
        target_kl=target_kl,
        cliprange=cliprange,
        cliprange_value=cliprange_value,
        max_grad_norm=max_grad_norm,
        whiten_rewards=whiten_rewards,
        adap_kl_ctrl=adap_kl_ctrl,
        init_kl_coef=init_kl_coef,
        seed=seed,
        optimize_cuda_cache=True,
    )

    ppo_trainer = PPOTrainer(
        cfg,
        model,
        ref_model=None,   # TRL clones a frozen reference internally
        tokenizer=tokenizer,
        dataset=None,     # we feed batches manually
    )

    data = build_ppo_dataset(n_train=n_train, use_instruction=use_instruction)

    if debug:
        print(
            f"[PPO] device={device} seed={seed} n_train={len(data)} "
            f"epochs={epochs} steps/epoch={steps_per_epoch} batch_size={batch_size}"
        )
        print(
            f"[PPO] gen kwargs: max_new={max_new_tokens} min_new={min_new_tokens} "
            f"top_k={top_k} top_p={top_p} temp={temperature} no_repeat={no_repeat_ngram_size}"
        )
        print(
            f"[PPO] kl target={target_kl} init_kl_coef={init_kl_coef} "
            f"adap_kl={adap_kl_ctrl} whiten_rewards={whiten_rewards}"
        )

    step_idx_global = 0

    for ep in range(epochs):
        print(f"\n=== PPO Epoch {ep + 1}/{epochs} ===")
        ptr, left = 0, steps_per_epoch

        while ptr < len(data) and left > 0:
            end = ptr + batch_size
            batch = data[ptr:end]
            ptr = end
            left -= 1
            step_idx_global += 1

            if len(batch) < batch_size:
                if remainder == "skip":
                    if debug:
                        print(f"[PPO][step {step_idx_global}] Skipping short last batch len={len(batch)}.")
                    break
                else:
                    need = batch_size - len(batch)
                    batch = batch + data[:need]

            queries = [ex["query"] for ex in batch]
            refs    = [ex["reference"] for ex in batch]

            # Encode inputs
            enc = tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            _safe_isfinite("queries/input_ids", enc["input_ids"])
            q_lens = _token_lengths(enc["input_ids"], tokenizer.pad_token_id)

            # Generate with current policy (batched)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response_tensors = [g for g in gen]
            response_texts   = tokenizer.batch_decode(gen, skip_special_tokens=True)

            # Response lengths
            resp_ids = torch.stack(response_tensors, dim=0)
            _safe_isfinite("responses", resp_ids)
            r_lens = _token_lengths(resp_ids, tokenizer.pad_token_id)

            # Rewards (ROUGE-L) + clamp + finite
            rewards = rouge_l_rewards(response_texts, refs, length_penalty=length_penalty)
            rewards = [torch.clamp(r.float(), -reward_clamp, reward_clamp) for r in rewards]
            rewards = [torch.where(torch.isfinite(r), r, torch.zeros_like(r)) for r in rewards]
            _ = [_safe_isfinite("reward", r) for r in rewards]

            # PPO update (lists of tensors; EXACT batch_size each)
            stats = ppo_trainer.step(
                [q for q in enc["input_ids"]],  # queries
                response_tensors,               # responses
                rewards,                        # scores
            )

            # Extract robust metrics across TRL versions
            mean_reward = torch.stack(rewards).mean().item()
            kl = _pick(stats, "objective/kl", "ppo/policy/kl", "kl")
            pol_loss = _pick(stats, "ppo/loss/policy", "train/mean_policy_loss", "ppo/mean_policy_loss")
            ent = _pick(stats, "ppo/policy/entropy", "policy/entropy")
            val_loss = _pick(stats, "ppo/loss/value", "train/mean_value_loss", "ppo/mean_value_loss")
            kl_coef = _kl_coef_from(ppo_trainer, stats)

            # Structured debug prints
            if debug and (step_idx_global % max(1, debug_every) == 0):
                print(f"[PPO][step {step_idx_global}] "
                      f"reward_mean={mean_reward:.4f} "
                      f"kl={kl:.4f} kl_coef={kl_coef:.4f} "
                      f"policy_loss={pol_loss:.4f} value_loss={val_loss:.4f} entropy={ent:.4f}")
                print(f"[PPO][step {step_idx_global}] query token lengths:   {_len_stats(q_lens)}")
                print(f"[PPO][step {step_idx_global}] response token lengths:{_len_stats(r_lens)}")
                print(f"[PPO][step {step_idx_global}] reward stats:         {_reward_stats(rewards)}")

                for i in range(min(print_examples, len(batch))):
                    print(f"--- example {i} ---")
                    print("[QUERY ]", _first_n(queries[i], 240))
                    print("[REF   ]", _first_n(refs[i], 200))
                    print("[OUTPUT]", _first_n(response_texts[i], 200))
                    print(f"[REWARD] {rewards[i].item():.4f}")

            # Flag negative KL clearly (often generation/top_p/temperature issue)
            try:
                if isinstance(kl, float) and kl < -1e-3:
                    print(f"[warn][step {step_idx_global}] KL is negative ({kl:.4f}). "
                          f"Consider lowering temperature/top_p or increasing target_kl/init_kl_coef.")
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved PPO model to {out_dir}")
    return out_dir