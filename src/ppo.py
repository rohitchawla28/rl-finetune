# src/ppo.py
"""
PPO for CNN/DailyMail summarization with proper length filtering and TRL generation.

Key points
- Filter-by-length FIRST, then shuffle & select n_train
- Uses PPOTrainer.generate(...) to keep policy/ref aligned (reduces negative-KL)
- Stays in fp32 for stability (no bf16 cast here)
- Auto steps_per_epoch if None
- Optional auto-tuning of sampling if KL becomes negative

Tested with: trl==0.10.1, transformers==4.57.x, torch==2.x
"""

from typing import List, Dict, Optional, Literal
import os, math, csv, json, torch
from datasets import load_dataset
from transformers import T5Tokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from tqdm.auto import tqdm

# ----- Rewards import (repo-local or package layout) -----
try:
    from rewards import rouge_l_rewards
except ModuleNotFoundError:
    from src.rewards import rouge_l_rewards


# ----------------------------
# Helpers
# ----------------------------
def _device_str(device: Optional[str] = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")

def _pick(d: Dict, *keys, default=float("nan")):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _safe_isfinite(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        raise ValueError(f"Non-finite values detected in {name}")

def _filter_by_len(ds, min_len: int, max_len: int):
    def ok(ex):
        L = len(ex["article"])
        return (L >= min_len) and (L <= max_len)
    return ds.filter(ok)


# ----------------------------
# Main entrypoint
# ----------------------------
def run_ppo(
    model_name: str = "google/flan-t5-small",
    out_dir: str = "checkpoints/flan_t5_ppo_from_base",

    # You can pass prebuilt {query, reference} pairs to bypass internal loading
    pairs: Optional[List[Dict[str, str]]] = None,

    # Data controls (used only if `pairs` is None)
    n_train: int = 300,
    min_len: int = 200,
    max_len: int = 1200,
    use_instruction: bool = False,
    seed: int = 42,

    # Batching / schedule
    batch_size: int = 4,
    steps_per_epoch: Optional[int] = None,   # if None -> ceil(len(data)/bs)
    epochs: int = 1,
    remainder: Literal["skip", "wrap"] = "skip",

    # Generation (safe defaults)
    max_new_tokens: int = 56,
    min_new_tokens: int = 8,
    no_repeat_ngram_size: int = 4,
    top_k: int = 30,
    top_p: float = 0.75,
    temperature: float = 0.7,

    # PPO knobs
    lr: float = 1e-6,
    ppo_epochs: int = 2,
    target_kl: float = 0.08,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    max_grad_norm: float = 1.0,
    whiten_rewards: bool = False,
    adap_kl_ctrl: bool = True,
    init_kl_coef: float = 0.4,   # slightly stronger KL pull to start

    # Rewards
    length_penalty: float = 0.0,
    reward_clamp: float = 0.5,   # clamp each reward to [-0.5, 0.5]
    reward_fn=None,
    reward_kwargs: Optional[dict] = None,

    # Logging / output
    verbosity: Literal["silent", "summary", "steps"] = "summary",
    tqdm_bar: bool = True,
    log_csv: Optional[str] = None,

    # Safety tweak: tighten sampling a bit if KL goes negative
    auto_tighten_on_neg_kl: bool = True,

    # Misc
    device: Optional[str] = None,
) -> str:
    device = _device_str(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Verbosity
    LVL = {"silent": 0, "summary": 1, "steps": 2}
    V = LVL[verbosity]

    # Tokenizer & policy (fp32)
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
    ppo_trainer = PPOTrainer(cfg, model, ref_model=None, tokenizer=tokenizer, dataset=None)

    # Data
    if pairs is None:
        raw = load_dataset("cnn_dailymail", "3.0.0", split="train")
        raw = _filter_by_len(raw, min_len=min_len, max_len=max_len)
        survivors = len(raw)
        raw = raw.shuffle(seed=seed)
        take = min(n_train, survivors)
        raw = raw.select(range(take))
        if use_instruction:
            def make_q(a): return f"Summarize the following article.\n\n{a}\n\nSummary:"
        else:
            def make_q(a): return a
        data = [{"query": make_q(ex["article"]), "reference": ex["highlights"]} for ex in raw]
        if V >= 1:
            print(f"[PPO] survivors after filter(min_len={min_len}, max_len={max_len}) = {survivors}")
            print(f"[PPO] taking n_train={take}")
    else:
        data = pairs
        if V >= 1:
            print(f"[PPO] using provided pairs n={len(data)}")

    data_len = len(data)
    # Schedule
    auto_steps = math.ceil(data_len / batch_size)
    steps_per_epoch = auto_steps if steps_per_epoch is None else min(steps_per_epoch, auto_steps)
    if V >= 1:
        print(f"[PPO] device={device} n_train={data_len} epochs={epochs} steps/epoch={steps_per_epoch} bs={batch_size}")

    # Reward fn
    if reward_fn is None:
        reward_fn = lambda preds, refs, **kw: rouge_l_rewards(preds, refs, length_penalty=length_penalty)
    rw_kwargs = reward_kwargs or {}

    # CSV logger
    csv_writer = None
    csv_f = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        csv_f = open(log_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_f, fieldnames=[
            "epoch","step","mean_reward","kl","policy_loss","value_loss","entropy","kl_coef"
        ])
        csv_writer.writeheader()

    # Generation kwargs (mutable so we can tighten if needed)
    gen_kwargs = dict(
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

    total_steps = epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, disable=(V == 0 or not tqdm_bar), leave=False)

    step_idx = 0
    for ep in range(epochs):
        if V >= 1:
            print(f"\n=== PPO Epoch {ep+1}/{epochs} ===")

        ptr, left = 0, steps_per_epoch
        while ptr < data_len and left > 0:
            batch = data[ptr: ptr + batch_size]
            ptr += batch_size
            left -= 1
            step_idx += 1

            if len(batch) < batch_size:
                if remainder == "skip":
                    if V >= 2:
                        print(f"[PPO] Skipping short batch len={len(batch)}")
                    break
                # else wrap
                need = batch_size - len(batch)
                batch = batch + data[:need]

            queries = [ex["query"] for ex in batch]
            refs    = [ex["reference"] for ex in batch]

            # Encode
            enc = tokenizer(
                queries, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            _safe_isfinite("queries/input_ids", enc["input_ids"])

            # Generate with TRL (keeps policy/ref aligned)
            gen = ppo_trainer.generate(
                query_tensors=[q for q in enc["input_ids"]],
                attention_mask=enc.get("attention_mask", None),
                return_prompt=False,
                **gen_kwargs
            )
            response_tensors = [g for g in gen]
            response_texts   = tokenizer.batch_decode(gen, skip_special_tokens=True)
            _safe_isfinite("responses", torch.stack(response_tensors, dim=0))

            # Rewards
            rewards = reward_fn(response_texts, refs, **rw_kwargs)
            rewards = [torch.clamp(r.float(), -reward_clamp, reward_clamp) for r in rewards]
            rewards = [torch.where(torch.isfinite(r), r, torch.zeros_like(r)) for r in rewards]

            # PPO step
            stats = ppo_trainer.step(
                [q for q in enc["input_ids"]],
                response_tensors,
                rewards,
            )

            # Metrics
            mean_reward = torch.stack(rewards).mean().item()
            kl = _pick(stats, "objective/kl", "ppo/policy/kl", "kl")
            pol_loss = _pick(stats, "ppo/loss/policy", "train/mean_policy_loss", "ppo/mean_policy_loss")
            val_loss = _pick(stats, "ppo/loss/value", "train/mean_value_loss", "ppo/mean_value_loss")
            ent = _pick(stats, "ppo/policy/entropy", "policy/entropy")
            # Try to get KL coef from stats, otherwise from trainer
            kl_coef = _pick(stats, "ppo/kl_coef", "train/kl_coef")
            if isinstance(kl_coef, float) and math.isnan(kl_coef):
                kl_ctl = getattr(ppo_trainer, "kl_ctl", None)
                for attr in ("value", "kl_coef", "coeff", "coef"):
                    v = getattr(kl_ctl, attr, None) if kl_ctl is not None else None
                    if isinstance(v, float):
                        kl_coef = v
                        break

            # Auto-tighten on negative KL (optional)
            if auto_tighten_on_neg_kl and isinstance(kl, float) and kl < 0.0:
                gen_kwargs["top_p"] = max(0.45, float(gen_kwargs["top_p"]) - 0.05)
                gen_kwargs["temperature"] = max(0.5, float(gen_kwargs["temperature"]) - 0.05)

            if V >= 2:
                print(f"[step {step_idx}] reward={mean_reward:.4f} kl={kl:.4f} "
                      f"policy_loss={pol_loss:.4f} value_loss={val_loss:.4f} "
                      f"entropy={ent:.4f} kl_coef={kl_coef if isinstance(kl_coef, float) else float('nan'):.4f}")

            if csv_writer is not None:
                csv_writer.writerow({
                    "epoch": ep+1, "step": step_idx,
                    "mean_reward": f"{mean_reward:.6f}",
                    "kl": f"{kl:.6f}" if isinstance(kl, float) else "",
                    "policy_loss": f"{pol_loss:.6f}" if isinstance(pol_loss, float) else "",
                    "value_loss": f"{val_loss:.6f}" if isinstance(val_loss, float) else "",
                    "entropy": f"{ent:.6f}" if isinstance(ent, float) else "",
                    "kl_coef": f"{kl_coef:.6f}" if isinstance(kl_coef, float) else "",
                })

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
    if csv_f is not None:
        csv_f.close()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "ppo_run_config.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "n_train": data_len,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "batch_size": batch_size,
            "min_len": min_len,
            "max_len": max_len,
            "gen": dict(
                max_new_tokens=int(gen_kwargs["max_new_tokens"]),
                min_new_tokens=int(gen_kwargs["min_new_tokens"]),
                top_k=int(gen_kwargs["top_k"]),
                top_p=float(gen_kwargs["top_p"]),
                temperature=float(gen_kwargs["temperature"]),
                no_repeat_ngram_size=int(gen_kwargs["no_repeat_ngram_size"]),
            ),
            "ppo": dict(
                lr=lr, ppo_epochs=ppo_epochs, target_kl=target_kl,
                cliprange=cliprange, cliprange_value=cliprange_value,
                max_grad_norm=max_grad_norm, init_kl_coef=init_kl_coef,
                adap_kl_ctrl=adap_kl_ctrl, whiten_rewards=whiten_rewards,
            ),
        }, f, indent=2)

    print(f"\nSaved PPO model to {out_dir}")
    return out_dir