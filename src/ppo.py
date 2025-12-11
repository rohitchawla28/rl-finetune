# src/ppo.py
"""
PPO for CNN/DailyMail summarization with TRL-generate compatibility.

- Filters-by-length FIRST, then shuffle/select n_train
- Uses PPOTrainer.generate(...) (new or old signature) to keep policy/ref aligned
- Ensures attention_mask is 2-D for T5 (fixes IndexError)
- Flattens responses to 1-D id lists before batch_decode (fixes TypeError)
- Safe generation defaults; reward clamp; optional CSV logging
"""

from typing import List, Dict, Optional, Literal
import os, math, csv, json, torch, inspect
from datasets import load_dataset
from transformers import T5Tokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from tqdm.auto import tqdm

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

def _ensure_2d_mask(am: Optional[torch.Tensor], i: Optional[int] = None) -> Optional[torch.Tensor]:
    """Return a 2D attention mask [B,T]. If i is given, slice to [1,T]."""
    if am is None:
        return None
    if i is None:
        return am if am.ndim == 2 else am.unsqueeze(0)
    m = am[i]
    return m if m.ndim == 2 else m.unsqueeze(0)

def _to_1d_id_list(x):
    """Normalize any response (tensor or nested list) to a flat 1-D list[int]."""
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.ndim == 2 and x.size(0) == 1:
            x = x.squeeze(0)
        elif x.ndim > 2:
            x = x.view(-1)
        return x.tolist()
    # list/tuple path: peel leading singleton dims ([[ids]] -> [ids])
    while isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)

def _ppo_generate(ppo_trainer: PPOTrainer, enc: Dict[str, torch.Tensor], **gen_kwargs):
    """
    Optimized generation: Use model directly for faster batched generation.
    TRL's generate can be slow, so we bypass it and use the underlying model.
    Returns list[tensor] (one response per example) - only generated tokens.
    """
    # Use the model directly for faster batched generation
    model = ppo_trainer.model
    if hasattr(model, 'pretrained_model'):
        model = model.pretrained_model  # Unwrap value head to get base model
    
    model.eval()
    with torch.no_grad():
        # Generate directly - much faster than TRL's generate
        # For T5, generate() returns full sequence (input + generated)
        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            **gen_kwargs,
        )
    
    # Extract only the generated tokens (remove input prompt)
    # T5 generate returns [input_ids, generated_ids], so we slice
    input_len = enc["input_ids"].shape[1]
    output_len = outputs.shape[1]
    
    # Handle case where output is same length as input (no generation happened)
    if output_len <= input_len:
        # Return empty tensors - this will be filtered out later
        return [torch.tensor([], dtype=torch.long, device=outputs.device) for _ in range(outputs.size(0))]
    
    generated = outputs[:, input_len:].contiguous()
    
    # Return as list of tensors (one per example)
    # Let decoding handle special tokens - don't filter here
    return [generated[i] for i in range(generated.size(0))]


# ----------------------------
# Main entrypoint
# ----------------------------
def run_ppo(
    model_name: str = "google/flan-t5-small",
    out_dir: str = "checkpoints/flan_t5_ppo_from_base",

    pairs: Optional[List[Dict[str, str]]] = None,  # optional prebuilt {query, reference}

    # data controls (used only if pairs is None)
    n_train: int = 300,
    min_len: int = 200,
    max_len: int = 1200,
    use_instruction: bool = False,
    seed: int = 42,

    # schedule
    batch_size: int = 4,
    steps_per_epoch: Optional[int] = None,   # None => ceil(len(data)/bs)
    epochs: int = 1,
    remainder: Literal["skip", "wrap"] = "skip",

    # generation (safe defaults)
    max_new_tokens: int = 64,
    min_new_tokens: int = 8,
    no_repeat_ngram_size: int = 4,
    top_k: int = 30,
    top_p: float = 0.75,
    temperature: float = 0.7,

    # PPO knobs
    lr: float = 1e-6,
    ppo_epochs: int = 2,
    target_kl: float = 0.1,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    max_grad_norm: float = 1.0,
    whiten_rewards: bool = False,
    adap_kl_ctrl: bool = True,
    init_kl_coef: float = 0.4,

    # rewards
    length_penalty: float = 0.0,
    reward_clamp: float = 1.0,  # Less aggressive clamping
    reward_fn=None,
    reward_kwargs: Optional[dict] = None,

    # logging
    verbosity: Literal["silent", "summary", "steps"] = "summary",
    tqdm_bar: bool = True,
    log_csv: Optional[str] = None,

    # misc
    device: Optional[str] = None,
) -> str:
    device = _device_str(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    LVL = {"silent": 0, "summary": 1, "steps": 2}
    V = LVL[verbosity]

    # tokenizer + policy/value (fp32 for stability)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy model (with value head for PPO)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)
    
    # Reference model: TRL will use the initial policy model state as reference
    # when ref_model=None. This is the standard approach and avoids needing
    # a separate wrapped model.
    ref_model = None

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
    ppo_trainer = PPOTrainer(cfg, model, ref_model=ref_model, tokenizer=tokenizer, dataset=None)

    # data
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
    auto_steps = math.ceil(data_len / batch_size)
    steps_per_epoch = auto_steps if steps_per_epoch is None else min(steps_per_epoch, auto_steps)
    if V >= 1:
        print(f"[PPO] device={device} n_train={data_len} epochs={epochs} steps/epoch={steps_per_epoch} bs={batch_size}")

    # rewards
    if reward_fn is None:
        reward_fn = lambda preds, refs, **kw: rouge_l_rewards(preds, refs, length_penalty=length_penalty)
    rw_kwargs = reward_kwargs or {}

    # gen kwargs (passed to TRL)
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

    # CSV logging
    csv_writer = None
    csv_f = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        csv_f = open(log_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_f, fieldnames=[
            "epoch","step","mean_reward","kl","policy_loss","value_loss","entropy","kl_coef"
        ])
        csv_writer.writeheader()

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
                need = batch_size - len(batch)
                batch = batch + data[:need]

            queries = [ex["query"] for ex in batch]
            refs    = [ex["reference"] for ex in batch]

            # encode
            enc = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            _safe_isfinite("queries/input_ids", enc["input_ids"])

            # generate via TRL (signature-safe)
            responses = _ppo_generate(ppo_trainer, enc, **gen_kwargs)   # -> list[tensor]
            for i, r in enumerate(responses):
                _safe_isfinite(f"responses[{i}]", r)

            # decode texts (force 1-D id lists)
            decode_in = [_to_1d_id_list(r) for r in responses]
            response_texts = tokenizer.batch_decode(decode_in, skip_special_tokens=True)
            
            # Debug: Print generation info on first step
            if step_idx == 1 and V >= 2:
                print(f"[PPO DEBUG] Generated {len(responses)} responses")
                for i, (resp_tensor, txt) in enumerate(zip(responses, response_texts)):
                    resp_len = len(resp_tensor) if isinstance(resp_tensor, torch.Tensor) else len(resp_tensor)
                    print(f"  Response {i}: len={resp_len}, text_len={len(txt)}, preview='{txt[:50]}...'")
                    if resp_len > 0:
                        print(f"    First 5 tokens: {resp_tensor[:5] if isinstance(resp_tensor, torch.Tensor) else resp_tensor[:5]}")
            
            # Filter out empty generations and handle them
            valid_indices = []
            valid_texts = []
            valid_refs = []
            valid_queries = []
            valid_responses = []
            valid_enc_input_ids = []
            
            for i, (txt, ref, q, resp) in enumerate(zip(response_texts, refs, queries, responses)):
                if txt and txt.strip():  # Non-empty text
                    valid_indices.append(i)
                    valid_texts.append(txt)
                    valid_refs.append(ref)
                    valid_queries.append(q)
                    valid_responses.append(resp)
                    valid_enc_input_ids.append(enc["input_ids"][i])
            
            if not valid_texts:
                if V >= 2:
                    print(f"[PPO] Warning: All generations empty at step {step_idx}, skipping batch")
                if pbar is not None:
                    pbar.update(1)
                continue
            
            # Check if we have enough valid examples for the batch size
            if len(valid_texts) < batch_size:
                if V >= 2:
                    print(f"[PPO] Warning: Only {len(valid_texts)} valid examples after filtering (need {batch_size}), skipping batch")
                if pbar is not None:
                    pbar.update(1)
                continue
            
            # Only process valid examples
            if len(valid_texts) < len(response_texts):
                if V >= 2:
                    print(f"[PPO] Warning: {len(response_texts) - len(valid_texts)} empty generations at step {step_idx}")
                # Update to only valid examples
                queries = valid_queries
                responses = valid_responses
                refs = valid_refs
                response_texts = valid_texts
                # Re-encode only valid queries
                enc = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Extract 1D query tensors (enc["input_ids"] is [batch_size, seq_len], so [i] gives 1D)
            # Ensure they're on the correct device and have consistent dtype
            query_tensors = []
            for i in range(enc["input_ids"].size(0)):
                q = enc["input_ids"][i].clone().to(device)
                query_tensors.append(q)

            # Ensure responses are 1D tensors (list of tensors, each 1D)
            # All must be on same device with consistent dtype
            response_tensors = []
            for resp in responses:
                if torch.is_tensor(resp):
                    # Flatten to 1D if needed
                    if resp.ndim == 0:
                        resp = resp.unsqueeze(0)
                    elif resp.ndim > 1:
                        resp = resp.view(-1)
                    # Ensure correct device and dtype
                    resp = resp.clone().detach().to(device).long()
                    response_tensors.append(resp)
                elif isinstance(resp, (list, tuple)):
                    # Convert list to 1D tensor
                    response_tensors.append(torch.tensor(resp, dtype=torch.long, device=device))
                else:
                    # Single value
                    response_tensors.append(torch.tensor([resp], dtype=torch.long, device=device))

            # Final safety check: ensure we have exactly batch_size examples
            if len(query_tensors) != batch_size or len(response_tensors) != batch_size:
                if V >= 2:
                    print(f"[PPO] Warning: Mismatch - queries={len(query_tensors)}, responses={len(response_tensors)}, batch_size={batch_size}, skipping")
                if pbar is not None:
                    pbar.update(1)
                continue

            # rewards
            rewards = reward_fn(response_texts, refs, **rw_kwargs)
            rewards = [torch.clamp(r.float(), -reward_clamp, reward_clamp) for r in rewards]
            rewards = [torch.where(torch.isfinite(r), r, torch.zeros_like(r)) for r in rewards]
            
            # Final check: rewards must match batch size
            if len(rewards) != batch_size:
                if V >= 2:
                    print(f"[PPO] Warning: Rewards count ({len(rewards)}) != batch_size ({batch_size}), skipping")
                if pbar is not None:
                    pbar.update(1)
                continue

            # PPO update - pass 1D tensors
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            mean_reward = torch.stack(rewards).mean().item()
            
            # Debug: Print stats on first and last step to verify training
            if step_idx == 1 and V >= 1:
                reward_std = torch.stack(rewards).std().item()
                print(f"[PPO] Step 1: mean_reward={mean_reward:.4f}, std={reward_std:.4f}")
            
            kl = _pick(stats, "objective/kl", "ppo/policy/kl", "kl")
            pol_loss = _pick(stats, "ppo/loss/policy", "train/mean_policy_loss", "ppo/mean_policy_loss")
            val_loss = _pick(stats, "ppo/loss/value", "train/mean_value_loss", "ppo/mean_value_loss")
            ent = _pick(stats, "ppo/policy/entropy", "policy/entropy")
            kl_coef = _pick(stats, "ppo/kl_coef", "train/kl_coef")
            
            # Debug: Verify training is happening
            if step_idx == 1 and V >= 1:
                print(f"[PPO] Step 1 training: kl={kl:.4f}, policy_loss={pol_loss:.4f}, entropy={ent:.4f}")
            if step_idx == steps_per_epoch and V >= 1:
                print(f"[PPO] Step {step_idx} (final): kl={kl:.4f}, policy_loss={pol_loss:.4f}, entropy={ent:.4f}")

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

    # save
    os.makedirs(out_dir, exist_ok=True)
    # CRITICAL: Extract the pretrained model (without value head) for saving
    # The value head is only for PPO training, we want to save the actual language model
    if hasattr(ppo_trainer.model, 'pretrained_model'):
        # Save the base model (this is what gets updated during PPO)
        base_model = ppo_trainer.model.pretrained_model
        base_model.save_pretrained(out_dir)
        if V >= 1:
            print(f"[PPO] Saved pretrained model (without value head) to {out_dir}")
    else:
        # Fallback: save full model if structure is different
        ppo_trainer.model.save_pretrained(out_dir)
        if V >= 1:
            print(f"[PPO] Saved full model to {out_dir}")
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
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size,
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