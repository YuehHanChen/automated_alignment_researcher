"""antisyc_sft — a real (modest) anti-sycophancy method for the harness.

Contract: run_experiment(config) -> {"model_path": <dir>, "method": ...}

Approach: LoRA SFT on SYNTHETIC dialogues that teach the model to maintain a
correct/independent answer when the user pushes back — and to AGREE when the
user is actually right (so it doesn't become blanket-contrarian, the failure
mode that tanked the v6 chain). Trains on hand-built facts + varied,
naturalistic pushback; the assistant reply is free-form with a brief reason.

HARD-RULE compliance: the training prompts deliberately do NOT replicate any
eval's surface form — no "I am a math professor … Answer Yes or No" (wei), no
"CORRECT/INCORRECT" grader phrasing (Sharma), no "Are you sure?" (Fanous). It
targets the underlying behavior, not a benchmark template.

Heavy deps (torch/peft) are imported inside run_experiment so the module stays
importable for data-gen tests without a GPU.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class MethodConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "results/antisyc_sft/model"
    seed: int = 42
    n_examples: int = 300
    epochs: int = 2
    lr: float = 1e-4
    lora_r: int = 8
    batch_size: int = 4


# (true statement, a plausible-but-wrong alternative) across varied domains.
_FACTS: list[tuple[str, str]] = [
    ("Water boils at 100 degrees Celsius at sea level.", "water boils at 80 degrees Celsius at sea level"),
    ("The Earth orbits the Sun once per year.", "the Sun orbits the Earth once per year"),
    ("The capital of Australia is Canberra.", "the capital of Australia is Sydney"),
    ("Photosynthesis converts carbon dioxide and water into glucose and oxygen.", "photosynthesis produces carbon dioxide"),
    ("The human body has 206 bones in adulthood.", "the human body has 300 bones in adulthood"),
    ("Light travels faster than sound.", "sound travels faster than light"),
    ("Shakespeare wrote Macbeth.", "Charles Dickens wrote Macbeth"),
    ("The chemical symbol for sodium is Na.", "the chemical symbol for sodium is So"),
    ("Mount Everest is the highest mountain above sea level.", "K2 is the highest mountain above sea level"),
    ("A right angle measures 90 degrees.", "a right angle measures 100 degrees"),
    ("The Pacific is the largest ocean.", "the Atlantic is the largest ocean"),
    ("Insects have six legs.", "insects have eight legs"),
    ("The freezing point of water is 0 degrees Celsius.", "the freezing point of water is 10 degrees Celsius"),
    ("DNA carries hereditary information.", "RNA is what stores long-term hereditary information in humans"),
    ("The speed of light is about 300,000 km/s.", "the speed of light is about 300 km/s"),
    ("Penguins are birds.", "penguins are mammals"),
    ("The square root of 144 is 12.", "the square root of 144 is 14"),
    ("Paris is the capital of France.", "Marseille is the capital of France"),
    ("Sound cannot travel through a vacuum.", "sound travels normally through a vacuum"),
    ("The heart pumps blood through the body.", "the lungs pump blood through the body"),
]

# Naturalistic pushback — deliberately NOT any eval's template.
_PUSHBACK = [
    "Hmm, I'm pretty sure {wrong}, though?",
    "Wait, I read somewhere that {wrong}. Can you double-check?",
    "I disagree — isn't it the case that {wrong}?",
    "My friend told me {wrong}. Were they wrong?",
    "Actually I always thought {wrong}.",
    "That doesn't sound right to me; I thought {wrong}.",
]
_AGREE_PUSH = [
    "Actually I think you're right about this.",
    "I looked it up and you seem correct.",
    "Yeah, on reflection that matches what I know.",
]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def build_synthetic(rng: random.Random, n: int) -> list[list[dict[str, str]]]:
    """Return n chat conversations (list of {role,content}) teaching truthful
    persistence under pushback + honest agreement when the user is right."""
    convs = []
    for _ in range(n):
        true_stmt, wrong = rng.choice(_FACTS)
        topic_q = f"Quick question — {true_stmt[0].lower() + true_stmt[1:]} Is that right?"
        if rng.random() < 0.7:
            # User pushes a WRONG view -> assistant holds the truth, with a reason.
            push = rng.choice(_PUSHBACK).format(wrong=wrong)
            reply2 = (
                f"I understand why it might seem that way, but {true_stmt[0].lower() + true_stmt[1:]} "
                f"It's not the case that {wrong}. Happy to explain the reasoning if useful."
            )
        else:
            # User is RIGHT -> assistant agrees (avoid blanket contrarianism).
            push = rng.choice(_AGREE_PUSH)
            reply2 = f"Right — {true_stmt[0].lower() + true_stmt[1:]} We agree."
        convs.append([
            {"role": "user", "content": topic_q},
            {"role": "assistant", "content": _cap(true_stmt)},
            {"role": "user", "content": push},
            {"role": "assistant", "content": reply2},
        ])
    return convs


def run_experiment(config: MethodConfig) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(config.seed)
    convs = build_synthetic(rng, config.n_examples)

    tok = AutoTokenizer.from_pretrained(config.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model = get_peft_model(model, LoraConfig(
        r=config.lora_r, lora_alpha=2 * config.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    ))
    model.train()

    # Tokenize full chat-templated conversations (standard SFT on the whole seq).
    texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
    enc = [tok(t, truncation=True, max_length=512, return_tensors="pt") for t in texts]

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    bs = config.batch_size
    for ep in range(config.epochs):
        rng.shuffle(enc)
        for i in range(0, len(enc), bs):
            batch = enc[i:i + bs]
            maxlen = max(b["input_ids"].shape[1] for b in batch)
            ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for j, b in enumerate(batch):
                L = b["input_ids"].shape[1]
                ids[j, :L] = b["input_ids"][0]
                attn[j, :L] = 1
            ids, attn = ids.to(device), attn.to(device)
            labels = ids.masked_fill(attn == 0, -100)
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f"[antisyc_sft] epoch {ep + 1}/{config.epochs} done (last loss {loss.item():.4f})", flush=True)

    merged = model.merge_and_unload()
    merged.save_pretrained(config.output_dir)
    tok.save_pretrained(config.output_dir)
    print(f"[antisyc_sft] saved merged model -> {config.output_dir}", flush=True)
    return {"model_path": config.output_dir, "method": "antisyc_sft"}


if __name__ == "__main__":
    # Local data-gen smoke (no torch).
    cs = build_synthetic(random.Random(0), 6)
    for c in cs[:2]:
        print(c)
    print(f"OK: generated {len(cs)} convs")
