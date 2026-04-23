"""
Probe LoRA training loop (Stage A of Algorithm 1).

Implements response-only-loss SFT on (source, self-translation) pairs. The loss
mask zeroes out prompt and response-prefix tokens, matching stage3's
`completion_only_loss=true` setting.

Intentionally lightweight: no checkpointing, no eval, no early stopping. The
probe is meant to collapse the model aggressively on a small pseudo-label set,
then be discarded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .prompt_utils import render_for_sample


@dataclass
class ProbeSample:
    input_ids: list[int]
    completion_mask: list[int]


class _InMemorySFTDataset(Dataset):
    def __init__(self, samples: list[ProbeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ProbeSample:
        return self.samples[idx]


def _collate(samples: list[ProbeSample], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(s.input_ids) for s in samples)
    input_ids, attn, labels = [], [], []
    for s in samples:
        pad_n = max_len - len(s.input_ids)
        input_ids.append(s.input_ids + [pad_id] * pad_n)
        attn.append([1] * len(s.input_ids) + [0] * pad_n)
        masked = [
            (tok if m == 1 else -100)
            for tok, m in zip(s.input_ids, s.completion_mask)
        ]
        labels.append(masked + [-100] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_probe_samples(
    *,
    tokenizer,
    records: list[dict[str, Any]],
    self_translations: list[str],
    prompt_templates: list[str],
    response_prefix: str,
    template_seed: int,
    fixed_template_index: int | None,
    max_length: int,
    add_eos_to_target: bool = True,
    drop_overlength: bool = True,
) -> list[ProbeSample]:
    if len(records) != len(self_translations):
        raise ValueError(
            f"records ({len(records)}) and self_translations ({len(self_translations)}) must align"
        )

    base_tok = getattr(tokenizer, "tokenizer", tokenizer)
    eos_id = base_tok.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id")

    samples: list[ProbeSample] = []
    skipped_empty = 0
    skipped_overlength = 0
    for rec, target in zip(records, self_translations):
        target = (target or "").strip()
        if not target:
            skipped_empty += 1
            continue
        prefix, _ = render_for_sample(
            sample_key=str(rec["sample_id"]),
            source_text=rec["source_text"],
            templates=prompt_templates,
            response_prefix=response_prefix,
            template_seed=template_seed,
            src_iso=rec["source_lang_iso"],
            tgt_iso=rec["target_lang_iso"],
            fixed_template_index=fixed_template_index,
        )
        prefix_ids = base_tok.encode(prefix, add_special_tokens=False)
        target_ids = base_tok.encode(target, add_special_tokens=False)
        input_ids = list(prefix_ids) + list(target_ids)
        completion_mask = [0] * len(prefix_ids) + [1] * len(target_ids)
        if add_eos_to_target and (not input_ids or input_ids[-1] != eos_id):
            input_ids.append(eos_id)
            completion_mask.append(1)
        if len(input_ids) > max_length:
            if drop_overlength:
                skipped_overlength += 1
                continue
            input_ids = input_ids[:max_length]
            completion_mask = completion_mask[:max_length]
        if 1 not in completion_mask:
            skipped_empty += 1
            continue
        samples.append(ProbeSample(input_ids=input_ids, completion_mask=completion_mask))
    print(
        f"[probe.build_samples] kept={len(samples)} "
        f"skipped_empty={skipped_empty} skipped_overlength={skipped_overlength}"
    )
    return samples


def train_probe(
    *,
    model,
    tokenizer,
    samples: list[ProbeSample],
    learning_rate: float,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    weight_decay: float = 0.0,
    logging_every: int = 5,
) -> dict[str, Any]:
    """Run an aggressive response-only-loss SFT pass. Returns a small metrics dict."""
    if not samples:
        return {"skipped": True, "reason": "no samples"}

    base_tok = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = base_tok.pad_token_id if base_tok.pad_token_id is not None else base_tok.eos_token_id
    ds = _InMemorySFTDataset(samples)
    loader = DataLoader(
        ds,
        batch_size=int(per_device_train_batch_size),
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, pad_id=pad_id),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters on probe model. Did you attach the LoRA adapter?")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    steps_per_epoch = max(1, math.ceil(len(loader) / max(1, int(gradient_accumulation_steps))))
    total_steps = int(num_train_epochs) * steps_per_epoch
    warmup_steps = max(0, int(float(warmup_ratio) * total_steps))

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        name = str(lr_scheduler_type).lower()
        if name == "constant":
            return 1.0
        if name == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "linear":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    device = next(model.parameters()).device
    losses: list[float] = []
    global_step = 0
    model.train()
    optimizer.zero_grad()
    accum = 0
    for epoch in range(int(num_train_epochs)):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / max(1, int(gradient_accumulation_steps))
            loss.backward()
            accum += 1
            if accum >= int(gradient_accumulation_steps):
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, float(max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum = 0
                global_step += 1
                losses.append(float(out.loss.detach().item()))
                if global_step % max(1, int(logging_every)) == 0:
                    print(
                        f"  probe step={global_step}/{total_steps} "
                        f"epoch={epoch + 1}/{num_train_epochs} "
                        f"loss={losses[-1]:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )
    if accum > 0:
        optimizer.step()
        optimizer.zero_grad()
    return {
        "skipped": False,
        "global_steps": global_step,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
    }
