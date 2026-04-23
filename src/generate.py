"""
Greedy translation generation for SCP before/after probe measurement.

Given a list of records and a model, produce:
    (sample_id, prompt, template_index, hypothesis)

Batched, length-sorted for throughput, with per-batch context-limit guard.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from .prompt_utils import clean_hypothesis, render_for_sample


def _context_limit(model, tokenizer) -> int | None:
    m = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(m, int) and m > 0:
        return int(m)
    t = getattr(tokenizer, "model_max_length", None)
    if isinstance(t, int) and 0 < t < 1_000_000:
        return int(t)
    return None


def _build_gen_kwargs(tokenizer, gen_cfg) -> dict[str, Any]:
    return {
        "max_new_tokens": int(gen_cfg.max_new_tokens),
        "do_sample": bool(gen_cfg.do_sample),
        "num_beams": int(gen_cfg.num_beams),
        "temperature": float(gen_cfg.temperature),
        "top_p": float(gen_cfg.top_p),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }


def generate_translations(
    *,
    model,
    tokenizer,
    records: list[dict[str, Any]],
    prompt_templates: list[str],
    response_prefix: str,
    template_seed: int,
    fixed_template_index: int | None,
    gen_cfg,
) -> list[dict[str, Any]]:
    """Return a list of dicts ordered to match `records`."""
    if not records:
        return []

    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    rendered = []
    for pos, rec in enumerate(records):
        prefix, idx = render_for_sample(
            sample_key=str(rec["sample_id"]),
            source_text=rec["source_text"],
            templates=prompt_templates,
            response_prefix=response_prefix,
            template_seed=template_seed,
            src_iso=rec["source_lang_iso"],
            tgt_iso=rec["target_lang_iso"],
            fixed_template_index=fixed_template_index,
        )
        rendered.append(
            {
                "_pos": pos,
                "sample_id": rec["sample_id"],
                "source_text": rec["source_text"],
                "prompt_prefix": prefix,
                "template_index": int(idx),
                "reference_text": rec.get("reference_text", ""),
                "metadata": rec.get("metadata", {}),
            }
        )

    if bool(getattr(gen_cfg, "sort_by_input_length", True)):
        rendered.sort(key=lambda x: (len(x["source_text"]), x["_pos"]))

    batch_size = max(1, int(gen_cfg.batch_size))
    limit = _context_limit(model, base_tokenizer)
    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []
    total = len(rendered)
    print(f"[generate] n={total} batch_size={batch_size} context_limit={limit}")

    for start in range(0, total, batch_size):
        chunk = rendered[start : start + batch_size]
        try:
            enc = base_tokenizer(
                [x["prompt_prefix"] for x in chunk],
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=False,
                pad_to_multiple_of=8,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            batch_input_len = int(input_ids.shape[1])
            kwargs = _build_gen_kwargs(base_tokenizer, gen_cfg)
            if limit is not None:
                allowed = int((limit - attn.sum(dim=1)).min().item())
                if allowed <= 0:
                    raise RuntimeError(f"input exceeds context limit: limit={limit}")
                kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"], allowed)
            with torch.inference_mode():
                out = model.generate(input_ids=input_ids, attention_mask=attn, **kwargs)
            for i, row in enumerate(chunk):
                decoded = base_tokenizer.decode(
                    out[i][batch_input_len:], skip_special_tokens=True
                )
                hyp = clean_hypothesis(decoded, response_prefix)
                results.append({**row, "hypothesis": hyp, "error": ""})
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            for row in chunk:
                results.append({**row, "hypothesis": "", "error": err})

        done = min(start + len(chunk), total)
        if done % (batch_size * 5) == 0 or done == total:
            print(f"  progress {done}/{total}")

    results.sort(key=lambda r: r["_pos"])
    for r in results:
        r.pop("_pos", None)
    return results
