"""Teacher API wrapper.

Pipeline per call:
  1. build cache_key
  2. cache.get -> if hit, log teacher_edit event from cache, return
  3. provider call (with retries)
  4. cache.put + ledger.log(teacher_edit)
  5. optional Weave trace (failure must not raise)

Providers are pluggable; the OpenAI/Anthropic clients are imported lazily.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .cache import TeacherCache, make_cache_key
from .config import TeacherCfg
from .ledger import Ledger
from .prompts import hash_prompt, load_prompt
from .schemas import TEACHER_ACTIONS


@dataclass
class TeacherResult:
    teacher_output: str
    teacher_action: str | None
    error_tags: list[str]
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: int
    cached: bool
    raw_response: Any | None = None


class TeacherClient:
    def __init__(
        self,
        cfg: TeacherCfg,
        cache: TeacherCache,
        ledger: Ledger,
        provider_call: Callable[[dict], dict] | None = None,
        weave_tracer: Callable[[dict, dict], None] | None = None,
    ):
        self.cfg = cfg
        self.cache = cache
        self.ledger = ledger
        self.weave_tracer = weave_tracer
        self.prompt_text = load_prompt(cfg.prompt_path)
        self.prompt_hash = hash_prompt(self.prompt_text)
        self._provider_call = provider_call or _default_provider_call(cfg)

    def _decoding(self) -> dict:
        return {
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

    def _build_request(self, source: str, student_output: str, probe_output: str | None) -> dict:
        user_msg = self.prompt_text.format(
            source=source,
            student=student_output,
            probe=probe_output or "",
        )
        msgs = []
        if self.cfg.system_prompt:
            msgs.append({"role": "system", "content": self.cfg.system_prompt})
        msgs.append({"role": "user", "content": user_msg})
        return {
            "provider": self.cfg.provider,
            "model": self.cfg.model,
            "messages": msgs,
            **self._decoding(),
        }

    def edit(
        self,
        *,
        round_id: int,
        example_id: str,
        source_text: str,
        student_output: str,
        probe_output: str | None,
        extra_event: dict | None = None,
    ) -> TeacherResult:
        cache_key = make_cache_key(
            source_text=source_text,
            student_output=student_output,
            probe_output=probe_output,
            teacher_model=self.cfg.model,
            prompt_hash=self.prompt_hash,
            decoding=self._decoding(),
        )
        cached = self.cache.get(cache_key)
        if cached:
            res = TeacherResult(
                teacher_output=cached.get("teacher_output", ""),
                teacher_action=cached.get("teacher_action"),
                error_tags=[],
                input_tokens=cached.get("input_tokens"),
                output_tokens=cached.get("output_tokens"),
                cost_usd=cached.get("cost_usd"),
                latency_ms=cached.get("latency_ms") or 0,
                cached=True,
                raw_response=cached.get("response_json"),
            )
            self._log_event(round_id, example_id, source_text, student_output,
                            probe_output, res, cache_key, extra_event)
            return res

        request = self._build_request(source_text, student_output, probe_output)
        last_err: Exception | None = None
        t0 = time.time()
        response: dict | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                response = self._provider_call(request)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        latency_ms = int((time.time() - t0) * 1000)
        if last_err is not None or response is None:
            raise RuntimeError(f"teacher call failed: {last_err}")

        teacher_output = response.get("output", "")
        teacher_action = response.get("teacher_action")
        if teacher_action is not None and teacher_action not in TEACHER_ACTIONS:
            teacher_action = None
        res = TeacherResult(
            teacher_output=teacher_output,
            teacher_action=teacher_action,
            error_tags=response.get("error_tags", []) or [],
            input_tokens=response.get("input_tokens"),
            output_tokens=response.get("output_tokens"),
            cost_usd=response.get("cost_usd"),
            latency_ms=latency_ms,
            cached=False,
            raw_response=response.get("raw"),
        )
        self.cache.put(cache_key, {
            "provider": self.cfg.provider,
            "model": self.cfg.model,
            "prompt_version": self.cfg.prompt_version,
            "prompt_hash": self.prompt_hash,
            "request_json": request,
            "response_json": response,
            "teacher_output": res.teacher_output,
            "teacher_action": res.teacher_action,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cost_usd": res.cost_usd,
            "latency_ms": res.latency_ms,
        })
        self._log_event(round_id, example_id, source_text, student_output,
                        probe_output, res, cache_key, extra_event)
        if self.weave_tracer is not None:
            try:
                self.weave_tracer(request, {"output": res.teacher_output, "raw": res.raw_response})
            except Exception:
                pass
        return res

    def _log_event(self, round_id, example_id, source, student, probe, res,
                   cache_key, extra_event):
        ev = {
            "event_type": "teacher_edit",
            "round_id": round_id,
            "example_id": example_id,
            "source_text": source,
            "student_output": student,
            "probe_output": probe,
            "teacher_output": res.teacher_output,
            "teacher_model": self.cfg.model,
            "provider": self.cfg.provider,
            "prompt_version": self.cfg.prompt_version,
            "prompt_hash": self.prompt_hash,
            "teacher_action": res.teacher_action,
            "error_tags": res.error_tags,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cost_usd": res.cost_usd,
            "latency_ms": res.latency_ms,
            "cache_key": cache_key,
            "cached": res.cached,
        }
        if extra_event:
            ev.update(extra_event)
        self.ledger.log(ev)


def _default_provider_call(cfg: TeacherCfg) -> Callable[[dict], dict]:
    """Lazy import provider SDK; raise on use if not configured."""
    def _call(request: dict) -> dict:
        provider = request["provider"]
        if provider == "openai":
            return _openai_call(request)
        if provider == "anthropic":
            return _anthropic_call(request)
        if provider == "echo":  # for testing
            last = request["messages"][-1]["content"]
            return {"output": last, "teacher_action": "no_change",
                    "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    "raw": {"echo": True}}
        raise ValueError(f"unknown provider: {provider}")
    return _call


def _openai_call(request: dict) -> dict:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=request["model"],
        messages=request["messages"],
        temperature=request.get("temperature", 0.0),
        max_tokens=request.get("max_tokens", 1024),
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return {
        "output": text,
        "teacher_action": _infer_action(text),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "cost_usd": None,
        "raw": json.loads(resp.model_dump_json()),
    }


def _anthropic_call(request: dict) -> dict:
    from anthropic import Anthropic
    client = Anthropic()
    sys_msg = next((m["content"] for m in request["messages"] if m["role"] == "system"), None)
    user_msgs = [m for m in request["messages"] if m["role"] != "system"]
    resp = client.messages.create(
        model=request["model"],
        system=sys_msg or "",
        messages=user_msgs,
        max_tokens=request.get("max_tokens", 1024),
        temperature=request.get("temperature", 0.0),
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return {
        "output": text,
        "teacher_action": _infer_action(text),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cost_usd": None,
        "raw": resp.model_dump() if hasattr(resp, "model_dump") else None,
    }


def _infer_action(text: str) -> str | None:
    """Best-effort: prompt should ask the model to emit an action tag.
    Returns None when not detectable; downstream classification can backfill.
    """
    head = text.strip().splitlines()[0].lower() if text.strip() else ""
    for a in TEACHER_ACTIONS:
        if a in head:
            return a
    return None
