"""Provider-agnostic LLM backends.

Three backends share one interface so that every experiment can be run either
against a live model or fully offline against recorded fixtures. Offline replay
is not a toy path: it is what makes the research results reproducible without an
API key, and it is the default.

Backend spec strings:
    anthropic:claude-sonnet-5          -> Anthropic Messages API
    openai:gpt-4o-mini                 -> OpenAI-compatible (honours OPENAI_BASE_URL)
    fixture:bench/fixtures/agent.json  -> deterministic replay from disk
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Prices are USD per 1M tokens. Cost accounting is a first-class reporting
# requirement for every experiment in this repo, not an afterthought.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "_default": (1.0, 5.0),
}


class Transient(Exception):
    """An upstream failure that is worth trying again."""


class Fatal(Exception):
    """An upstream failure that retrying cannot fix."""


# Retrying a dead key is not resilience, it is delay. The distinction below is
# the whole point of classifying rather than blanket-retrying: a rate limit
# clears on its own, an exhausted quota or a bad key never does, and today a
# quota 403 killed four concurrent runs that each spent their retries first.
_FATAL_MARKERS = ("quota", "insufficient_quota", "billing", "invalid_api_key",
                  "authentication", "permission", "model_not_found",
                  "does not exist", "model_unavailable")
# Note the marker is `model_unavailable`, not a bare `unavailable`: a plain 503
# "service unavailable" is exactly the kind of thing worth retrying, while this
# relay's GROUP_MODEL_UNAVAILABLE means the model is not offered at all.


def classify(exc: Exception) -> Exception:
    """Sort an upstream exception into Transient or Fatal."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).lower()

    # Checked before the status code: a 403 or a 429 can carry either meaning,
    # and the body is what says which.
    if any(m in text for m in _FATAL_MARKERS):
        return Fatal(str(exc)[:300])
    if status in (429, 500, 502, 503, 504, 408, 409):
        return Transient(str(exc)[:300])
    if status in (400, 401, 403, 404, 422):
        return Fatal(str(exc)[:300])
    # Connection resets and read timeouts arrive with no status at all.
    if any(m in text for m in ("timeout", "timed out", "connection", "reset",
                               "temporarily", "overloaded", "rate limit")):
        return Transient(str(exc)[:300])
    return Fatal(str(exc)[:300])


def with_retries(call, *, attempts: int = 4, base: float = 1.5,
                 on_retry=None):
    """Run `call`, retrying only what is worth retrying.

    Exponential backoff with jitter. The jitter matters because the sweeps run
    several agents at once: without it, a shared rate limit synchronises every
    worker's retry into the same instant and they collide again.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except (Transient, Fatal):
            raise
        except Exception as exc:                      # noqa: BLE001
            kind = classify(exc)
            if isinstance(kind, Fatal):
                raise kind from exc
            last = kind
            if attempt == attempts - 1:
                break
            delay = base * (2 ** attempt) * (0.5 + random.random())
            if on_retry:
                on_retry(attempt + 1, delay, str(kind)[:160])
            time.sleep(delay)
    raise last or Transient("exhausted retries")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def cost_usd(self, model: str) -> float:
        pin, pout = PRICES.get(model, PRICES["_default"])
        return (self.input_tokens * pin + self.output_tokens * pout) / 1_000_000


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model's arguments could not be parsed. The call still exists
    # -- see _parse_arguments for why it must.
    parse_error: str = ""


# Sentinel key: an unparseable argument blob is kept under this so the loop can
# hand the model back its own broken JSON instead of a generic complaint.
RAW_ARGS = "__raw__"


def _parse_arguments(blob: str | None) -> dict[str, Any]:
    """Decode a tool call's arguments, tolerating what models actually emit.

    Two failures are common enough to be routine rather than exceptional. The
    first is truncation: a `write_file` whose `content` runs into the output
    token limit produces JSON that simply stops. The second is a model emitting
    a bare string or list where an object belongs.

    Neither may raise. A raised exception here aborts the whole run from inside
    the backend, which is the wrong place and the wrong severity -- the model
    made a recoverable mistake and should be told about it on the next turn.
    So the call is always constructed, and the problem travels on it.
    """
    if not blob:
        return {}
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return {RAW_ARGS: blob}
    if not isinstance(parsed, dict):
        return {RAW_ARGS: blob}
    return parsed


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # Mean per-token logprob of the completion when the provider exposes it.
    # This powers the `token_entropy` baseline; None means "not available".
    mean_logprob: float | None = None
    raw: Any = None


class LLMBackend:
    """Interface every backend implements."""

    name: str = "abstract"
    model: str = "unknown"

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
        fixture_key: str | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def text(self, prompt: str, *, system: str | None = None, **kw: Any) -> LLMResponse:
        return self.complete([{"role": "user", "content": prompt}], system=system, **kw)


class AnthropicBackend(LLMBackend):
    name = "anthropic"
    on_retry = None

    def __init__(self, model: str) -> None:
        import anthropic  # lazy: offline runs need no SDK

        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, messages, *, system=None, tools=None, temperature=0.0,
                 max_tokens=4096, seed=None, fixture_key=None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]

        resp = with_retries(lambda: self._client.messages.create(**kwargs),
                            on_retry=self.on_retry)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens, 1),
            raw=resp,
        )


class OpenAICompatBackend(LLMBackend):
    name = "openai"
    # Set by the caller to record retries in the trace; a retry that nothing
    # reports is indistinguishable from a slow call.
    on_retry = None

    def __init__(self, model: str, *, timeout: float | None = None,
                 max_retries: int = 4) -> None:
        from openai import OpenAI

        self.model = model
        self.max_retries = max_retries
        # A hung upstream should fail the one call, not the whole sweep. Relays
        # in particular will happily hold a socket open forever.
        self._client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            timeout=timeout if timeout is not None
            else float(os.environ.get("NANOCODE_TIMEOUT", "90")),
            # The SDK's own retries are turned off so that retrying is done
            # here, where a permanent failure can be told apart from a
            # temporary one and where each attempt is visible.
            max_retries=0,
        )

    def complete(self, messages, *, system=None, tools=None, temperature=0.0,
                 max_tokens=4096, seed=None, fixture_key=None) -> LLMResponse:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": _to_openai_tool(t)} for t in tools]
        if seed is not None:
            kwargs["seed"] = seed

        resp = with_retries(lambda: self._client.chat.completions.create(**kwargs),
                            attempts=self.max_retries, on_retry=self.on_retry)
        choice = resp.choices[0]

        calls = []
        for tc in choice.message.tool_calls or []:
            calls.append(ToolCall(id=tc.id, name=tc.function.name,
                                  arguments=_parse_arguments(tc.function.arguments)))

        mean_lp = None
        lp = getattr(choice, "logprobs", None)
        if lp and getattr(lp, "content", None):
            vals = [t.logprob for t in lp.content if t.logprob is not None]
            if vals:
                mean_lp = sum(vals) / len(vals)

        u = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=calls,
            usage=Usage(u.prompt_tokens, u.completion_tokens, 1) if u else Usage(calls=1),
            mean_logprob=mean_lp,
            raw=resp,
        )


class FixtureBackend(LLMBackend):
    """Deterministic replay.

    Fixtures map a stable key to a list of canned responses. The key is either an
    explicit `fixture_key` passed through `complete`, or a hash of the messages.
    Repeated calls with the same key walk the list, which is exactly the shape
    needed for "sample k candidates at temperature > 0".
    """

    name = "fixture"

    def __init__(self, path: str | Path, model: str = "fixture") -> None:
        self.model = model
        self.path = Path(path)
        self._data: dict[str, list[dict[str, Any]]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        self._cursor: dict[str, int] = {}

    @staticmethod
    def key_for(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def complete(self, messages, *, system=None, tools=None, temperature=0.0,
                 max_tokens=4096, seed=None, fixture_key: str | None = None) -> LLMResponse:
        # A fixture file may declare "__sequence__": responses are then replayed in
        # order regardless of the messages. That is the practical way to script a
        # multi-turn agent run, where every turn has a different prompt hash.
        if "__sequence__" in self._data:
            key = "__sequence__"
        else:
            key = fixture_key or self.key_for(
                json.dumps(messages, sort_keys=True, ensure_ascii=False))
        bucket = self._data.get(key)
        if not bucket:
            raise KeyError(
                "no fixture for key " + repr(key) + ". Record one, or run with a live "
                "backend, e.g. NANOCODE_BACKEND=anthropic:claude-sonnet-5"
            )
        i = self._cursor.get(key, 0)
        if key == "__sequence__" and i >= len(bucket):
            raise IndexError("fixture sequence exhausted after " + str(len(bucket)) + " turns")
        item = bucket[i % len(bucket)]
        self._cursor[key] = i + 1
        return LLMResponse(
            text=item.get("text", ""),
            tool_calls=[ToolCall(**c) for c in item.get("tool_calls", [])],
            usage=Usage(item.get("input_tokens", 0), item.get("output_tokens", 0), 1),
            mean_logprob=item.get("mean_logprob"),
            raw=item,
        )


def _to_anthropic_tool(t: dict[str, Any]) -> dict[str, Any]:
    return {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}


def _to_openai_tool(t: dict[str, Any]) -> dict[str, Any]:
    return {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}


def make_backend(spec: str | None = None) -> LLMBackend:
    """Build a backend from a spec string, defaulting to $NANOCODE_BACKEND."""
    spec = spec or os.environ.get("NANOCODE_BACKEND", "fixture:bench/fixtures/agent.json")
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        return AnthropicBackend(rest or "claude-sonnet-5")
    if kind in {"openai", "openai-compat"}:
        return OpenAICompatBackend(rest or "gpt-4o-mini")
    if kind == "fixture":
        return FixtureBackend(rest or "bench/fixtures/agent.json")
    raise ValueError("unknown backend spec: " + repr(spec))
