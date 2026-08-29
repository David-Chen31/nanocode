"""Conversation history and context management.

The loop appends every tool result in full, and tool results are where an agent's
context actually goes: one `read_file` on a thousand-line module is worth more
tokens than every message the agent has produced up to that point. Left alone,
a long task walks into the window limit and dies there.

Three rules, in the order they apply:

1.  Cap each tool output when it is captured. Nothing unbounded ever enters the
    history, so no single message can exceed the budget on its own.
2.  Drop superseded reads first. If the agent read a file and then read it
    again, the earlier copy is dead weight -- the later one is what it is
    working from.
3.  Then evict oldest-first, leaving a recent window untouched.

THE CONSTRAINT THAT SHAPES ALL OF THIS

A tool result cannot simply be removed. Both wire formats pair it with the
assistant message that requested it -- `tool_call_id` on OpenAI, `tool_use_id`
on Anthropic -- and an assistant turn whose result has vanished is a malformed
request, not a shorter one. So eviction rewrites a message's *content* and
leaves the message itself in place. The conversation stays structurally valid at
every size.

The stub that replaces the content says what was dropped and why, so the model
can decide to read the file again rather than hallucinate what was in it. That
is deliberate: a silent hole invites confabulation, a labelled one invites a
re-read.

WHICH RULE ACTUALLY CARRIES THE LOAD

Measured, not assumed. On real runs against a padded package, rule 1 does very
nearly all of it: four clipped reads land the history at 2337 tokens against a
2600 budget, and eviction never fires. That is the design working as intended
rather than a reason to delete rules 2 and 3 -- capping each output bounds the
per-message cost, but nothing bounds the *number* of messages, so a long run
still needs a way to reclaim. Eviction is the backstop, and it is tested
offline (tests/test_context.py) precisely because it is rare in practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Rough token estimate without pulling in a tokeniser. ASCII text runs about
# four characters per token; CJK runs closer to one. Counting non-ASCII as a
# whole token each overestimates slightly, which is the safe direction for a
# budget -- being early to compact costs a re-read, being late costs the run.
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ch < "\x80")
    return ascii_chars // 4 + (len(text) - ascii_chars) + 1


def _message_text(msg: dict[str, Any]) -> str:
    """Every string that will be billed, flattened out of either wire format."""
    out: list[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            for key in ("text", "content"):
                v = block.get(key)
                if isinstance(v, str):
                    out.append(v)
            if isinstance(block.get("input"), dict):
                out.append(str(block["input"]))
    for call in msg.get("tool_calls", []) or []:
        fn = call.get("function", {})
        out.append(str(fn.get("name", "")) + str(fn.get("arguments", "")))
    return "\n".join(out)


@dataclass
class ContextPolicy:
    """Everything tunable about how history is kept, in one place."""

    max_tokens: int = 48_000
    # Turns at the tail that are never evicted: the agent's working state. Too
    # small and it forgets what it just did; too large and there is nothing left
    # to reclaim.
    keep_recent: int = 8
    # Applied when a tool result is captured, before it ever reaches history.
    tool_output_chars: int = 6_000
    # Head-and-tail rather than head-only: the end of a traceback or a test
    # summary is usually the part that matters.
    head_fraction: float = 0.6

    def headroom(self) -> int:
        """Budget left once the protected tail is accounted for, worst case.

        The tail is never evicted, so a policy whose tail can exceed the budget
        cannot honour that budget however hard it compacts. Negative headroom is
        a misconfiguration, and it is better to see it here than as a 400 from
        the provider halfway through a run.
        """
        worst_tail = self.keep_recent * (self.tool_output_chars // 4)
        return self.max_tokens - worst_tail


def clip_tool_output(text: str, policy: ContextPolicy) -> tuple[str, bool]:
    """Cap one tool output. Returns the text and whether it was clipped."""
    limit = policy.tool_output_chars
    if len(text) <= limit:
        return text, False
    head = int(limit * policy.head_fraction)
    tail = limit - head
    dropped = len(text) - limit
    return (f"{text[:head]}\n\n... [{dropped} characters omitted from the middle "
            f"of this output] ...\n\n{text[-tail:]}", True)


@dataclass
class Conversation:
    """The message list, plus the bookkeeping that keeps it inside a budget."""

    policy: ContextPolicy = field(default_factory=ContextPolicy)
    backend: str = "openai"
    messages: list[dict[str, Any]] = field(default_factory=list)
    # tool_call_id -> (tool name, short description of the arguments), so an
    # eviction stub can say what it is replacing.
    _origin: dict[str, tuple[str, str]] = field(default_factory=dict)
    _evicted: set[int] = field(default_factory=set)
    n_compactions: int = 0
    n_evicted: int = 0
    n_clipped: int = 0
    over_budget: bool = False

    # -- building ---------------------------------------------------------
    def add(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)

    def note_call(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        detail = args.get("path") or args.get("command") or ""
        self._origin[call_id] = (name, str(detail)[:80])

    # -- accounting -------------------------------------------------------
    def token_estimate(self) -> int:
        return sum(estimate_tokens(_message_text(m)) for m in self.messages)

    def _is_tool_result(self, i: int) -> bool:
        m = self.messages[i]
        if m.get("role") == "tool":
            return True
        content = m.get("content")
        return (m.get("role") == "user" and isinstance(content, list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in content))

    def _result_ids(self, i: int) -> list[str]:
        m = self.messages[i]
        if m.get("role") == "tool":
            return [m.get("tool_call_id", "")]
        return [b.get("tool_use_id", "") for b in m.get("content", [])
                if isinstance(b, dict) and b.get("type") == "tool_result"]

    def _stub(self, call_id: str) -> str:
        name, detail = self._origin.get(call_id, ("a tool", ""))
        where = f" {detail}" if detail else ""
        return (f"[earlier output of {name}{where} was dropped to stay inside the "
                f"context budget. Run it again if you still need it.]")

    def _evict(self, i: int) -> int:
        """Replace one message's content with stubs. Returns tokens reclaimed."""
        before = estimate_tokens(_message_text(self.messages[i]))
        m = self.messages[i]
        if m.get("role") == "tool":
            m["content"] = self._stub(m.get("tool_call_id", ""))
        else:
            for block in m.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block["content"] = self._stub(block.get("tool_use_id", ""))
        self._evicted.add(i)
        self.n_evicted += 1
        return before - estimate_tokens(_message_text(m))

    # -- the policy -------------------------------------------------------
    def compact(self, on_event: Callable[[dict[str, Any]], None] | None = None) -> bool:
        """Try to bring the history under budget. True when anything was evicted.

        Best effort, not a guarantee: the first message and the recent window are
        protected, so if those alone exceed the budget nothing can be done here.
        When that happens it is reported rather than passed off as success --
        `reached_budget` in the event, and `over_budget` in stats.
        """
        total = self.token_estimate()
        if total <= self.policy.max_tokens:
            return False

        start_total = total
        # Index 0 is the task itself and is never touched; the tail is the
        # agent's working state.
        protected_tail = max(1, len(self.messages) - self.policy.keep_recent)
        candidates = [i for i in range(1, protected_tail)
                      if self._is_tool_result(i) and i not in self._evicted]

        # Pass one: reads the agent has already superseded by reading the same
        # path again. These cost nothing to lose.
        seen_later: dict[str, int] = {}
        for i in candidates:
            for cid in self._result_ids(i):
                name, detail = self._origin.get(cid, ("", ""))
                if name == "read_file" and detail:
                    seen_later.setdefault(detail, 0)
                    seen_later[detail] += 1
        superseded = [i for i in candidates
                      if any(self._origin.get(c, ("", ""))[0] == "read_file"
                             and seen_later.get(self._origin.get(c, ("", ""))[1], 0) > 1
                             for c in self._result_ids(i))]

        for group in (superseded, candidates):
            for i in group:
                if total <= self.policy.max_tokens:
                    break
                if i in self._evicted:
                    continue
                total -= self._evict(i)

        self.n_compactions += 1
        self.over_budget = total > self.policy.max_tokens
        if on_event:
            on_event({"reclaimed": start_total - total, "tokens_before": start_total,
                      "tokens_after": total, "evicted": self.n_evicted,
                      "reached_budget": not self.over_budget})
        return True

    def render(self) -> list[dict[str, Any]]:
        return self.messages

    def stats(self) -> dict[str, Any]:
        return {"messages": len(self.messages), "tokens": self.token_estimate(),
                "compactions": self.n_compactions, "evicted": self.n_evicted,
                "clipped_outputs": self.n_clipped, "over_budget": self.over_budget}
