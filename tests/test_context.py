"""Context management: the budget must hold and the pairing must survive.

The second of those is the one that bites. Removing a tool result to save
tokens leaves the assistant turn that requested it dangling, and the provider
rejects the whole request -- so the failure shows up as a 400 on the *next*
call, far from the code that caused it. These tests pin the invariant down.
"""
from __future__ import annotations

from agent.context import (ContextPolicy, Conversation, clip_tool_output,
                           estimate_tokens)


def _openai_exchange(convo: Conversation, call_id: str, path: str, payload: str) -> None:
    convo.add({"role": "assistant", "content": None,
               "tool_calls": [{"id": call_id, "type": "function",
                               "function": {"name": "read_file",
                                            "arguments": f'{{"path": "{path}"}}'}}]})
    convo.note_call(call_id, "read_file", {"path": path})
    convo.add({"role": "tool", "tool_call_id": call_id, "content": payload})


def test_estimate_counts_cjk_more_heavily_than_ascii():
    # Same character count, but CJK costs closer to a token each.
    assert estimate_tokens("这是一段中文文本" * 8) > estimate_tokens("ascii text here" * 8)


def test_clip_keeps_head_and_tail():
    policy = ContextPolicy(tool_output_chars=200)
    text = "HEAD" + "x" * 5_000 + "TAIL"
    out, clipped = clip_tool_output(text, policy)
    assert clipped
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")          # the end of a traceback is the useful part
    assert len(out) < len(text)


def test_clip_leaves_short_output_alone():
    out, clipped = clip_tool_output("short", ContextPolicy(tool_output_chars=200))
    assert (out, clipped) == ("short", False)


def test_compaction_preserves_every_tool_call_id():
    """The invariant: an evicted result is rewritten, never removed."""
    convo = Conversation(policy=ContextPolicy(max_tokens=300, keep_recent=2))
    convo.add({"role": "user", "content": "the task"})
    for i in range(10):
        _openai_exchange(convo, f"call{i}", f"mod{i}.py", "payload " * 400)

    requested = {c["id"] for m in convo.messages for c in m.get("tool_calls", []) or []}
    convo.compact()
    answered = {m["tool_call_id"] for m in convo.messages if m.get("role") == "tool"}
    assert requested == answered, "an assistant tool_call lost its result"


def test_compaction_protects_the_task_and_the_recent_window():
    convo = Conversation(policy=ContextPolicy(max_tokens=200, keep_recent=4))
    convo.add({"role": "user", "content": "the task"})
    for i in range(8):
        _openai_exchange(convo, f"call{i}", f"mod{i}.py", "payload " * 400)
    convo.compact()

    assert convo.messages[0]["content"] == "the task"
    tail = convo.messages[-4:]
    for m in tail:
        if m.get("role") == "tool":
            assert "dropped to stay inside" not in m["content"], \
                "the working state was evicted"


def test_superseded_reads_go_first():
    """Reading a file twice makes the earlier copy dead weight."""
    convo = Conversation(policy=ContextPolicy(max_tokens=400, keep_recent=2))
    convo.add({"role": "user", "content": "the task"})
    _openai_exchange(convo, "a", "same.py", "OLD " * 300)
    _openai_exchange(convo, "b", "other.py", "OTHER " * 300)
    _openai_exchange(convo, "c", "same.py", "NEW " * 300)
    _openai_exchange(convo, "d", "tail.py", "TAIL " * 300)
    convo.compact()

    by_id = {m["tool_call_id"]: m["content"] for m in convo.messages
             if m.get("role") == "tool"}
    assert "dropped" in by_id["a"], "the superseded read survived"


def test_stub_names_the_tool_so_the_model_can_recover():
    convo = Conversation(policy=ContextPolicy(max_tokens=50, keep_recent=1))
    convo.add({"role": "user", "content": "t"})
    _openai_exchange(convo, "x", "pkg/helpers.py", "body " * 500)
    _openai_exchange(convo, "y", "pkg/core.py", "body " * 500)
    convo.compact()
    stub = next(m["content"] for m in convo.messages
                if m.get("role") == "tool" and m["tool_call_id"] == "x")
    assert "read_file" in stub and "pkg/helpers.py" in stub
    assert "again" in stub          # tells the model it can recover the content


def test_anthropic_shape_is_compacted_too():
    convo = Conversation(policy=ContextPolicy(max_tokens=120, keep_recent=1),
                         backend="anthropic")
    convo.add({"role": "user", "content": "the task"})
    for i in range(6):
        convo.add({"role": "assistant",
                   "content": [{"type": "tool_use", "id": f"u{i}",
                                "name": "read_file", "input": {"path": f"m{i}.py"}}]})
        convo.note_call(f"u{i}", "read_file", {"path": f"m{i}.py"})
        convo.add({"role": "user",
                   "content": [{"type": "tool_result", "tool_use_id": f"u{i}",
                                "content": "payload " * 300}]})
    convo.compact()

    used = {b["id"] for m in convo.messages for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "tool_use"}
    answered = {b["tool_use_id"] for m in convo.messages
                for b in (m.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "tool_result"}
    assert used == answered


def test_compact_is_a_noop_under_budget():
    convo = Conversation(policy=ContextPolicy(max_tokens=100_000))
    convo.add({"role": "user", "content": "the task"})
    _openai_exchange(convo, "a", "m.py", "small")
    before = [dict(m) for m in convo.messages]
    assert convo.compact() is False
    assert convo.messages == before
    assert convo.n_compactions == 0


def test_budget_is_met_when_the_policy_leaves_headroom():
    policy = ContextPolicy(max_tokens=4_000, keep_recent=4, tool_output_chars=2_000)
    assert policy.headroom() > 0
    convo = Conversation(policy=policy)
    convo.add({"role": "user", "content": "the task"})
    for i in range(12):
        _openai_exchange(convo, f"c{i}", f"m{i}.py", "payload " * 200)
    convo.compact()
    assert convo.token_estimate() <= policy.max_tokens
    assert convo.over_budget is False


def test_an_unreachable_budget_is_reported_not_hidden():
    """The tail is protected, so a tiny budget cannot be honoured. Say so."""
    policy = ContextPolicy(max_tokens=50, keep_recent=4, tool_output_chars=6_000)
    assert policy.headroom() < 0          # the misconfiguration is visible up front
    convo = Conversation(policy=policy)
    convo.add({"role": "user", "content": "the task"})
    for i in range(6):
        _openai_exchange(convo, f"c{i}", f"m{i}.py", "payload " * 300)

    seen: list[dict] = []
    convo.compact(seen.append)
    assert convo.token_estimate() > policy.max_tokens   # genuinely could not
    assert convo.over_budget is True
    assert seen[0]["reached_budget"] is False
    assert convo.stats()["over_budget"] is True
