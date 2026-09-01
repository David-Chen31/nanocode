"""Retry only what retrying can fix.

A quota exhaustion killed four concurrent runs in a real sweep. Retrying that
is not resilience, it is a slower way to fail -- and it burns the backoff
budget that a genuine rate limit would have needed.
"""
from __future__ import annotations

import pytest

from agent.llm import Fatal, Transient, classify, with_retries


class Err(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status_code = status


def test_an_exhausted_quota_is_not_retried():
    """The exact failure from a real sweep, verbatim."""
    exc = Err("Error code: 403 - {'error': {'message': 'user quota is not enough'}}", 403)
    assert isinstance(classify(exc), Fatal)


def test_a_rate_limit_is_retried():
    assert isinstance(classify(Err("rate limit exceeded", 429)), Transient)


def test_server_errors_are_retried():
    for code in (500, 502, 503, 504):
        assert isinstance(classify(Err("upstream boom", code)), Transient)


def test_a_plain_503_is_still_transient():
    """Only the relay's model-unavailable is fatal, not every 'unavailable'."""
    assert isinstance(classify(Err("Service Unavailable", 503)), Transient)


def test_a_model_the_relay_does_not_offer_is_fatal():
    assert isinstance(classify(Err("GROUP_MODEL_UNAVAILABLE: no channel", 503)), Fatal)


def test_a_bad_key_is_not_retried():
    assert isinstance(classify(Err("invalid_api_key", 401)), Fatal)


def test_a_connection_reset_is_retried():
    assert isinstance(classify(Err("Connection reset by peer")), Transient)


# -- the retry loop ---------------------------------------------------------

def test_it_succeeds_after_a_transient_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Err("rate limit", 429)
        return "ok"

    assert with_retries(flaky, attempts=4, base=0.001) == "ok"
    assert calls["n"] == 3


def test_a_fatal_error_is_raised_on_the_first_attempt():
    calls = {"n": 0}

    def dead():
        calls["n"] += 1
        raise Err("user quota is not enough", 403)

    with pytest.raises(Fatal):
        with_retries(dead, attempts=4, base=0.001)
    assert calls["n"] == 1, "a permanent failure was retried"


def test_retries_are_bounded():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise Err("boom", 500)

    with pytest.raises(Transient):
        with_retries(always, attempts=3, base=0.001)
    assert calls["n"] == 3


def test_each_retry_is_reported():
    seen = []

    def flaky():
        if len(seen) < 2:
            raise Err("overloaded", 503)
        return "ok"

    with_retries(flaky, attempts=4, base=0.001,
                 on_retry=lambda n, d, why: seen.append((n, why)))
    assert [n for n, _ in seen] == [1, 2]


def test_backoff_grows():
    delays = []
    def always():
        raise Err("boom", 500)
    with pytest.raises(Transient):
        with_retries(always, attempts=4, base=0.001,
                     on_retry=lambda n, d, why: delays.append(d))
    assert delays[-1] > delays[0], "the delay did not grow"
