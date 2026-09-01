"""Power warnings must not simulate impossible correctness effects."""
from experiments.power import (effect_grid, mde, required_paired_blocks,
                               simulate_paired_noninferiority, wilson_interval)


def test_effect_grid_stops_at_the_attainable_positive_effect():
    assert effect_grid(0.94) == [0.02, 0.04, 0.06]


def test_no_detectable_feasible_effect_is_not_reported_as_over_seventy_points():
    seen = []

    def never(base, eff, reps, n, rng, **kwargs):
        seen.append(eff)
        return False

    assert mde(0.94, 2, 8, never, trials=1) is None
    assert max(seen) == 0.06


def test_paired_noninferiority_size_uses_discordance_not_two_independent_arms():
    blocks = required_paired_blocks(6 / 36, margin=0.05)
    assert 520 <= blocks <= 525
    lo, hi = wilson_interval(6, 36)
    assert lo < 6 / 36 < hi
    assert required_paired_blocks(hi, margin=0.05) > blocks


def test_zero_discordance_always_passes_the_bootstrap_rule():
    assert simulate_paired_noninferiority(
        3, 3, 0.0, trials=10, boots=40, seed=1) == 1.0
