"""The judge's plumbing: both orders, caching, cost guards and the held-out lock.

No test calls the API: the backend is a mock or a stub that fails if called.
"""

from __future__ import annotations

import pytest

from commit_suggestor import judge as judge_module
from commit_suggestor.human import Pair
from commit_suggestor.judge import (
    Cache,
    Judge,
    Reply,
    Request,
    calibration_report,
    combine,
    estimate_usd,
    mock_backend,
    user_prompt,
)

DIFF = "diff --git a/x.py b/x.py\n+A: not the candidate line\n-B: nor this one"


def shorter(_prompt: str, a: str, b: str) -> str:
    return "A" if len(a) < len(b) else "B" if len(b) < len(a) else "tie"


class TestBothOrders:
    def test_a_consistent_preference_wins(self):
        assert combine("A", "B").winner == "x"  # x first, then x second
        assert combine("B", "A").winner == "y"

    def test_choosing_the_same_slot_twice_is_position_bias_and_a_tie(self):
        verdict = combine("A", "A")
        assert (verdict.winner, verdict.consistent) == ("tie", False)

    def test_two_ties_are_a_consistent_tie(self):
        assert combine("tie", "tie") == judge_module.Verdict("tie", "tie", "tie", True)

    def test_an_error_is_never_a_win(self):
        assert combine("error", "B").winner == "tie"
        assert not combine("error", "B").consistent

    def test_the_judge_maps_slots_back_to_subjects(self):
        j = Judge(mock_backend(shorter))
        assert j.compare(DIFF, "Add retries", "Add retries to the HTTP client").winner == "x"
        assert j.compare(DIFF, "Add retries to the HTTP client", "Add retries").winner == "y"

    def test_the_mock_reads_the_candidates_not_lines_of_the_diff(self):
        seen = []
        Judge(mock_backend(lambda _p, a, b: seen.append((a, b)) or "tie")).compare(
            DIFF, "first", "second"
        )
        assert seen == [("first", "second"), ("second", "first")]


class TestCache:
    def test_a_cached_request_is_never_sent_again(self, tmp_path):
        calls = []

        def backend(req: Request) -> Reply:
            calls.append(req)
            return Reply("A", "because", 1000, 100, "end_turn")

        path = tmp_path / "cache.jsonl"
        Judge(backend, cache=Cache(path)).compare(DIFF, "a", "b")
        Judge(backend, cache=Cache(path)).compare(DIFF, "a", "b")
        assert len(calls) == 2  # two orders, once each

    def test_spending_is_summed_from_real_usage(self, tmp_path):
        cache = Cache(tmp_path / "cache.jsonl")
        cache.put(
            Request("claude-haiku-5-5", "low", "s", "u"),
            Reply("A", "r", 1_000_000, 1_000_000, "end_turn"),
        )
        assert cache.spent_usd() == pytest.approx(0.10 + 0.50)
        assert Cache(tmp_path / "cache.jsonl").spent_usd() == pytest.approx(0.60)

    def test_a_memory_cache_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Judge(mock_backend(shorter)).compare(DIFF, "a", "bb")
        assert list(tmp_path.iterdir()) == []


def test_the_estimate_counts_every_uncached_call():
    reqs = [Request("claude-haiku-5-5", "low", "x" * 300, user_prompt(DIFF, "a", "b"))] * 10
    cost = estimate_usd(reqs)
    assert cost["calls"] == 10
    assert 0 < cost["usd"] < 0.01


def test_perfect_agreement_gives_kappa_one():
    pairs = [
        Pair("v2", i, "r", "d", "ref", "ft", "base", label, "dev")
        for i, label in enumerate(["finetuned", "base", "tie", "finetuned"])
    ]
    verdicts = [
        combine(*answers) for answers in [("A", "B"), ("B", "A"), ("tie", "tie"), ("A", "B")]
    ]
    report = calibration_report(pairs, verdicts)
    assert report["kappa_3class"] == 1.0
    assert report["position_consistent_pct"] == 100.0


class TestCommandLine:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(judge_module, "JUDGE_DIR", tmp_path)
        monkeypatch.setattr(judge_module, "CACHE", tmp_path / "cache.jsonl")
        monkeypatch.setattr(judge_module, "HELDOUT_LOCK", tmp_path / "heldout.json")

        def no_api() -> judge_module.Backend:
            raise AssertionError("the API must not be called")

        monkeypatch.setattr(judge_module, "anthropic_backend", no_api)
        return tmp_path

    def test_a_dry_run_sends_nothing_and_writes_nothing(self, isolated, capsys):
        judge_module.main(["calibrate", "--split", "dev"])
        assert "dry run" in capsys.readouterr().out
        assert list(isolated.iterdir()) == []

    def test_a_run_over_the_budget_is_refused(self, monkeypatch):
        monkeypatch.setattr(judge_module, "BUDGET_USD", 0.001)
        with pytest.raises(SystemExit, match="project budget"):
            judge_module.main(["calibrate", "--split", "dev", "--yes"])

    def test_a_run_over_max_usd_is_refused(self):
        with pytest.raises(SystemExit, match="max-usd"):
            judge_module.main(["calibrate", "--split", "dev", "--yes", "--max-usd", "0.0001"])

    def test_held_out_needs_final_and_runs_once(self, isolated):
        with pytest.raises(SystemExit, match="--final"):
            judge_module.main(["calibrate", "--split", "heldout"])
        (isolated / "heldout.json").write_text("{}")
        with pytest.raises(SystemExit, match="already scored"):
            judge_module.main(["calibrate", "--split", "heldout", "--final"])
