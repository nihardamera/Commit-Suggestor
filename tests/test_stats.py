"""The statistics behind every interval and agreement number in the report."""

from __future__ import annotations

import random

import pytest
from scipy import stats as scipy_stats

from commit_suggestor.stats import (
    cohen_kappa,
    confusion,
    kappa_from_confusion,
    kendall_tau_b,
    spearman,
    stratified_bootstrap,
    wilson,
)


class TestKappa:
    def test_a_two_by_two_matrix_worked_by_hand(self):
        # 50 items. Agree on 20 "A" and 15 "B": observed agreement 35/50 = 0.70.
        # Rater 1 says A 25 times, rater 2 says A 30 times, so chance agreement
        # is 0.5 * 0.6 + 0.5 * 0.4 = 0.50, and kappa = (0.70 - 0.50) / 0.50.
        assert kappa_from_confusion([[20, 5], [10, 15]]) == pytest.approx(0.4)

    def test_labels_become_the_same_matrix(self):
        a = ["A"] * 25 + ["B"] * 25
        b = ["A"] * 20 + ["B"] * 5 + ["A"] * 10 + ["B"] * 15
        assert confusion(a, b, ["A", "B"]) == [[20, 5], [10, 15]]
        assert cohen_kappa(a, b) == pytest.approx(0.4)

    def test_perfect_agreement_is_one_and_chance_level_is_zero(self):
        assert cohen_kappa(list("ABTAB"), list("ABTAB")) == 1.0
        assert cohen_kappa(list("AABB"), list("ABAB")) == pytest.approx(0.0)

    def test_three_classes_worked_by_hand(self):
        a = list("AABBTTABTA")
        b = list("ABBBTAABTT")
        # Agree on 7 of 10. Marginals a: A4 B3 T3, b: A3 B4 T3, so chance is
        # 0.4 * 0.3 + 0.3 * 0.4 + 0.3 * 0.3 = 0.33.
        assert cohen_kappa(a, b, "ABT") == pytest.approx((0.7 - 0.33) / (1 - 0.33))


class TestWilson:
    def test_matches_the_closed_form(self):
        interval = wilson(40, 50)
        assert interval.estimate == 0.8
        assert (interval.low, interval.high) == pytest.approx((0.6697, 0.8875), abs=1e-4)

    def test_stays_inside_zero_and_one(self):
        assert wilson(0, 10).low == 0.0
        assert wilson(10, 10).high == 1.0

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            wilson(11, 10)


class TestRankCorrelation:
    @pytest.fixture
    def scores(self):
        rng = random.Random(0)
        x = [rng.randint(1, 5) for _ in range(60)]  # many ties, like 1-5 ratings
        y = [v + rng.random() * 3 for v in x]
        return x, y

    def test_spearman_matches_scipy(self, scores):
        x, y = scores
        assert spearman(x, y) == pytest.approx(scipy_stats.spearmanr(x, y).statistic)

    def test_kendall_tau_b_matches_scipy(self, scores):
        x, y = scores
        assert kendall_tau_b(x, y) == pytest.approx(scipy_stats.kendalltau(x, y).statistic)


class TestStratifiedBootstrap:
    def test_point_estimates_are_the_pooled_and_per_stratum_means(self):
        values = [1.0, 1.0, 1.0, 0.0]
        strata = ["big", "big", "big", "small"]
        assert stratified_bootstrap(values, strata).estimate == pytest.approx(0.75)
        assert stratified_bootstrap(values, strata, macro=True).estimate == pytest.approx(0.5)

    def test_interval_brackets_the_estimate_and_is_reproducible(self):
        rng = random.Random(1)
        values = [rng.random() for _ in range(300)]
        strata = [i % 3 for i in range(300)]
        first = stratified_bootstrap(values, strata, seed=7)
        assert first.low < first.estimate < first.high
        assert stratified_bootstrap(values, strata, seed=7) == first

    def test_a_constant_stratum_adds_no_variance(self):
        interval = stratified_bootstrap([2.0] * 10, ["a"] * 10)
        assert interval.low == interval.high == 2.0

    def test_scaled_rounds_for_reporting(self):
        assert stratified_bootstrap([0.5, 0.5], ["a", "b"]).scaled(100) == [50.0, 50.0, 50.0]
