import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.plot_benchmark_results import (
    load_runs,
    mean_ci,
    plot_attack_rankings,
    plot_paired_deltas,
    ranked_operations,
)


class BenchmarkPlotTests(unittest.TestCase):
    def test_mean_ci_reports_seed_interval(self):
        mean, margin = mean_ci([0.6, 0.7, 0.8])
        self.assertAlmostEqual(mean, 0.7)
        self.assertGreater(margin, 0.0)

    def test_ranked_operations_sorts_within_track_and_condition(self):
        grouped = {
            ("creditcard", "standard", "none", "fedavg"): [
                {"seed": 42, "pr_auc": 0.6},
                {"seed": 43, "pr_auc": 0.8},
            ],
            ("creditcard", "standard", "none", "fedprox"): [
                {"seed": 42, "pr_auc": 0.71},
                {"seed": 43, "pr_auc": 0.71},
            ],
        }
        ranked = ranked_operations(
            grouped, "creditcard", "standard", "none", "pr_auc")
        self.assertEqual([row["method"] for row in ranked],
                         ["fedprox", "fedavg"])
        self.assertAlmostEqual(ranked[0]["mean"], 0.71)

    def test_plotter_rejects_incomplete_matrix_by_default(self):
        with TemporaryDirectory() as folder:
            results_dir = Path(folder)
            (results_dir / "manifest.json").write_text(
                json.dumps({"config": {"attacks": ["none", "sign_flip"]},
                            "jobs": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "jobs are incomplete"):
                load_runs(results_dir, allow_partial=False)

    def test_partial_mode_omits_incomplete_jobs(self):
        with TemporaryDirectory() as folder:
            results_dir = Path(folder)
            (results_dir / "manifest.json").write_text(
                json.dumps({"config": {"attacks": ["none", "sign_flip"]},
                            "jobs": {}}),
                encoding="utf-8",
            )
            grouped, partial, incomplete, attacks = load_runs(
                results_dir, allow_partial=True)
            self.assertEqual(grouped, {})
            self.assertTrue(partial)
            self.assertEqual(len(incomplete), 8)
            self.assertEqual(attacks, ["none", "sign_flip"])

    def test_attack_rankings_write_metric_files_per_attack(self):
        grouped = {
            ("creditcard", "standard", "none", "fedavg"): [
                {"seed": 42, "pr_auc": 0.6, "roc_auc": 0.8},
                {"seed": 43, "pr_auc": 0.7, "roc_auc": 0.9},
            ],
            ("creditcard", "heterogeneous", "none",
             "heterogeneous_encoder_fedavg"): [
                {"seed": 42, "pr_auc": 0.55, "roc_auc": 0.75},
                {"seed": 43, "pr_auc": 0.65, "roc_auc": 0.85},
            ],
        }
        with TemporaryDirectory() as folder:
            paths = plot_attack_rankings(
                grouped, Path(folder), partial=False, attacks=["none"])
            self.assertEqual(
                {path.name for path in paths},
                {
                    "ranked_none_pr_auc.pdf", "ranked_none_pr_auc.png",
                    "ranked_none_roc_auc.pdf", "ranked_none_roc_auc.png",
                },
            )

    def test_paired_plots_skip_unmatched_partial_results(self):
        grouped = {
            ("creditcard", "standard", "none", "contribution_aware"): [
                {"seed": 42, "pr_auc": 0.6, "roc_auc": 0.8},
            ],
        }
        with TemporaryDirectory() as folder:
            paths = plot_paired_deltas(
                grouped, Path(folder), partial=True, attacks=["none"])
            self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
