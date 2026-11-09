import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from fl.aggregation import coordinate_median, krum, trimmed_mean
from fl.data_heg import make_schema_shards
from fl.robustness import transform_update
from fl.scoring import ClientScorer
from fl.heterogeneous import HeterogeneousClient, HeterogeneousFedServer
from partition.scenarios import make_scenario_partitions
from models.heterogeneous import HeterogeneousFraudModel
from experiments.prepare_fraud_dataset import prepare_dataset


class Phase2ComponentTests(unittest.TestCase):
    def setUp(self):
        self.updates = [
            {"client_id": str(i), "state_dict": {"weight": torch.tensor([float(v)])},
             "n_samples": 1, "metrics": {"val_loss": 1.0, "accuracy": 0.5}}
            for i, v in enumerate([0, 1, 2, 3, 100])
        ]

    def test_robust_aggregators_reduce_outlier_influence(self):
        median, median_influence = coordinate_median(self.updates)
        trimmed, trimmed_influence = trimmed_mean(self.updates, trim_ratio=0.2)
        selected, weights = krum(self.updates, byzantine_clients=1)
        self.assertEqual(median["weight"].item(), 2.0)
        self.assertEqual(trimmed["weight"].item(), 2.0)
        self.assertAlmostEqual(median_influence.sum(), 1.0)
        self.assertAlmostEqual(trimmed_influence.sum(), 1.0)
        self.assertEqual(median_influence[-1], 0.0)
        self.assertEqual(trimmed_influence[-1], 0.0)
        self.assertNotEqual(selected["weight"].item(), 100.0)
        self.assertEqual(weights.sum(), 1.0)

    def test_attack_transforms_only_update_delta(self):
        global_state = {"weight": torch.tensor([2.0])}
        update = {"state_dict": {"weight": torch.tensor([3.0])}, "metrics": {}}
        flipped = transform_update(update, global_state, "sign_flip")
        self.assertEqual(flipped["state_dict"]["weight"].item(), 1.0)
        scaled = transform_update(update, global_state, "scale", scale=2.0)
        self.assertEqual(scaled["state_dict"]["weight"].item(), 4.0)
        noise_a = transform_update(update, global_state, "noise", scale=1.0, seed=9)
        noise_b = transform_update(update, global_state, "noise", scale=1.0, seed=9)
        self.assertTrue(torch.equal(noise_a["state_dict"]["weight"],
                                    noise_b["state_dict"]["weight"]))
        self.assertEqual(update["state_dict"]["weight"].item(), 3.0)

    def test_heterogeneous_model_shares_torso_only(self):
        model = HeterogeneousFraudModel(input_dim=7)
        self.assertEqual(tuple(model(torch.randn(3, 7)).shape), (3, 1))
        shared = model.shared_state_dict()
        self.assertTrue(shared)
        self.assertTrue(all(key.startswith("torso.") for key in shared))
        clone = HeterogeneousFraudModel(input_dim=11)
        clone.load_shared_state_dict(shared)
        with self.assertRaises(ValueError):
            clone.load_shared_state_dict({"encoder.network.0.weight": torch.zeros(64, 11)})

    def test_feature_keyed_encoder_initialization_is_stable(self):
        left = HeterogeneousFraudModel(
            input_dim=2, feature_names=["Time", "V1"], model_seed=17)
        right = HeterogeneousFraudModel(
            input_dim=3, feature_names=["V1", "Amount", "Time"], model_seed=17)
        self.assertTrue(torch.equal(
            left.encoder.network[0].weight[:, 1],
            right.encoder.network[0].weight[:, 0]))
        self.assertTrue(torch.equal(
            left.torso.network[0].weight, right.torso.network[0].weight))

    def test_preprocessing_supports_categorical_dataset_and_schema_shards(self):
        rng = np.random.default_rng(7)
        labels = np.asarray([0] * 80 + [1] * 20)
        frame = pd.DataFrame({
            "transaction_id": np.arange(len(labels)),
            "amount": rng.normal(size=len(labels)),
            "merchant_category": np.resize(
                np.asarray(["food", "travel", "retail"]), len(labels)),
            "is_fraud": labels,
        })
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "raw.csv"
            frame.to_csv(source, index=False)
            output = root / "processed"
            metadata = prepare_dataset(
                source, output, "is_fraud", ["transaction_id"], seed=8)
            self.assertEqual(metadata["fraud_counts"]["train"], 14)
            train = pd.read_csv(output / "train.csv")
            self.assertIn("Class", train.columns)
            self.assertNotIn("transaction_id", train.columns)
            self.assertTrue(any(name.startswith("merchant_category_")
                                for name in train.columns))
            shards = make_schema_shards(
                output / "train.csv", root / "shards", seed=9, num_clients=5)
            self.assertEqual(len(shards), 5)
            self.assertEqual(sum(value["rows"] for value in shards.values()),
                             len(train))

    def test_heterogeneous_attack_and_robust_aggregation(self):
        rng = np.random.default_rng(4)
        n = 120
        labels = np.asarray([0, 1] * (n // 2), dtype=int)
        rng.shuffle(labels)
        frame = pd.DataFrame({
            f"feature_{i}": rng.normal(loc=labels * (0.2 + i * 0.1), size=n)
            for i in range(7)
        })
        frame["Class"] = labels
        with TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for index in range(5):
                path = root / f"bank_{index}.csv"
                frame[[f"feature_{index}", f"feature_{(index + 1) % 7}",
                       "Class"]].to_csv(path, index=False)
                paths.append(path)
            for aggregation in ("fedavg", "median", "trimmed_mean", "krum"):
                clients = [
                    HeterogeneousClient(
                        f"bank_{index}", path, batch_size=32, seed=13)
                    for index, path in enumerate(paths)
                ]
                server = HeterogeneousFedServer(
                    clients, aggregation=aggregation, attack="sign_flip",
                    attack_client=0, seed=13)
                history = server.fit(rounds=1, local_epochs=1)
                self.assertEqual(len(history), 1)
                self.assertTrue(np.isfinite(server.evaluate(frame)["pr_auc"]))

    def test_heterogeneous_contribution_scoring_aggregates_shared_torso(self):
        rng = np.random.default_rng(32)
        n = 120
        labels = np.asarray([0, 1] * (n // 2), dtype=int)
        rng.shuffle(labels)
        frame = pd.DataFrame({
            "Time": rng.normal(size=n),
            "V1": rng.normal(loc=labels * 0.8, size=n),
            "V2": rng.normal(loc=labels * 0.4, size=n),
            "Class": labels,
        })
        with TemporaryDirectory() as folder:
            root = Path(folder)
            left_path, right_path = root / "left.csv", root / "right.csv"
            frame[["Time", "V1", "Class"]].to_csv(left_path, index=False)
            frame[["Time", "V2", "Class"]].to_csv(right_path, index=False)
            clients = [
                HeterogeneousClient("left", left_path, batch_size=16, seed=8),
                HeterogeneousClient("right", right_path, batch_size=16, seed=8),
            ]
            scorer = ClientScorer(seed=8, bootstrap_samples=4)
            scorer.adaptive_pairs = True
            server = HeterogeneousFedServer(
                clients, aggregation="adaptive_pairs",
                scorer=scorer,
                reference_frame=frame)
            server.fit(rounds=1, local_epochs=1)
            record = server.history[0]
            self.assertEqual(record["aggregation"], "adaptive_pairs")
            self.assertAlmostEqual(sum(record["weights"]), 1.0)
            self.assertEqual(len(record["weights"]), 2)
            self.assertIn("complementarity_lcb", record["scores"])
            np.testing.assert_allclose(
                np.asarray(record["scores"]["local_pair_weight"])
                + np.asarray(record["scores"]["utility_pair_weight"]),
                np.ones(2))
            self.assertTrue(np.all(
                np.asarray(record["scores"]["utility_pair_weight"]) >= 0.2))
            self.assertTrue(np.all(
                np.asarray(record["scores"]["utility_pair_weight"]) <= 0.8))
            self.assertTrue(all(key.startswith("torso.")
                                for key in server.shared_state))
            self.assertTrue(np.isfinite(server.evaluate(frame)["pr_auc"]))

    def test_reliability_penalizes_extreme_scale_not_directional_difference(self):
        scorer = ClientScorer(bootstrap_samples=0)
        global_state = {"weight": torch.zeros(2)}
        updates = [
            {"client_id": str(i), "state_dict": {"weight": value},
             "n_samples": 10, "metrics": {"pr_auc": 0.5, "val_fraud_count": 30}}
            for i, value in enumerate((torch.tensor([0.1, 0.0]),
                                       torch.tensor([0.0, 0.1]),
                                       torch.tensor([-0.1, 0.0]),
                                       torch.tensor([100.0, 0.0])))
        ]
        trust = scorer._trust(updates, global_state)
        self.assertLess(trust[-1], 0.01)
        self.assertTrue(np.all(trust[:3] > 0.99))

    def test_quality_shrinks_unreliable_tiny_fraud_sample(self):
        scorer = ClientScorer(bootstrap_samples=0)
        updates = [
            {"metrics": {"pr_auc": 1.0, "val_fraud_count": 1}},
            {"metrics": {"pr_auc": 0.7, "val_fraud_count": 40}},
            {"metrics": {"pr_auc": 0.6, "val_fraud_count": 40}},
        ]
        quality = scorer._quality(updates)
        self.assertLess(quality[0], 0.7)

    def test_paired_bootstrap_preserves_both_classes(self):
        scorer = ClientScorer(bootstrap_samples=20, seed=5)
        targets = np.array([1, 1, 0, 0, 0, 0])
        full = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        reduced = np.array([0.8, 0.7, 0.6, 0.7, 0.4, 0.3])
        point, lower = scorer._paired_lcb(
            targets, full, reduced, np.random.default_rng(2))
        self.assertTrue(np.isfinite(point))
        self.assertTrue(np.isfinite(lower))
        self.assertNotAlmostEqual(point, lower)

    def test_stress_scenarios_are_reproducible_and_conserve_rows(self):
        rng = np.random.default_rng(12)
        n = 600
        labels = (rng.random(n) < 0.15).astype(int)
        frame = pd.DataFrame({
            "Time": np.arange(n, dtype=float),
            "V1": rng.normal(loc=labels * 2.0, size=n),
            "V2": rng.normal(size=n),
            "Class": labels,
        })
        for scenario in ("label_skew", "quantity_skew", "feature_skew", "temporal"):
            kwargs = {"feature_clusters": 4} if scenario == "feature_skew" else {}
            shards = make_scenario_partitions(
                frame, scenario=scenario, num_clients=3, seed=7, **kwargs)
            again = make_scenario_partitions(
                frame, scenario=scenario, num_clients=3, seed=7, **kwargs)
            combined = pd.concat(shards, ignore_index=True)
            self.assertEqual(len(combined), len(frame))
            self.assertTrue(all(len(shard) > 0 for shard in shards))
            self.assertEqual(set(combined["Time"]), set(frame["Time"]))
            self.assertEqual(int(combined["Class"].sum()), int(frame["Class"].sum()))
            for left, right in zip(shards, again):
                self.assertTrue(left.equals(right))

        temporal = make_scenario_partitions(frame, "temporal", num_clients=3)
        self.assertLess(temporal[0]["Time"].max(), temporal[1]["Time"].min())

    def test_noisy_label_scenario_flips_only_configured_client(self):
        rng = np.random.default_rng(14)
        n = 600
        frame = pd.DataFrame({
            "Time": np.arange(n, dtype=float), "V1": rng.normal(size=n),
            "Class": (rng.random(n) < 0.15).astype(int),
        })
        shards = make_scenario_partitions(
            frame, "noisy_label", num_clients=3, noise_rate=0.3,
            noise_client=1, seed=9)
        merged = pd.concat(shards, ignore_index=True)
        expected = frame.set_index("Time")["Class"]
        changed = merged.set_index("Time")["Class"] != expected.loc[merged["Time"]].to_numpy()
        changed_by_time = pd.Series(changed.to_numpy(), index=merged["Time"])
        self.assertEqual(int(changed_by_time.sum()), round(len(shards[1]) * 0.3))
        self.assertEqual(set(merged["Time"]), set(frame["Time"]))


if __name__ == "__main__":
    unittest.main()
