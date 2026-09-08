# -*- coding: utf-8 -*-
"""One-round federated smoke test on small synthetic data (not committed)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
COLS = [f"f{i}" for i in range(30)] + ["Class"]


def make_csv(path, n, fraud_frac):
    X = rng.normal(size=(n, 30))
    y = (rng.random(n) < fraud_frac).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
    df["Class"] = y
    df.to_csv(path, index=False)


make_csv("smoke_bank_a.csv", 300, 0.02)
make_csv("smoke_bank_b.csv", 200, 0.06)
make_csv("smoke_test.csv", 150, 0.04)

from fl import FedClient, FedServer  # noqa: E402

clients = [
    FedClient("bank_a", "smoke_bank_a.csv", batch_size=64),
    FedClient("bank_b", "smoke_bank_b.csv", batch_size=64),
]

for agg in ["contribution_aware", "fedavg", "loss_weighted", "accuracy_weighted"]:
    server = FedServer(clients, "smoke_test.csv", aggregation=agg,
                       proxy_size=64, batch_size=64)
    hist = server.fit(rounds=2, local_epochs=1, lr=0.001, verbose=False)
    m = hist[-1]["metrics"]
    print(f"[{agg}] OK  round2 ROC-AUC={m['roc_auc']:.4f} F1={m['f1']:.4f}")

print("SMOKE TEST PASSED")