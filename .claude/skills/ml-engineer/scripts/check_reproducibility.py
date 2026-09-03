#!/usr/bin/env python3
"""Verify a training setup is reproducible before spending hours on a run.

Two runs with the same seed must produce identical weights. Discovering
otherwise after an eighteen-fold training sweep is expensive; discovering it
here costs seconds.

Checks the seeding of Python's `random`, numpy, and torch, then proves it by
constructing and stepping a small model twice.

Usage:
    python check_reproducibility.py               # run all checks
    python check_reproducibility.py --quick       # skip the training step test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def set_all_seeds(seed: int) -> None:
    """Seed every source of randomness in the training path.

    Missing any one of these makes a run irreproducible in a way that is hard
    to attribute later: the weights differ, and nothing says why.
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def check_python_random(seed: int) -> dict:
    random.seed(seed)
    a = [random.random() for _ in range(5)]
    random.seed(seed)
    b = [random.random() for _ in range(5)]
    return {"check": "python_random", "passed": a == b,
            "note": "" if a == b else "random.seed does not reproduce — unexpected"}


def check_numpy_random(seed: int) -> dict:
    np.random.seed(seed)
    a = np.random.rand(5)
    np.random.seed(seed)
    b = np.random.rand(5)
    legacy_ok = bool(np.array_equal(a, b))

    # The Generator API is preferred: it does not share global state, so one
    # library reseeding cannot silently change another's stream.
    g1 = np.random.default_rng(seed)
    g2 = np.random.default_rng(seed)
    gen_ok = bool(np.array_equal(g1.random(5), g2.random(5)))

    return {"check": "numpy_random", "passed": legacy_ok and gen_ok,
            "legacy_api": legacy_ok, "generator_api": gen_ok,
            "note": "prefer np.random.default_rng(seed) over the global np.random.seed"}


def check_torch_seeding(seed: int) -> dict:
    if not TORCH_AVAILABLE:
        return {"check": "torch_seeding", "passed": True, "note": "torch not installed — skipped"}

    torch.manual_seed(seed)
    a = torch.randn(5)
    torch.manual_seed(seed)
    b = torch.randn(5)
    ok = bool(torch.equal(a, b))

    return {"check": "torch_seeding", "passed": ok,
            "cuda_available": torch.cuda.is_available(),
            "note": "" if ok else "torch.manual_seed does not reproduce"}


def check_model_init(seed: int) -> dict:
    """Two models built under the same seed must have identical weights."""
    if not TORCH_AVAILABLE:
        return {"check": "model_init", "passed": True, "note": "torch not installed — skipped"}

    def build():
        set_all_seeds(seed)
        return torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    m1, m2 = build(), build()
    identical = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values())
    )

    return {"check": "model_init", "passed": identical,
            "note": "" if identical else
                    "identical seeds produced different initial weights — "
                    "seeding is incomplete"}


def check_training_step(seed: int) -> dict:
    """A few optimizer steps must land on identical weights."""
    if not TORCH_AVAILABLE:
        return {"check": "training_step", "passed": True, "note": "torch not installed — skipped"}

    def run():
        set_all_seeds(seed)
        model = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        g = torch.Generator().manual_seed(seed)
        x = torch.randn(32, 10, generator=g)
        y = torch.randint(0, 2, (32, 1), generator=g).float()

        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(round(float(loss.item()), 8))
        return losses, model.state_dict()

    l1, s1 = run()
    l2, s2 = run()

    losses_match = l1 == l2
    weights_match = all(torch.equal(a, b) for a, b in zip(s1.values(), s2.values()))

    return {"check": "training_step", "passed": losses_match and weights_match,
            "losses_run_1": l1, "losses_match": losses_match, "weights_match": weights_match,
            "note": "" if losses_match and weights_match else
                    "identical seeds produced different training results — "
                    "something in the step path is unseeded"}


def check_dataloader_seeding(seed: int) -> dict:
    """DataLoader shuffling must be seeded through a generator.

    Relying on global torch state works until another library reseeds it, at
    which point batch order silently changes between runs.
    """
    if not TORCH_AVAILABLE:
        return {"check": "dataloader_seeding", "passed": True, "note": "torch not installed — skipped"}

    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.arange(100).float().unsqueeze(1))

    def order(use_generator: bool):
        if use_generator:
            g = torch.Generator().manual_seed(seed)
            dl = DataLoader(ds, batch_size=10, shuffle=True, generator=g)
        else:
            set_all_seeds(seed)
            dl = DataLoader(ds, batch_size=10, shuffle=True)
        return [int(b[0][0].item()) for b in dl]

    gen_ok = order(True) == order(True)
    global_ok = order(False) == order(False)

    return {"check": "dataloader_seeding", "passed": gen_ok,
            "generator_reproducible": gen_ok, "global_seed_reproducible": global_ok,
            "note": "pass an explicit torch.Generator to DataLoader rather than "
                    "relying on global seed state"}


def check_cuda_determinism() -> dict:
    """cuDNN picks nondeterministic algorithms unless told not to."""
    if not TORCH_AVAILABLE:
        return {"check": "cuda_determinism", "passed": True, "note": "torch not installed — skipped"}
    if not torch.cuda.is_available():
        return {"check": "cuda_determinism", "passed": True,
                "note": "no CUDA device — CPU training is deterministic by default"}

    deterministic = bool(torch.backends.cudnn.deterministic)
    benchmark = bool(torch.backends.cudnn.benchmark)
    ok = deterministic and not benchmark

    return {"check": "cuda_determinism", "passed": ok,
            "cudnn_deterministic": deterministic, "cudnn_benchmark": benchmark,
            "note": "" if ok else
                    "set torch.backends.cudnn.deterministic = True and "
                    "torch.backends.cudnn.benchmark = False before training"}


def check_config_records_seed(config_path: Path | None) -> dict:
    """The seed must be recorded, or a reproducible run cannot be repeated."""
    if config_path is None:
        return {"check": "config_records_seed", "passed": True, "note": "no config supplied — skipped"}
    if not config_path.exists():
        return {"check": "config_records_seed", "passed": False,
                "note": f"config not found: {config_path}"}

    try:
        cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        return {"check": "config_records_seed", "passed": False, "note": f"malformed config: {exc}"}

    def find_seed(obj) -> bool:
        if isinstance(obj, dict):
            return any("seed" in str(k).lower() or find_seed(v) for k, v in obj.items())
        return False

    has_seed = find_seed(cfg)
    return {"check": "config_records_seed", "passed": has_seed,
            "note": "" if has_seed else
                    "config records no seed — the run cannot be reproduced from it"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, help="training config JSON to inspect")
    parser.add_argument("--quick", action="store_true", help="skip the training step test")
    args = parser.parse_args()

    checks = [
        check_python_random(args.seed),
        check_numpy_random(args.seed),
        check_torch_seeding(args.seed),
        check_model_init(args.seed),
        check_dataloader_seeding(args.seed),
        check_cuda_determinism(),
        check_config_records_seed(args.config),
    ]
    if not args.quick:
        checks.append(check_training_step(args.seed))

    print("=" * 74)
    print(f"REPRODUCIBILITY CHECK — seed {args.seed}")
    print("=" * 74)
    if not TORCH_AVAILABLE:
        print("NOTE: torch is not installed; torch-dependent checks were skipped.")
    print()

    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
        for k, v in c.items():
            if k in ("check", "passed", "note"):
                continue
            print(f"       {k}: {v}")
        if c.get("note"):
            print(f"       -> {c['note']}")
        print()

    passed = all(c["passed"] for c in checks)
    print("=" * 74)
    print("RESULT:", "setup is reproducible" if passed else "REPRODUCIBILITY NOT GUARANTEED")
    print("=" * 74)

    if not passed:
        print()
        print("A run that cannot be reproduced cannot be defended. Fix the failures")
        print("above before starting a training sweep — see")
        print("references/reproducibility_checklist.md")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
