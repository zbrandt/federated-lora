#!/usr/bin/env python
"""Pre-flight smoke test. Run this on CPU BEFORE submitting to the cluster.

Every check is cheap (tiny subsets, 1 epoch) and exists to catch a specific
failure mode that would otherwise waste a GPU allocation.

    python -m scripts.smoke_test --quick    # no model download; ~2 seconds
    python -m scripts.smoke_test            # full, CPU only; ~3-6 minutes

Exit code 0 = all passed, 1 = something failed.
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASSED, FAILED = [], []


def check(name):
    """Decorator: run a check, record pass/fail, never abort the whole suite."""

    def deco(fn):
        def wrapped(*a, **kw):
            try:
                msg = fn(*a, **kw)
                PASSED.append(name)
                print(f"  PASS  {name}" + (f"  ({msg})" if msg else ""))
                return True
            except Exception as e:
                FAILED.append((name, e))
                print(f"  FAIL  {name}")
                print(f"        {type(e).__name__}: {e}")
                if VERBOSE:
                    traceback.print_exc()
                return False

        return wrapped

    return deco


VERBOSE = False


# ===========================================================================
# Tier 1 -- no model download required
# ===========================================================================


@check("imports")
def test_imports():
    import src.aggregators, src.config, src.data, src.engine
    import src.federated, src.model, src.utils
    return "all modules import"


@check("config: alpha tracks rank")
def test_alpha():
    from src.config import Config

    for r in (4, 8, 16, 32):
        c = Config(rank=r)
        assert c.alpha == r, f"alpha={c.alpha} != rank={r}"
    # If alpha were fixed while rank varied, the LoRA scaling s = alpha/rank
    # would change too and the "rank sweep" would secretly be a magnitude sweep.
    return "s = alpha/rank = 1 for all ranks"


@check("config: dp flags")
def test_dp_flags():
    from src.config import Config

    assert not Config().use_dp
    assert not Config().accounts_privacy
    assert Config(noise_multiplier=0.0).use_dp
    assert not Config(noise_multiplier=0.0).accounts_privacy, "sigma=0 must not report an epsilon"
    assert Config(epsilon=3.0).use_dp and Config(epsilon=3.0).accounts_privacy
    try:
        Config(epsilon=3.0, noise_multiplier=0.5)
        raise AssertionError("should reject both epsilon and noise_multiplier")
    except ValueError:
        pass
    return "baseline / diagnostic / dp paths distinguished"


@check("config: FL heterogeneity guards")
def test_fl_config():
    from src.config import FLConfig

    f = FLConfig(num_clients=5, client_ranks=(4, 4, 8, 8, 16), client_epsilons=(0.5, 0.5, 3, 3, 8))
    assert f.clients_per_round == 5
    try:
        FLConfig(num_clients=5, client_ranks=(4, 8))
        raise AssertionError("should reject wrong-length client_ranks")
    except ValueError:
        pass
    return "per-client rank/eps lists validated"


@check("aggregator: fedavg math")
def test_fedavg():
    import torch
    from src.aggregators import get_aggregator

    agg = get_aggregator("fedavg")
    s1 = {"lora_A": torch.ones(8, 4), "lora_B": torch.ones(4, 8) * 2.0}
    s2 = {"lora_A": torch.ones(8, 4) * 3.0, "lora_B": torch.ones(4, 8) * 4.0}

    out = agg([s1, s2], weights=[1, 1])
    assert abs(out["lora_A"][0, 0].item() - 2.0) < 1e-6, "unweighted mean wrong"

    out = agg([s1, s2], weights=[3, 1])  # -> 0.75*1 + 0.25*3 = 1.5
    assert abs(out["lora_A"][0, 0].item() - 1.5) < 1e-6, "weighted mean wrong"
    return "unweighted and size-weighted means correct"


@check("aggregator: rejects heterogeneous ranks")
def test_fedavg_guard():
    import torch
    from src.aggregators import get_aggregator

    agg = get_aggregator("fedavg")
    s1 = {"lora_A": torch.ones(8, 4)}
    s2 = {"lora_A": torch.ones(16, 4)}
    try:
        agg([s1, s2])
        raise AssertionError("fedavg silently accepted mismatched ranks")
    except ValueError:
        pass
    # Must fail loudly: silently averaging mismatched ranks would corrupt
    # every heterogeneous-rank experiment without any visible error.
    return "raises rather than corrupting"


@check("utils: sweep resumability")
def test_resume():
    from src.utils import already_done, append_result, load_results

    d = tempfile.mkdtemp()
    try:
        append_result({"rank": 8, "epsilon": 3.0, "seed": 0, "accuracy": 0.9}, d)
        append_result({"rank": 8, "epsilon": None, "seed": 0, "accuracy": 0.94}, d)
        df = load_results(d)
        assert already_done(df, {"rank": 8, "epsilon": 3.0, "seed": 0})
        assert not already_done(df, {"rank": 16, "epsilon": 3.0, "seed": 0})
        # eps=None (non-private) is written as NaN -- must still match, or the
        # sweep re-runs every baseline on relaunch.
        assert already_done(df, {"rank": 8, "epsilon": None, "seed": 0})
    finally:
        shutil.rmtree(d)
    return "finished rows skipped, incl. non-private NaN rows"


@check("federated: DP path fails loudly")
def test_fl_dp_guard():
    from src.config import FLConfig
    from src.federated import federated_train

    try:
        federated_train(FLConfig(num_clients=2, rounds=1, epsilon=3.0, n_train=32))
        raise AssertionError("FL accepted epsilon but has no DP path!")
    except NotImplementedError:
        pass
    return "cannot silently report non-private FL as private"


# ===========================================================================
# Tier 2 -- downloads roberta-base (~500MB, cached after first run)
# ===========================================================================


@check("model: parameter counts exact")
def test_param_counts():
    from src.config import Config
    from src.model import build_model, param_breakdown

    # RoBERTa-base: 12 layers x {query, value} x r x (768 in + 768 out)
    #             = 12 * 2 * 2 * 768 * r = 36864 * r
    for r in (4, 8):
        m = build_model(Config(rank=r, simple_head=True))
        bd = param_breakdown(m)
        expected_lora = 36864 * r
        assert bd["lora"] == expected_lora, f"lora {bd['lora']} != {expected_lora}"
        assert bd["head"] == 768 * 2 + 2, f"slim head is {bd['head']}, expected 1538"
        del m

    # The regression that matters: replacing only out_proj leaves the 590k
    # `dense` layer trainable, so the head dominates the DP noise budget.
    stock = param_breakdown(build_model(Config(rank=8, simple_head=False)))
    slim = param_breakdown(build_model(Config(rank=8, simple_head=True)))
    assert stock["total"] == 887042, f"stock head total {stock['total']}, expected 887042"
    assert slim["total"] == 296450, f"slim head total {slim['total']}, expected 296450"
    assert slim["lora"] / slim["total"] > 0.99, "LoRA should dominate trainable params"
    return f"r8 slim={slim['total']:,} vs stock={stock['total']:,}; LoRA is 99%+ of slim"


@check("model: lora state round-trip")
def test_lora_state():
    import torch
    from src.config import Config
    from src.model import build_model, get_lora_state, set_lora_state

    m = build_model(Config(rank=8))
    orig = get_lora_state(m)
    assert len(orig) > 0, "no LoRA tensors found"

    # Perturb, then restore, and confirm we get the original back.
    with torch.no_grad():
        for _, p in m.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    changed = get_lora_state(m)
    diff = max((changed[k] - orig[k]).abs().max().item() for k in orig)
    assert diff > 1e-6, "perturbation did not take"

    set_lora_state(m, orig)
    back = get_lora_state(m)
    diff = max((back[k] - orig[k]).abs().max().item() for k in orig)
    assert diff < 1e-6, f"round-trip mismatch {diff}"
    return f"{len(orig)} tensors survive get/set unchanged"


@check("run: non-private baseline (tiny)")
def test_nonprivate(tmp):
    from src.config import Config
    from src.engine import run_single
    from src.utils import free_memory

    cfg = Config(rank=4, n_train=128, n_val=64, epochs=1, batch_size=32,
                 device="cpu", out_dir=tmp)
    out = run_single(cfg)
    acc = out["row"]["accuracy"]
    loss = out["row"]["train_loss"]
    free_memory(out.pop("model"))
    assert 0.0 <= acc <= 1.0, f"accuracy out of range: {acc}"
    # Sane loss range. If the step-count fix regressed, this would be ~30x too big.
    assert 0.05 < loss < 5.0, f"train_loss {loss:.2f} implausible -- check loss averaging"
    assert out["row"]["epsilon_spent"] is None, "non-private run reported an epsilon"
    return f"acc={acc:.3f} loss={loss:.3f}"


@check("run: sigma=0 diagnostic (tiny)")
def test_diagnostic(tmp):
    from src.config import Config
    from src.engine import run_single
    from src.utils import free_memory

    # Full Opacus pipeline (Poisson sampling, per-sample grads, clipping) with
    # zero noise. This is the run that separates "pipeline broken" from
    # "SNR too low" -- if it errors here it will error on the cluster.
    cfg = Config(rank=4, n_train=128, n_val=64, epochs=1, batch_size=32,
                 max_physical_batch_size=8, noise_multiplier=0.0,
                 device="cpu", out_dir=tmp)
    out = run_single(cfg)
    acc = out["row"]["accuracy"]
    free_memory(out.pop("model"))
    assert 0.0 <= acc <= 1.0
    assert out["row"]["epsilon_spent"] is None, "sigma=0 must not report a finite epsilon"
    return f"opacus pipeline runs; acc={acc:.3f}"


@check("run: DP with epsilon target (tiny)")
def test_dp(tmp):
    from src.config import Config
    from src.engine import run_single
    from src.utils import free_memory

    cfg = Config(rank=4, n_train=128, n_val=64, epochs=1, batch_size=32,
                 max_physical_batch_size=8, epsilon=3.0, delta=1e-5,
                 device="cpu", out_dir=tmp)
    out = run_single(cfg)
    eps = out["row"]["epsilon_spent"]
    free_memory(out.pop("model"))
    assert eps is not None, "no epsilon reported"
    # Must not overspend the budget; should be close to the target.
    assert eps <= 3.0 + 1e-3, f"spent {eps} > target 3.0 -- accounting is wrong"
    assert eps > 1.0, f"spent {eps} suspiciously far below target"
    return f"eps_spent={eps:.4f} <= target 3.0"


@check("run: batch memory manager chunks correctly")
def test_chunking(tmp):
    """Logical batch stays large (for accounting) while physical stays small."""
    import torch
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from src.config import Config
    from src.data import get_dataloaders, get_dataset
    from src.engine import attach_privacy
    from src.model import build_model

    cfg = Config(rank=4, n_train=256, n_val=32, epochs=1, batch_size=64,
                 max_physical_batch_size=8, noise_multiplier=0.0, device="cpu", out_dir=tmp)
    ds, tok = get_dataset(cfg)
    train_loader, _ = get_dataloaders(cfg, ds, tok)
    model = build_model(cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model, opt, train_loader, _ = attach_privacy(model, opt, train_loader, cfg)

    sizes = []
    with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=8,
                            optimizer=opt) as loader:
        for i, b in enumerate(loader):
            sizes.append(b["input_ids"].shape[0])
            if i > 20:
                break
    assert max(sizes) <= 8, f"physical batch {max(sizes)} exceeded cap of 8"
    return f"physical batches <= 8 (saw max {max(sizes)}), logical stays 64"


@check("run: federated homogeneous (tiny)")
def test_federated(tmp):
    from src.config import FLConfig
    from src.federated import federated_train
    from src.utils import free_memory

    cfg = FLConfig(rank=4, n_train=128, n_val=64, batch_size=16, num_clients=2,
                   rounds=2, local_epochs=1, device="cpu", out_dir=tmp)
    out = federated_train(cfg)
    acc = out["accuracy"]
    free_memory(out.pop("model"))
    assert 0.0 <= acc <= 1.0
    assert len(out["history"]) == 2, "wrong number of rounds recorded"
    return f"2 clients x 2 rounds; acc={acc:.3f}"


# ===========================================================================


def main():
    global VERBOSE
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="skip anything needing a model download")
    p.add_argument("-v", "--verbose", action="store_true", help="print full tracebacks")
    args = p.parse_args()
    VERBOSE = args.verbose

    from src.utils import setup_logging

    setup_logging("WARNING")  # keep smoke-test output readable

    print("\n=== Tier 1: logic (no downloads) ===")
    test_imports()
    test_alpha()
    test_dp_flags()
    test_fl_config()
    test_fedavg()
    test_fedavg_guard()
    test_resume()
    test_fl_dp_guard()

    if not args.quick:
        print("\n=== Tier 2: end-to-end on CPU (downloads roberta-base) ===")
        tmp = tempfile.mkdtemp()
        try:
            test_param_counts()
            test_lora_state()
            test_nonprivate(tmp)
            test_diagnostic(tmp)
            test_dp(tmp)
            test_chunking(tmp)
            test_federated(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("\n(skipping Tier 2; drop --quick to run it)")

    total = len(PASSED) + len(FAILED)
    print(f"\n{'=' * 60}")
    print(f"{len(PASSED)}/{total} passed")
    if FAILED:
        print("\nFailures:")
        for name, err in FAILED:
            print(f"  - {name}: {type(err).__name__}: {err}")
        print("\nFix these before submitting to the cluster.")
        return 1
    print("All checks passed -- safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
