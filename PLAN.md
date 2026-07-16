# Federated LoRA: Structured Comparison — Experiment Design

Design doc for structured comparison of and small reproduction of 1–2 federated 
LoRA methods. First pass: FedIT baseline plus FFA-LoRA only. The harness should 
extend the existing `train_lora.py` / `inference.py` pipeline.

## Goal

Turn the centralized LoRA pipeline into a federated harness with a pluggable 
aggregation strategy, then compare two methods on the same harness: naive 
FedAvg-of-LoRA (FedIT-style baseline) and FFA-LoRA. Deliverables include a 
comparison table that can mention related methods, plus reproduction-quality 
plots for the two implemented methods.

## The core problem the methods address

With LoRA, each module's update is ΔW = (α/r)·BA. If K clients train locally and 
the server averages A and B separately (the naive approach, used by FedIT):

    ΔW_naive = (α/r) · ( (1/K)Σ B_i ) · ( (1/K)Σ A_i )
    ΔW_exact = (α/r) · (1/K) Σ (B_i A_i)

These differ by cross-client interference terms, "aggregation noise," that grows 
with data heterogeneity and local steps. A second, orthogonal problem is rank 
heterogeneity. Naive averaging requires every client to use the same r, but 
device capabilities differ. Each method resolves one or both:

| Method | Aggregation bias | Heterogeneous ranks | Mechanism | Comm. cost/round/module | Status in this project |
|---|---|---|---|---|---|
| FedIT (Zhang et al., arXiv:2305.05644) | Present (baseline) | No | FedAvg A and B separately | r(d_in+d_out) | **Implement (baseline)** |
| FFA-LoRA (Sun et al., ICLR 2024, arXiv:2403.12313) | **Eliminated**: A frozen at shared init, so avg(B_i)·A = avg(B_i·A) exactly | No | Freeze A, train/communicate only B; halves communication; noted more stable under DP noise | r·d_out (half of FedIT) | **Implement (repro 1)** |
| HetLoRA (Cho et al., EMNLP 2024, arXiv:2401.06432) | Present, mitigated | **Yes** | Client-specific r_i; zero-pad to r_max for aggregation, truncate for distribution; rank self-pruning + sparsity-weighted aggregation | r_i-dependent | Comparison table only |
| FLoRA (Wang et al., arXiv:2409.05976) | Eliminated (stacking is exact) | Yes | Stack all clients' A's and B's; product of stacked matrices = exact sum of products | Σr_i (grows with K) | Comparison table only |
| Bai, Lee/FedSVD, Zhang/FedRot-LoRA | — | — | (fill in from reading) | — | Comparison table only |

<!-- The written comparison should classify every method on along these two axes plus privacy compatibility (does the method tolerate DP noise / secure aggregation?) — that third axis is what connects this to the group's IT-privacy agenda and is worth a paragraph per paper even for methods not implemented. -->

## Harness architecture

Refactor `train_lora.py` into components, where the current script becomes 
roughly the `Client.local_update` body plus a config.

```
fed-lora/
  PLAN.md               # this file
  config.py             # dataclass: model, rank, K, rounds, local_epochs, alpha_dirichlet, method, seed
  data.py               # load dataset, split into articles, partition across clients
  client.py             # Client: holds data shard; local_update(adapter_state) -> adapter_state, n_tokens
  aggregate.py          # Strategy interface + FedITAvg, FFAAvg
  server.py             # training loop: broadcast -> local updates -> aggregate -> eval
  eval.py               # perplexity on validation; per-client perplexity
  main.py               # entry point; one method+config per run, logs to CSV/JSON
  plots.ipynb           # perplexity-vs-round and vs-bytes figures (plotly)
```

Design decisions:

- Adapter state only. Use `peft.get_peft_model_state_dict` / `set_peft_model_state_dict` so the server and clients exchange only LoRA tensors. The base GPT-2 is loaded once and shared (in simulation, one model instance; swap adapter states per client to keep memory flat).
- Manual local loop, not `Trainer`. Per-round `Trainer` instantiation is heavy and hides the optimizer state question (reset optimizer each round — standard FedAvg assumption). A ~20-line loop over the collator's batches with AdamW is easier to instrument.
- Strategy interface. `aggregate(states: list[dict], weights: list[float]) -> dict` plus method-specific hooks: FFA-LoRA needs `init` (broadcast one shared A, mark it non-trainable via `lora_B`-only optimization).
- Keep it model/dataset-agnostic: the harness should accept any PEFT model + tokenized dataset
- Communication accounting. Log bytes uploaded per round per method (count adapter tensor sizes) — this gives the perplexity-vs-communication plot, which is where FFA-LoRA's halving shows up.

## 4. Data partitioning

Recommended first-pass dataset: keep WikiText-2 if you want the fastest path from the current script to a federated benchmark. It is already wired, small enough for repeated runs, and close to the language-modeling setup in the existing code.

If you want a slightly cleaner benchmark for comparing FedIT vs FFA-LoRA only, TinyStories is the better alternative: it is easy to shard into short sequences, trains quickly on GPT-2-scale models, and usually produces less noisy curves than WikiText-2. The tradeoff is that it is less representative of the original WikiText-2 setup.

Split the raw train corpus into articles or documents, then use two regimes: **iid** — shuffle documents uniformly across K clients; **non-iid** — quantity skew via Dirichlet(α) over document counts, α ∈ {100, 1.0, 0.1}. If the chosen dataset does not have clear article boundaries, use fixed-length document chunks instead. K = 8–10 clients, full participation.

<!-- ## 5. Experiments and metrics

Primary metric: validation perplexity vs. communication round, and vs. cumulative bytes uploaded. Secondary: per-client perplexity spread (heterogeneity), wall-clock, trainable-parameter count.

| # | Question | Setup | Expected result (from papers) |
|---|---|---|---|
| E1 | Does aggregation bias hurt? | FedIT vs FFA-LoRA, iid vs α=0.1, local epochs ∈ {1, 3} | Gap grows with heterogeneity and local epochs; FFA-LoRA matches or beats FedIT at half the communication |
| E2 | Sanity: federated → centralized limit | K=1, or K clients with 1 local step | Matches centralized `train_lora.py` curve |
| E3 (stretch) | DP interaction | Gaussian noise on client updates, FedIT vs FFA-LoRA | FFA-LoRA degrades less — this is its headline claim and the bridge to Yue's privacy agenda |

Fixed across runs: GPT-2, `target_modules=["c_attn"]`, α_lora=32, fp16, max_length=256, seeded partitioning and init. Budget per run at 1% of WikiText: minutes; scale to 10–100% once the harness is trusted. Every figure needs ≥3 seeds with mean±std.

Sanity checks to build in before any comparison run: (a) E2 equivalence test; (b) unit test that FFA-LoRA aggregation is exact (assemble ΔW both ways, assert allclose); (c) HetLoRA pad-then-truncate round-trip is identity for equal ranks. -->

## References

- Zhang et al., *Towards Building the Federated GPT: Federated Instruction Tuning* (FedIT) — [arXiv:2305.05644](https://arxiv.org/abs/2305.05644)
- Sun et al., *Improving LoRA in Privacy-preserving Federated Learning* (FFA-LoRA), ICLR 2024 — [arXiv:2403.12313](https://arxiv.org/abs/2403.12313)
- Cho et al., *Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models* (HetLoRA), EMNLP 2024 — [arXiv:2401.06432](https://arxiv.org/abs/2401.06432)
- Wang et al., *FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations* — [arXiv:2409.05976](https://arxiv.org/abs/2409.05976), code: [github.com/ATP-1010/FederatedLLM](https://github.com/ATP-1010/FederatedLLM)
- Plus Bai, Lee/FedSVD, Zhang/FedRot-LoRA from Yue's email (comparison table only).
