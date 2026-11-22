# Experimental fixed expert subsets for code review

This local research fork supports a fixed per-layer subset of Qwen3.8-Flash-Next routed experts. Shared experts, the remaining dense model and top-10 computation remain unchanged. It is based on upstream commit 87dfca0e7c3ee24ad36cc3f69a31b859587245f8.

Status: CPU mapping, sparse-loader and pack tests passed. A CUDA build and Qwen inference have **not** been run in the preparation environment. Complete those checks on the target GPU before accepting benchmark results. This is not a supported upstream feature.

## Prepare a subset

`python tools/review_subset.py --base-pack /models/full-pack --mask /path/mask.json --out /models/review-half --profile data/expert-profile.bin`

The mask JSON has schema_version=1, layers=48, num_experts=512, top_k=10, keep_fraction, and allowed: one sorted list of original expert IDs per layer. Every layer retains the same number, at least 10. The full mask reuses the original pack. The base must be a standalone native v3/v4 GGUF pack with raw BF16 router rows (iq_pack.py --compat-bf16); no existing expert pruning, borrowed extra.bin, Hadamard v5 or NVFP4 layout.

Use the output as --pack, keep --native pointing to the original full GGUF, and use its expert-profile.bin. No expert tensor is requantized or duplicated on disk. The compact router rows and ID map select original expert file slices. Both native RAM arena and mmap/resident-budget sources are adapted. The map is process-wide, immutable after startup, and shared by all native batch slots.

## Batched reviews

The existing server config parallel=N (normally 2..8) selects native `--batch N`. Each slot retains upstream's separate GDN, QSA and PLE state. All slots on one worker use one mask. For different masks, restart the process or use separate workers. Batch many code items under each loaded mask to amortize loading. Do not reuse KV/GDN/prefix state between masks.

For a controlled run: drafting=off, mtp=false; `--spec 2 --suffix-draft 0 --lookup-chain 0 --lookup-max-t 0 --adapt-swaps 0 --pcie-frac 0`. Disable STRATA_ROUTE_TAIL_SKIP, STRATA_ROUTE_PRIOR, STRATA_ROUTE_RESIDENT, STRATA_ROUTE_RESIDENT_MTP, STRATA_PREFILL_CPU_SHARE and STRATA_EXPERT_NIBBLE. Use actual spaced flag/value arguments, as in `--spec 2`. Native GGUF requires spec>=2 even with no drafter.

Startup logs print selected count and an FNV1a64 checksum of the actual map. The normal server /metrics.engine response carries review_subset_schema, review_subset_source_experts, review_subset_kept and review_subset_fnv1a64 (full for the baseline). Read effective batch_slots there: the engine may reduce the requested size if VRAM is insufficient.

The 512-expert specialized batch router falls back to the existing generic router for compact counts; expert/attention batching remains. Measure actual performance. A half-size pool still computes 10 routed experts per token; its benefit can be expert residency, not half the arithmetic.

## CPU test without CUDA

`g++ -std=c++17 -O2 -pthread -Iinclude src/core/review_subset_test.cpp -o /tmp/review_subset_test`

Create an empty temporary directory and pass it to that test executable. It verifies identity mode, strict mapping, actual sparse plane reads and padding, short-read errors, a cross-language map checksum and four concurrent readers. It does not test GPU inference.

## Required GPU validation

Build this fork for the target GPU. Compare the original full pack against a stock build from the pinned base with identical settings and greedy token IDs. Check actual masked geometry, source-to-compact mapping, memory and sane output. Verify that simultaneous independent requests do not leak state. Then run the paired code-review benchmark with equal review count/token budgets and separately report wall-clock costs.
