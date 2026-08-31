# Exact renewal-Hawkes score acceleration

Decision: **CERTIFIED_EXACT_ON_FROZEN_REPLAY_WITH_FOCUSED_REPAIR**.
The one independent final certificate passes the bounded scope described below.

The isolated candidate accelerates the existing complete likelihood/score
evaluator. It does not continue optimization or change its model, estimator,
cutoff, precision, seeds, quadrature, gates, covariance or scientific decision.
The frozen reference decision remains
`PAIRED_DEVELOPMENT_COMPARISON_UNRESOLVED`.

## Measured bottleneck and change

The unmodified evaluator first reproduced all seven arrays and every diagnostic
of `training_shared_S00_R0_C00_A0.npz` exactly. Its cProfile trace attributed
16.17 of 18.74 profiled seconds to `renewal_trace_log_density`, which repeatedly
called the scalar fitted-law hazard and cumulative hazard over roughly 160,000
gaps. `conditional_batch` took 0.93 seconds. Profiling overhead is included in
those numbers; the comparative timings below are uninstrumented.

`vector_score.py` has only one added import and one changed call. The new
`exact_trace_density.py` batches independent spline queries within each trace.
It preserves one-element NumPy powers, each original contiguous one-row
matrix/vector contraction, Python `math.log`, the sequential gap accumulation,
stationary-age normalization, crossing survival and the unchanged kernel
density. No proposal density or saved result is substituted for evaluation.

The fast path is restricted to the existing `FittedRenewalLaw` and its
`BreakpointHazardChart`. Unsupported law/chart types, exceptional inputs and
caller-defined floating-point traps take the original path. The full public
25-coordinate API, conditional recursion, boundary score, normalization,
control variate, bias/covariance/MCSE reductions and independent scalar/value
parity checks remain unchanged. The generic optimizer is untouched.
The final repair also selects the original path whenever a warning filter has
action `error`, preserving the reference's warning-as-exception order.

## Before/after measurements

Three alternating-order paired evaluations per existing archive, same runtime,
single-thread BLAS. Every timed result was compared against archived bytes.

| Saved input | Baseline median seconds | Candidate median seconds | Speedup |
|---|---:|---:|---:|
| Training shared G | 10.351 | 2.637 | 3.93x |
| Last likelihood-only training node | 9.077 | 2.577 | 3.52x |
| Last orthogonal-C3 training node | 8.810 | 2.500 | 3.52x |
| Selection shared G | 9.811 | 2.700 | 3.63x |

Sum-of-medians evaluator speedup: **3.65x**. Across these trials, baseline calls
took 8.75–10.66 seconds and candidate calls 2.46–2.71 seconds. Full observations
and per-call memory are in the benchmark JSON files.

Two fresh-interpreter runs on shared G took 9.595/9.607 seconds for the baseline
first evaluation versus 2.563/2.519 seconds for the candidate. Including imports,
context loading and input reconstruction, script time was 11.787/11.834 versus
4.783/4.771 seconds. Cold here means a new interpreter, not a forced disk-cache
flush. Peak working set was 240.55–240.57 MiB baseline and 239.77–240.97 MiB
candidate. Saved archive loading/reconstruction took about 0.08 seconds.
After the focused repair, two paired repeats on shared training G and the final
selection C3 node remained exact and gave an aggregate 3.66x speedup
(baseline 8.94–10.08 seconds; candidate 2.53–2.62 seconds).

On the same 16 saved archives, two repeats gave fresh two-worker pool times of
29.374/29.388 seconds, persistent two-worker times of 27.521/27.998 seconds, and
persistent four-worker times of 16.810/15.527 seconds. Every payload digest was
identical in stable job order. Available RAM before this comparison was 9.53 GB
on a 33.41 GB, 16-logical-CPU system. These are replay-only observations;
production worker defaults are unchanged and fresh-sampler stability is not
certified.

The archived 295.36-minute experiment includes sampling, scoring, constraints,
I/O and process costs. Its 624 jobs/79,872 traces, 156 PIDs, 276.73 scoring-node
minutes and 525.37 summed worker minutes were reconciled from saved metadata.
**The evaluator speedup is not an end-to-end optimizer speedup.** No new
sampler timing or production optimization was run.

## Exactness and replay evidence

The exactness contract was written before comparisons and is hash-pinned in
`CANDIDATE_FREEZE.json`. All candidate Python sources and prereplay evidence
were frozen before the one complete replay. The full replay recomputes each
of the 624 original 128-trace archives once, then uses the original replicate
and chunk ordering to reconstruct 39 nodes at both 256 and 512 traces per
replicate. No trace is regenerated. Canonical seed/proposal/trace digests and
raw-file/claim/receipt hash chains are checked before evaluation.

`history_replay.py` invokes the original direction construction, training update,
integration, covariance combination and selection comparison functions using
the recomputed node arrays. It rebuilds coefficient dictionaries in their
original runtime insertion order rather than reusing JSON-sorted reductions.
Unchanged, hash-pinned G/C3 map and derivative records are declared inputs;
their compatibility predicates are rerun, but the map solvers are not rerun.
This is replay of the existing history, not a new adaptive optimization.

Prereplay coverage includes 135 deterministic checks on empty/small/invalid
windows, zero/subnormal/nonfinite gaps, empty roots and floating-point traps,
plus a complete representative node. Additional exploratory probes covered
7,242 saved gaps at three points and spline/horizon boundaries. The original
eight preflight random traces were not regenerated because their trace inputs
were not archived. Two test-fixture construction failures are retained as
failure receipts; neither was a candidate/reference numerical mismatch.

The one complete replay finished in **573.97 seconds**, with peak worker working
set **316.75 MiB**. All **4,368 raw-array**, **1,014 node-array**, and **624 scalar
diagnostic-record** comparisons passed exact byte checks. The original history
passed **68,412 strict checks**: six accepted steps, the one chart rejection and
backtracking branch, construction/independent-validation separation, shared-G
covariance, all endpoint IDs and the unresolved final decision are unchanged.
See `evidence/full_replay/EXACTNESS_REPLAY.json`, `evidence/history_replay.json`
and `evidence/replay_reconciliation.json`. Preservation checks before and after
replay verified all 2,160 frozen manifest entries and permitted dependencies.

During its audit, the fresh certifier found one edge case: with warnings
promoted to exceptions, vector gap division could raise a later overflow before
the reference's earlier underflow. Repair occurred after the complete replay.
The **single focused repair** adds only an
upfront reference fallback for any `error` warning filter. Original helper bytes
and the exact diff are preserved in `evidence/repair_001/`; `REPAIR_001.json`
links the complete replay, source lineage, and focused tests. All **96 warning
policy/error-mode cases**, the repeated **135 deterministic cases**, and the
two repeated full-archive checks passed on the repaired helper.

There was **one complete replay, under the original frozen helper hash**, not a
second full replay under the repaired hash. The final evidence combines that
complete byte replay with the source-level fact that the default arithmetic
path is unchanged, and focused equality of the added reference-dispatch branch.
The specified runtime has `context_aware_warnings=0`; its observed default
filters contain no `error` action. `FINAL_CANDIDATE_LOCK.json` binds the repaired
delivery. This distinction is part of the certification boundary.

The independent certificate is in `INDEPENDENT_CERTIFICATE.md` and
`INDEPENDENT_CERTIFICATE.json`. It independently reconciles the complete
payloads and source preservation, and confirms the repaired exception behavior.
The parent may perform the conditionally authorized integration after preserving
all evidence and verifying the integrated code; no integration or cleanup was
performed by this task.

## Excluded alternatives and certification boundary

Vector-size powers happened to match small probes but were not selected or
certified. Direct directional propagation, projected-influence covariance,
changed BLAS reductions, compiler reassociation/FMA, reduced accuracy and
cross-candidate reuse were not introduced. No measured non-exact alternative
was promoted, and no tolerance, rounding or fixture-output lookup is used.

Certification is limited to CPython 3.14.0 (MSC v.1944 AMD64), NumPy 2.4.6,
SciPy 1.17.1, double precision, single-thread BLAS, the pinned model sources,
the specified saved corpus and deterministic fixtures. Finite replay plus
operation preservation is not universal bitwise proof across platforms,
future NumPy/SciPy versions, mutated charts or new model families.

Scientific payload comparison includes dtype, shape, ordering and raw bytes,
including signed zero. Scalar diagnostic JSON is additionally compared by
float64 bit pattern. Non-scientific exclusions are timestamps, elapsed time,
PIDs, relocated paths, source/version hashes and lossless container metadata.
Seed, candidate and trace identities are not excluded.

## Reproduction and integration handoff

Run from the integrated repository root with the same installed runtime:

```powershell
$python = 'C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe'
$reference = 'C:\Users\luisi\.codex\worktrees\9e00\hawkes-tools\d7_c3_orthogonalized_opt_1_work'
$work = Join-Path (Get-Location) 'd7_c3_score_acceleration_work'
& $python -I -B "$work\profile_baseline.py" --reference $reference
& $python -I -B "$work\benchmark.py" --reference $reference
& $python -I -B "$work\edge_fixtures.py" --reference $reference
& $python -I -B "$work\worker_benchmark.py" --reference $reference
& $python -I -B "$work\full_replay.py" --reference $reference --workers 4
& $python -I -B "$work\history_replay.py" --reference $reference --nodes "$work\evidence\full_replay\nodes" --output "$work\evidence\history_replay.json"
& $python -I -B "$work\verify_final_delivery.py"
& $python -I -B "$work\verify_preservation.py" --reference $reference
```

Complete replay output is write-once. A separately requested reproduction must
use a new `--output` directory and point the history/reconciliation tools at it;
never overwrite certified evidence. Fresh-interpreter commands are
`cold_benchmark.py --reference $reference --implementation baseline|candidate
--output evidence/NEW_NAME.json`.
`verify_replay_evidence.py` verified the original freeze before repair; use
`verify_final_delivery.py` for the delivered repaired-source lineage. Run the
focused `repair_001_recheck.py --reference $reference` only in a preserved copy
when reproducing its fixed output paths; do not overwrite linked evidence.

Only `vector_score.py`, `exact_trace_density.py` and the unchanged
`directional_score_reference.py` are runtime delivery modules. They contain no
dependency on this temporary worktree path. Integrate them as a separate adapter
directory or select this `vector_score` module in the next authorized optimizer
implementation; do not overwrite the frozen reference. Model decoding and
analytic-score dependencies continue to come from the existing main-checkout
adapter and its pinned preservation sources. The replay additionally requires
the original frozen 9e00 artifacts and saved CF windows. The released package
and manuscript trees are untouched.

The parent must preserve this report, tests, JSON evidence, certificate and the
ignored recomputed node NPZ files before removing this temporary worktree.
NPZ replay payloads and profiler binaries are intentionally excluded from Git;
no bulk raw archives are copied or committed. `.cache` is disposable and not
scientific evidence. No merge, push or worktree deletion was performed here.

Task settings: the parent verified actual turn metadata as GPT-5.6 Sol / Ultra.
Saved configuration contains `service_tier = "priority"`; the creation API does
not expose the effective per-task speed setting. Global settings were not changed.
