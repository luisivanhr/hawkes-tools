# Frozen-input replay map (read-only archaeology)

Inspected 2026-08-31. No sampler, likelihood/score evaluator, optimizer, truth file, or reference write was run. The only filesystem write by this subtask is this note. Required report, contract, complete-score derivation, scientific audit, evaluator/service/optimizer sources, permitted adapter sources and metadata were read. The ancestor `.codex/AGENTS.md` was empty; no experiment-local AGENTS.md was found. A requested Python metadata-only process was denied before launch; ZIP headers were subsequently inspected through read-only .NET streams. A PowerShell metadata serialization error was corrected without any file change.

In pointers below, `FROZEN` is `C:/Users/luisi/.codex/worktrees/9e00/hawkes-tools/d7_c3_orthogonalized_opt_1_work`; `MAIN` is `C:/Users/luisi/Documents/Programming/Python/Hawkes/hawkes-tools`.

## Verified accounting (metadata, not a microprofile)

| Item | Independently read result |
|---|---:|
| Raw NPZ files / completion receipts | 624 / 624 |
| Archive size in every receipt | 128 |
| Primary attempts / backup attempts | 624 / 0 |
| Distinct receipt worker PIDs | 156 |
| Production latent traces | 79,872 |
| Nodes at level 256 / 512 | 39 / 39 |
| Sum of all 78 level durations | 276.72954104000235 min |
| Training / selection level-duration sums | 217.0643347166663 / 59.66520632333607 min |
| Sum of 624 worker durations | 525.3650160350012 min |
| Results end-to-end duration | 295.35845181333326 min |
| Total raw compressed bytes | 4,477,685,688 |

Each raw archive has 20 NPY members. All 624 `trace_crossing_distance` headers are C-order `<f8`, shape `(128,)`; no object/pickle crossing array occurs in this corpus. Receipt count/size/attempt/PID checks and actual raw-file count were independent of the ledger summary. Worker durations include sampling, evaluation, compression/write, hashing, and collection (`gradient_sampling._work:35-44`); they are not score-only timing.

## Replay inputs and constructor

The authoritative job key is registry `(phase, slot, replicate, chunk)`, not completion order. `LATENT_SEED_REGISTRY.json.rows[].attempts[]` gives seed, claim/raw/receipt/quarantine relative paths. `assignments/{phase}_{slot:02d}.json` gives the exact target point, candidate ID, role and purpose. Check the raw point/seed against these and the receipt chain before evaluating.

Packing is specified by `FROZEN/gradient_sampling.py:25-33`. The seven evaluator arrays are:

- `log_weights`, `conditional_log_likelihood`: float64 `(64,128)`.
- `complete_gradients`, `conditional_gradients`: float64 `(64,128,25)`.
- `trace_gradients`: float64 `(128,25)`.
- `source_trace_log_density`, `proposal_log_density`: float64 `(128,)`.

Other members are the float64 point `(25,)`, int64 seed `(1,)`, three float64 ragged payloads (`trace_root_times`, `trace_internal_gaps`, `trace_cluster_event_times`) with int64 offsets `(129,)`, float64 cutoff/age/crossing arrays `(128,)`, int64 component counts `(128,2)`, and float64 delay sums `(128,2)`.

Reconstruct each trace using `adapter.d7.base.FiniteRootTrace`, slicing ragged payloads with its offsets and preserving trace order. Convert sequence fields to Python tuples of floats and component counts to tuple of ints, matching the original dataclass. The constructor lives in the immutable preservation payload `stationary_boundary_score.py:73`; it enforces cutoffs, ages, roots, gaps and crossing conventions. A deterministic empty-root fixture needs `crossing_distance=None`; production has numeric crossing arrays. Do not invent a serialized NaN-to-None convention for production without checking its trace semantics.

Construct `adapter.d7.TraceArchive(traces=tuple(...), proposal_log_density=<saved>, component_indices=np.zeros(N,dtype=np.int16), cutoff=<verified common saved cutoff>, seed=<saved>)`. `TraceArchive` is defined in `MAIN/d7_blind_joint_recovery_work/d7_blind_core.py:348`. Component indices are omitted by production packing but are exactly zero for the single-component own-target builder (`d7_fixed_archive_score_adapter.build_centered_archive:143-160`); this can be certified by matching the original canonical digest, not by sampling. `score_adapter.trace_archive_sha256:72-97` includes seed, cutoff, proposal, int16 component indices and every trace field. Require exact agreement with each receipt before evaluator replay. Do not call either archive builder.

The existing public call is `vector_score.evaluate_archive(point, context=context, archive=archive, windows=windows, adapter=adapter, clipping_cap=1e8)`. Compare every returned scientific array by dtype, shape and bytes, plus every diagnostic. `trace_archive_sha256`, seed, own-target identity error, value/gradient parity errors and clipped count are retained input/scientific diagnostics, not provenance exclusions. Evaluation still invokes full scalar/batch parity references (`vector_score.py:134-147`), which must not be dropped.

## Saved windows and dependency hazards

`orthogonal_common.load_paths:117-141` reads the existing `MAIN/d7_cf_joint_1_work/data/{training,selection}/path_00.npz` through `path_15.npz`, with original registry/claim/receipt checks. It does not generate paths. The archived training/selection access receipts name and hash all 32 files. `adapter.build_windows:247-261` loops path-major, then starts `(25.,325.,625.,925.)`, takes events strictly greater than start and strictly less than start+10, subtracts start, and stores `tuple(map(float, selected))`. There are 64 windows of horizon 10. Indices are int16 `path_index`; folds use path index modulo four.

Use `orthogonal_common.safe_adapter`'s permitted-source verification/read boundary; do not call the adapter's broad convenience verifier that reads the truth-generator source. Context comes from CF contract plus `MAIN/d7_adaptive_joint_recovery_v13_work/D7_ADAPTIVE_JOINT_RECOVERY_CONTRACT.json` via `adapter.load_contract_context:185-192`. It reaches immutable Finding-182 preservation modules. A specific side effect requires containment: `d7_blind_core.build_chart_context:137-139` unconditionally calls `WORK_ROOT/'.numba_cache'.mkdir(exist_ok=True)` before `NUMBA_CACHE_DIR` setdefault. `-B` alone does not prevent this directory operation. The acceleration harness should keep compilation caches local and prevent reference-tree writes without weakening numerical/source checks.

The frozen vector module imports local `directional_score_reference._weight_stats`. The adapter also imports adaptive/blind cores, differentiated score methods, CF implementation helpers and scratch G modules; keep the pinned imports read-only. For decision-only replay with unchanged G/C3 code, saved `directions/*.json.gradient_diagnostics` and full constraint arrays are deterministic inputs to unchanged direction/decision routines; they must not be presented as a new map-solver revalidation.

`PREFLIGHT_ARRAYS.npz` retains evaluated arrays and two deterministic fixture gradients but **not** its eight sampled latent traces. Its eight-sample full evaluation cannot be replayed from that file without resampling. Instead select production raw archives, and reuse the explicit deterministic trace constructors/events in `preflight.py:38-41`. Do not invoke `preflight.main`, which samples at line 59. The existing preproduction scalar/vector checks and scientific audit used tolerances; they do not establish the new byte-equality requirement.

## Reduction and retained-node replay

Use `vector_score.summarize(replicate_chunks, indices)` without algebraic rearrangement. `GradientService.ensure:157-165` orders replicates 0..3 outside chunks 0..(level/128-1). Level 256 is chunks 0,1 of each replicate; level 512 adds chunks 2,3. This is distinct from dispatch order, which loops chunk outside replicate. Every archive should be evaluated once and the unchanged reductions run at both saved levels.

All 13 arrays in each node NPZ must match: gradient, raw_gradient, gradient_covariance, gradient_bias, window_gradients, window_covariances, window_bias, path_gradients, path_covariances, trace_mean, trace_covariance, log_normalizers, replicate_gradients. `_vector_ratio:151-169` retains full leave-one-out subtraction order, window-by-window influence accumulation, coordinate covariance and path covariance; replacing jackknife algebra by an equivalent identity was already non-bitwise in the old audit (up to about 1e-10).

Rebuild `technical_checks`, `quality_pass`, previous gradient/covariance and all `direction_checks` in `gradient_sampling:104-125,167-172`. Level 256 is deliberately nonterminal, has no previous array and thus no positive/qualified decision; level 512 uses the 256 arrays. JSON conversion `orthogonal_common.ready` maps nonfinite floats to null, so compare archived JSON after that same conversion rather than silently dropping nonfinite fields.

## Two-arm history replay

Reconstruct six directions with `optimizer_core.construct_direction` using the regenerated construction gradient plus unchanged saved G/C3 gradient diagnostics. Construction nodes are shared `training_shared_00` for update 1, then each arm's slots 03 and 08 for updates 2 and 3. Recomputed candidate-point bytes and IDs must match all assignments, direction records, trials, checkpoints and endpoint lock.

Independent validation uses `training_shared_01` for both first directions, then each arm's slots 04 and 09. Shared validation needs both directional projections and joint coordinate covariance; construction traces never independently validate their own direction. Chord nodes are slots 00..03, 05..08 and 10..13. Selection uses `selection_shared_00` and each arm's slots 00..03. Selection common G must retain covariance for both endpoint chords.

The seven recorded trials are six ACCEPTED and control update 3 trial 0 CHART_G_OR_C3_NUMERICS_REJECTED. That full step has coordinate zero `-1.4502699609153332e-6` and was rejected before scoring. Control accepted fractions are 1,1,0.5; C3 fractions are 1,1,1. No saved 1024 expansion occurred; replay should assert that the unchanged uncertainty predicates remain false, not fabricate an unobserved 1024 branch. Saved `gain.node_ids` provides exact five-node order. Both arms stop at THREE_ACCEPTED_UPDATES_LIMIT.

Run unchanged `integrate_nodes` and `combine_integrals`; rebuild independent-direction gates, quadrature cap, acceptance lower-bound predicate, cumulative gains, both selection signs and final paired decision. **Do not reduce archived coefficient dictionaries directly:** `write_json(sort_keys=True)` destroys their original insertion order. `integrate_nodes` builds coefficients in Simpson-node order (shared G first), and paired comparison inserts control's coefficients before C3's. Rebuild coefficients from regenerated integrals and preserve that insertion order in the covariance sums (`run_experiment:107-114`).

Expected final IDs: control `0a66404ee65ae41fdba9959bd97c829fae168c112b55cb43665adae7ea2a5423`; C3 `40801ffe5e4369329b27d9c3e1686f9c82f85b6eb8c7546cd96c8aa92a73edb0`. The scientific decision remains PAIRED_DEVELOPMENT_COMPARISON_UNRESOLVED. The trace arrays must flow through the changed evaluator before these downstream comparisons; archived decision values alone are not replay.

## Payload exclusions and certification boundary

Legitimate exclusions: new elapsed/runtime fields, timestamps, PIDs, replay file paths, source-code/version provenance hashes for changed code, output-container hashes, and ZIP metadata/compression bytes. Retain immutable input hashes, seeds, trace digests, candidate IDs, every scientific scalar/array, diagnostic gate and decision branch. Compare decoded NPZ payloads rather than new ZIP bytes; do not confuse container-hash exclusion with permission to change numerical arrays. Full constraint/map and descriptive fitted-curve calculations are unchanged and can remain pinned evidence; clearly separate those from changed-evaluator replay coverage.

This map establishes replay availability and ordering, not baseline reproducibility, candidate exactness, a hotspot, speedup, or universal bitwise equivalence. Those require the root task's numerical runs and final independent certification.
