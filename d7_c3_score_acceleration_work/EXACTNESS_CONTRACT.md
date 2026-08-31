# Exact saved-input acceleration contract

Frozen before implementation comparisons on 2026-08-31.

The reference is the hash-locked D7-C3-ORTHOGONALIZED-OPT-1 implementation and
its existing 624 saved 128-trace inputs, 39 scoring nodes at both 256 and 512
traces per replicate, and existing two-arm decision history. No sampling,
optimization continuation, new observation/model paths, seeds, cohorts, truth
access, gate/accuracy/cutoff changes, or model-family changes are authorized.

Every retained scientific scalar/array at the original interface must preserve
values, shape, dtype, ordering and raw bytes, including signed zero and nonfinite
behavior. Bounds checking, normalization and deterministic reduction order stay
unchanged. All 25 coordinates, complete boundary-law plus conditional score,
control variate, covariance, bias, MCSE, independent validation, shared-node
covariance, G/C3 checks, quadrature allowances and decision branches remain.

A specialized directional interface would need exact equality to the original
full-gradient projection and propagated uncertainty; it may not replace the
complete public interface. No reference-output lookup, expected-output
regeneration, rounding, tolerance relaxation, reassociation or approximate
substitution can establish certification.

Non-scientific exclusions: timestamps, elapsed durations, process identifiers,
source/version provenance hashes, relocated file paths, and lossless container
metadata (including ZIP timestamps). Decoded payloads are compared. Input seed,
candidate and trace identities are scientific provenance and are not excluded.

Sequence: reproduce unmodified baseline on representative saved inputs; profile;
test a small number of changes with identical inputs and cold/warm repeated
timings and memory; only after exactness and real speed benefit run one complete
frozen-input candidate replay and full unchanged reductions/decision arithmetic;
one fresh independent final certifier, at most one focused repair/recheck.

Finite replay certifies only the specified runtime and corpus, with an operation
preservation argument. It is not universal bitwise equivalence across platforms
or future model families. Non-exact variants stay outside the mergeable path.

All reference locations remain read-only, bytecode disabled, compilation and
profiling caches local. The parent handles integration and worktree removal
only after a certified result and preservation of evidence. No merge, push or
worktree deletion occurs here.
