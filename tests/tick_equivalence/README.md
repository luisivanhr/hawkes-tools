# Tick Equivalence Ledger

This directory tracks breadth-first algorithmic parity against a frozen snapshot
of the local tick Hawkes test surface without importing tick's compiled
extensions or requiring the original tick checkout.

Run the ledger with:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests\tick_equivalence
```

Print the current status counts with:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" tests\tick_equivalence\report_equivalence.py
```

Status meanings:

- `pass`: source-backed behavior is covered by current `hawkes-tools` tests.
- `xfail_equivalence_gap`: a tick test is in scope but algorithmic parity is not
  yet claimed.
- `skip_optional_backend`: the reference test depends on an optional backend
  that is intentionally not exercised in the current environment.
- `out_of_scope_non_hawkes`: reserved for future inventory items outside the
  Hawkes/point-process target.

Current ledger target for this slice: 171 `pass`, 0 `xfail_equivalence_gap`,
and 0 `skip_optional_backend`.

