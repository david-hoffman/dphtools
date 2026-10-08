# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Public contract R4 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This scientific task is separate from the routine-maintenance ten-task pilot.

## Current state

- Approval/scope: R3, the 33-scenario exception and R4 physical-plausibility
  clarification remain approved. The owner requested repair of failed CI for
  [PR #18](https://github.com/david-hoffman/dphtools/pull/18). This update pins
  interpreter patch versions consistently across collection, workers and
  aggregation. It changes no numerical behavior, contract, tests or check gates.
  Existing authorization covers updating `codex/spectrum-fitting`; merge/release
  remains unauthorized.
- Route: environment/tooling repair, with independent diagnosis and fresh
  infrastructure diff review. Reuse the accepted R4 numerical checkpoint and
  existing runtime. No test window or scientific product repair is opened;
  prior allowances stay spent.
- Candidate pointer: documentation candidate
  `0634499296e143bab6a16daba3f504b5c6e4391d` passed exact canonical local full
  verification. Its [hosted run](https://github.com/david-hoffman/dphtools/actions/runs/37483079692)
  failed before worker 0 ran tests: Linux collection used Python 3.10.21, while
  worker 0 and aggregation used 3.10.22. The interpreter patch was the only
  differing portable manifest field, independently confirmed from artifacts.
  Pin Linux to 3.10.22 and macOS ARM64/Windows to their supported 3.10.11 builds;
  preserve exact identity checks. Prepare this record before verifying the repair;
  record the new candidate's own hash/results outside its tracked tree in PR #18.
- Runtime/checkpoint: C3 product commit
  `f3a655709920f6c7f13a392c5c6265ae8d2f045e` remains unchanged, including internal
  physical-unit conditioning. Accepted A18/B14 checkpoint: 297 cases across
  unchanged 33 scenarios, same correction window accepted/closed at 3/3.
  Tests SHA-256 `22083617a37363f398116749087c4b0a8f080cc3cb27e4ebcaaeb0093342bf35`;
  report SHA-256 `b376f8eb13184564ec3836af0c818b65c011b3f5bf385a9f587cfa41f4d8e5ea`.
- Human plots: preserve the original [single-peak examples](../examples/spectrum-fitting/README.md).
  Add [multi-peak examples](../examples/spectrum-fitting/MULTI-PEAK.md): six
  Gaussian peaks found automatically, seven Lorentzian peaks fitted from center
  guesses, and eight Voigt peaks fitted from full guesses, including overlapping
  pairs. Four PNGs show totals, individual fitted/true components and residuals;
  NPZ/JSON retain physical samples, truth, guesses, fits, covariance, seeds and
  hashes. Three actual default-LM calls returned without warnings and preserved
  inputs. Residual RMS is 0.00965401–0.01041662 V for 0.01 V simulated read noise.
  These are human gut checks, not new numerical acceptance or uniqueness claims.
- Review/check pointer: the multi-peak visual/documentation review accepted the
  published plots/data. Runtime, tests, fixtures and verification programs remain
  byte-identical to the accepted base. For this CI repair, retain the failed run's
  diagnostics, verify the narrow workflow diff independently, and run canonical
  full on the exact candidate before pushing under the owner's supplied rules.
  A fresh complete hosted workflow must pass; its outcome remains pending until
  actual completion. Failed-job-only retries cannot reuse the prior attempt's
  sealed collection manifest.
- History/limits: D1/D2 rejections, failed coverage receipt, all rejected test
  rounds, numerical failures, cancelled CI and previous spending remain retained.
  The artificial 1e12-offset accuracy example stays outside R4. Python 3.8 and
  native wider-than-float64 remain unverified; shell-hook coverage is unsupported.
  The ten-task maintenance pilot is separate and remains incomplete.
- Next: review and verify the interpreter pin; push the passing candidate to
  PR #18, then confirm the complete Linux/macOS/Windows CI gate succeeds.
- Metrics: 33 unchanged scenarios; 297 accepted spectrum cases. Historical
  specialist launches A/B 18/14, initial C one, repairs 3/3 spent, D1/D2
  rejected and D3 accepted; 39 completed formal roles. Current documentation
  author one; fresh visual reviewer one; design helper one. Current CI repair:
  author one, independent diagnosis one, fresh infrastructure reviewer one. Closed A/B windows
  2/2, 1/2, 1/2, 1/2, 2/2, 2/2, 1/2, 1/2, 3/3 remain counted. Historical formal
  usage: 50,825,949 input (47,700,736 cached subset), 765,941 output (290,725
  reasoning subset). Current parent/helper and exact billing metering unavailable.
  No fixed execution cap, spending reset, further test rewrite or C repair authority.
