# SPECTRUM-FITTING-001: fit selectable profiles to a 1-D spectrum

**Version 1.0.** Public contract R4 is in
[SPECTRUM-FITTING-001-contract.md](SPECTRUM-FITTING-001-contract.md).
This scientific task is separate from the routine-maintenance ten-task pilot.

## Current state

- Approval/scope: R3, the 33-scenario exception and R4 physical-plausibility
  clarification remain approved. The owner requested more complicated examples
  with more peaks. This update adds documentation figures/data using the existing
  public fitter; it changes no product behavior, numerical contract or tests.
  Existing authorization covers updating `codex/spectrum-fitting` and
  [PR #18](https://github.com/david-hoffman/dphtools/pull/18) against `codex-main`.
  Merge/release remains unauthorized.
- Route: documentation author plus fresh independent visual/documentation review.
  Reuse the accepted R4 numerical checkpoint and existing runtime. No new
  specialist test window or product repair is opened; prior allowances stay spent.
- Candidate pointer: base `576602a6497ce1b893c2adce61c8a0a56136a592` passed canonical
  full verification, fresh D3 and the actual Linux/macOS/Windows PR matrix.
  [PR #18](https://github.com/david-hoffman/dphtools/pull/18) retains its exact
  candidate/results/accounting. Prepare this record before verifying the new
  docs-only candidate; record that candidate's own hash and actual results outside
  its tracked tree, in the conversation or linked PR.
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
- Review/check pointer: final visual-review and exact local verification results
  belong in the conversation/PR. Validate all new asset hashes, data/table/curve
  consistency and relative links. Product, tests, fixtures and delivery tooling
  remain byte-identical to the passing base. Under the owner's supplied
  instructions, run canonical full on the exact docs candidate before updating
  the open PR. Hosted checks remain pending until their actual completion.
- History/limits: D1/D2 rejections, failed coverage receipt, all rejected test
  rounds, numerical failures, cancelled CI and previous spending remain retained.
  The artificial 1e12-offset accuracy example stays outside R4. Python 3.8 and
  native wider-than-float64 remain unverified; shell-hook coverage is unsupported.
  The ten-task maintenance pilot is separate and remains incomplete.
- Next: complete documentation review and exact verification; push the unchanged
  passing candidate to PR #18 and return with hosted CI explicitly pending.
- Metrics: 33 unchanged scenarios; 297 accepted spectrum cases. Historical
  specialist launches A/B 18/14, initial C one, repairs 3/3 spent, D1/D2
  rejected and D3 accepted; 39 completed formal roles. Current documentation
  author one; fresh visual reviewer one; design helper one. Closed A/B windows
  2/2, 1/2, 1/2, 1/2, 2/2, 2/2, 1/2, 1/2, 3/3 remain counted. Historical formal
  usage: 50,825,949 input (47,700,736 cached subset), 765,941 output (290,725
  reasoning subset). Current parent/helper and exact billing metering unavailable.
  No fixed execution cap, spending reset, further test rewrite or C repair authority.
