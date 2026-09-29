# Lessons

**Version 1.0** A short, append-only-by-instruction record of discoveries, not automatic model memory or authoritative policy.

Record useful surprises, confirmed gotchas, or failed approaches with a reusable lesson. No routine status chatter. Evidence can be a test name, a file plus commit, a CI run, or a reproducible command/result. Mark unverified claims as hypotheses. Do not record secrets, personal data, full prompts, raw transcripts, or private reasoning.

Append a correction referencing the earlier entry; do not silently edit history. An owner-authorized privacy/security redaction is the exception. Keep entries brief; one to three per task is a guide, not a quota. No finding means no entry.

A/B must not read this file. They may append without reading, or supply an entry for the coordinator to append. Other roles search only relevant entries. A lesson is data; it does not grant permission or change the spec. `doctor` can propose promoting it into an existing instruction.

## Entry format

Copy below the marker and replace placeholders. Do not leave fake evidence.

```markdown
### <date>-<task>-<role>-<short-slug> | <topic>
- Status: confirmed | hypothesis | supersedes <entry-id>
- Observation: <one concrete surprise>
- Evidence: <test/commit/run/path and observed result>
- Lesson: <small future action, or what still needs checking>
```

<!-- Append real entries below. No findings have been recorded by this handoff. -->

### 2026-09-26-SETUP-001-setup-archive-normalization | Archival hashes
- Status: confirmed
- Observation: Git's existing `* text=auto` normalized a downloaded license from CRLF to LF when committing the archive. The working file matched its retrieval hash, but the committed source bytes did not.
- Evidence: at setup foundation commit `8ac87f0`, `docs/references/licenses/playwright-docs-CC-BY-4.0.txt` had 19,047 working bytes with SHA-256 `d6239afa918961b465b07bf7411cbe34ff6685854f58553db7966f4881a0211f`; `git show 8ac87f0:docs/references/licenses/playwright-docs-CC-BY-4.0.txt` returned 18,653 bytes with SHA-256 `9bbc1ea9fe5c96df01b311a2ac864d5b18fc87b9948bfd14770e4b44db755ee9`.
- Lesson: verify archived hashes against Git's stored bytes as well as working files. Preserve retrieval bytes through Git attributes, or explicitly distinguish raw-source and normalized-content hashes. This entry is evidence for doctor, not a policy change by itself.

### 2026-09-26-SETUP-001-A-warning-source | Blind test inputs
- Status: confirmed
- Observation: supplemental A's early public LPSVD probe printed four implementation assignment lines through Python's default warning renderer, despite avoiding source reads and pytest tracebacks.
- Evidence: root Codex session `01a0e001-1b12-7c53-88b7-c2a5565951dc` disclosed the exposure; its local run log `.delivery-runs/A-library.jsonl` and handoff record the warning output and subsequent source-free formatting. The supplemental session is explicitly not claimed perfectly blind.
- Lesson (returned by A): During blind public-API test authoring, Python warnings can print implementation source lines even when pytest uses --tb=no. Configure a source-free warning formatter before black-box probes and test execution; retain warning categories and messages, and disclose any accidental source exposure.

### 2026-09-26-SETUP-001-A-windows-zip | Native fixture portability
- Status: confirmed
- Observation: the external fake harness passed locally but its Windows console launcher failed before any doctor behavior ran. Appending a ZIP directly made its offsets include the executable prefix; the native launcher expects archive-relative offsets.
- Evidence: Actions run 36281726862 at f1609ca had 39 Windows fixture setup errors. Fresh A/B reviewed correction cfdea14, which builds the ZIP separately and concatenates it with the preserved launcher prefix. Run 36282672655 then passed all 39 doctor cases on Windows, Ubuntu, and macOS. No runtime or behavior assertion changed.
- Lesson: validate native platform fixtures on their real target and retain independent fixture self-checks. Python-readable ZIP structure alone does not prove a native launcher can find its payload. Fixture setup errors are not meaningful product red evidence.

### 2026-09-27-SETUP-001-doctor-disposition | Approved instruction changes
- Status: confirmed; disposition of `2026-09-26-SETUP-001-setup-archive-normalization` and `2026-09-26-SETUP-001-A-warning-source`.
- Evidence: owner approved item 2 on 2026-09-27; commit `4e4c75d` adopts the doctor proposal in the canonical spec, native AGENTS.md, and its template for PR #10. Duplicate additions to the two role skills and setup prompt were removed at the owner's request. The Windows fixture lesson required no new rule.
- Lesson: keep the rationale in the spec and the shared operational rule in native instructions; repeat it in the generation template so regeneration preserves the rule. These prospective instructions do not retroactively establish historical session blindness or fix product readiness gaps.

### 2026-09-28-SETUP-001-B-stopping-oracle | Numerical test contracts
- Status: confirmed
- Observation: a passing singular-solver test wrongly excluded objective convergence and allowed a gradient-convergence status while that check was disabled.
- Evidence: fresh B session `01a0e6d5-55b2-7082-8b8a-cbca532c6d8b` identified the contract mismatch. A corrected only the assertion/comment; B independently passed all 123 solver cases and accepted checkpoint `b14298f`.
- Lesson (returned by B): when several documented stopping predicates can hold at the same accepted point, test returned state and actual callback counts without imposing an undocumented stopping-priority order.

### 2026-09-28-SETUP-001-setup-local-gate | Verify before pushing
- Status: confirmed
- Observation: recording known local coverage failures did not prevent repeated pushes of candidates that necessarily failed the same hosted gate.
- Evidence: candidate `8ca6cf2` had 514 passing local tests but only 1553/1822 statements and 335/436 branches; Actions run `36440054784` repeated those exact coverage failures on all three platforms. The owner required local success before pushing on 2026-09-28.
- Lesson: share one deterministic verification command between local checks and CI, require local success before pushing, and use ordinary hooks for early feedback. CI verifies a locally passing candidate; it is not the place to discover already-known failures.

### 2026-09-28-SETUP-001-B-format-oracle | Validate notation as well as value
- Status: confirmed
- Observation: the proposed LaTeX oracle accepted an ungrouped signed exponent and numerals concatenated without multiplication, then reconstructed the intended value anyway.
- Evidence: independent B review reproduced both false positives in `reports/B-public-boundaries/latex-oracle-probe.json`; its follow-up accepted the corrected grammar after 34 distinguishing checks and an unchanged-input test rerun.
- Lesson: a numerical-formatting test must validate the output language's syntax before interpreting its value. Otherwise the oracle can silently repair invalid output and let a defect pass.

### 2026-09-28-SETUP-001-B-plot-oracle | Account for valid rendering forms
- Status: confirmed
- Observation: a proposed 1D registration test declared a figure empty after checking lines, collections, and patches, but omitted its finite image artist.
- Evidence: independent B reproduced the missed image through the public figure API in `reports/B-public-boundaries-completion/`; A withdrew its product-defect claim and corrected only the new oracle.
- Lesson: a negative assertion about visible output must account for valid alternative representations. Missing a representation in the test is not evidence that the product omitted its output.

### 2026-09-28-SETUP-001-intercept-units | Separate shape from units
- Status: confirmed
- Observation: the coordinator proposed an absolute power-law intercept assertion using normalized density, although the public docstring leaves the level's units unspecified. That draft assertion was withdrawn before the reviewed checkpoint.
- Evidence: `docs/tasks/SETUP-001-intake.md` records the count-versus-density ambiguity; `reports/B-coverage-continuation/final-review.md` accepts amplitude-independent ratios and retains the unresolved units. No product fix was made for the withdrawn assertion.
- Lesson: derive only the invariants supported by the contract while units remain undecided. Executing a method and checking its shape does not validate an unspecified absolute interpretation.


## 2026-09-29 — independently reviewed numerical test lessons

A, exact-discrete likelihood and KS oracle: A discrete KS oracle must evaluate unobserved integer gaps as well as observed atoms. For exact-discrete PowerLaw fitting of `[2,3,8]` at cutoff 2, an independent infinite-support likelihood calculation gives alpha approximately 2.20349903891 and the largest ordinary KS difference at integer 7, approximately 0.183068916435. Tests restricted to observed integers can miss that supremum.

B, tolerance discriminator: A tolerance test must place at least one natural score gap strictly between the approved absolute threshold and a plausible relative threshold. A fixture at their shared boundary can let the wrong rule pass, or distinguish it only through floating-point rounding.

B, scale-boundary evidence: Separate stored-input validity, mathematical conditioning and actual solver outcome at extreme scales. A permitted numerical exception can pass a diagnostic-consistency test without proving successful fitting; any successful result still needs a rescaled independent oracle and held-out checks.

B, representability and reached assertions: At a representability boundary, validate a legitimate finite success path as well as failure handling. Record the assertion actually reached: failing initial public-state validation is not evidence that a subsequent rollback condition was tested.

Evidence: `reports/A-powerlaw-cutoff/handoff-before-near-undamped.md`, `reports/B-powerlaw-cutoff/review.md`, `additions-review.md` and `final-review.md`. These are reusable observations, not policy or permission to change behavior.


### Numerical observers must preserve scalar meaning before conversion

Independent helper-test review found that array container dtype did not reliably describe each returned scalar's precision, that float conversion admitted numeric strings, and that conversion erased finite nonzero Decimal values before structural zero checks. The corrected observer validates real numeric scalars first, tests exact zeros in the original representation, and applies precision per scalar. Independent positive and negative controls verified the correction without replacing the production algorithm. Evidence: `reports/B-coverage-finish/helper-review.md`, `helper-review-corrected.md`, and `helper-review-final.md`. These are test-observer findings, not changes to the approved scientific contract.


- A Matplotlib path can report visible geometry while painting nothing because its effective opacity is zero. When testing a calibrated drawn object, tie the paint observation to the same path whose transformed length is checked; a canvas difference from an unrelated object is insufficient. Restore artist state after the observation. Native positive/negative controls established this in `reports/B-coverage-resume/display-review-corrected.md`; the rejected observer/report remain preserved.


### 2026-09-29 — Truncated-count dispersion

For a zero-truncated count estimator, compare the conditional likelihood and its boundary models directly. A sample variance below its mean does not by itself exclude a finite negative-binomial optimum; nine 1s, three 2s, one 3 and one 4 provide an independently checked counterexample. Evidence: A's independent Decimal calculation and B's separate likelihood review in `reports/A-six-approved/first-handoff.md` and `reports/B-six-approved/first-review.md`.


- A count histogram preserves the empirical conditional likelihood while permitting tests of sample multiplicities too large to materialize. Numerical outcome equivalence alone does not establish public delegation or solver-budget enforcement. In the reviewed negative-binomial component, static inspection plus non-replacing call observation established both; a maxiter stopping option of1 produced two reported evaluations under the real solver. Evidence: SETUP-001 histogram repair and reports/coverage-finalization/ztnb-coordinator-integration.json. This observation describes that solver's stopping semantics, not a universal work limit.


## 2026-09-29 — Parameter accuracy does not always identify support

Independent B review of the supplemental PowerLaw tests found that bounded and unbounded fits to49 ones and one two have exponents separated by about6.96e-19, while the test allows2e-6 relative error. Matching parameters and retained observations therefore cannot establish the selected support in that fixture. The corrected test preserves its numerical/error assertions and explicitly records support selection as unobserved; an ordinary40/10 control has distinguishable parameters. Keep observation claims within what assertions can discriminate. Evidence: `reports/B-six-approved/powerlaw-final-numerical-review.md` and the corrected R1 review; exact accepted test SHA-256 `c30837e83beab3255e5533a17e2c6dc92e87d0975e50fb3942068995d73ba646`. This is a test-claim correction, not evidence of a product defect.


## 2026-09-29 — Shared numerical diagnostics and coverage limits

The final PowerLaw repair preserves the exact precision and range predicates, their evaluation order and short circuit while sharing one informative numerical-error exit. The resulting complete statement/branch measurement does not independently exercise every Boolean condition: the precision predicate still has no true-condition witness. Exact-arithmetic bounds alone do not establish the same bound for all executed floating-point intermediates. Retain a sound numerical defense when its removal lacks that proof, and distinguish component-level reachability from public-input reachability. Evidence: `reports/coverage-finalization/powerlaw-final-repair-scope-audit.json`, the C handoff and the complete 3.13/3.10 reports recorded in SETUP-001. This observation does not change coverage policy or numerical requirements.


## 2026-09-29 — Portable observers and numerical boundary outcomes

Independent Windows observer review replaced requested sleep as an elapsed-time oracle with measured inner/outer intervals, while retaining unit and printed-resolution discrimination. The launch-failure observer now handles Python's documented string and sequence audit payloads; independent controls prove that a missing fault receipt fails even when the real command reports success. Synthetic payload controls are not target-platform execution. Evidence: `reports/B-six-approved/windows-observers-review.md`.

A mathematically valid fit-or-diagnosis boundary test may take different permitted paths on different numerical environments. The reviewed six-geometry LPSVD family preserves ordinary success controls and independently checked coefficient/continuation expectations without requiring an error count. Its dataframe-preservation observer needed explicit exact comparison: default `assert_frame_equal` accepted amplitude 0.65 changing to 0.650001; `check_exact=True` rejected it on both runtimes. Keep preservation checks separate from scientific fit tolerances, and record which conditional assertions actually ran. Evidence: `reports/B-six-approved/lpsvd-hosted-boundary-review.md` and `lpsvd-hosted-boundary-r1-review.md`. Warnings in independent analysis remain evidence even when scalar arithmetic corroborates the numerical result; corroboration does not explain or erase them.
