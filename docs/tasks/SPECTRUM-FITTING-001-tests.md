# SPECTRUM-FITTING-001 — historical independent A test evidence

Approved expectation source: [public contract R3](SPECTRUM-FITTING-001-contract.md),
approved on 2026-10-05, including the 33-scenario exception. Approved runtime
baseline supplied by the A packet: `a835a490373835fa01e4460ce2159365150d1946`.
This report preserves historical A authorship, checks, and checkpoint evidence.
The task record alone owns the authoritative Current state section; it was not
read by A. This report grants no additional scope, review rounds, implementation
repairs, or publication authority, and records no independent acceptance.

## Historical authorship and review evidence

Initial A authored the two permitted output files for all 33 approved scenarios
(263 parametrized cases). Its source-free fixture/derivative sanity check passed.
Its last baseline returned 1 failure and 262 errors, exit status 1, because the
public entry point was absent; no numerical product assertion was reached.
Initial A used one root author session, no delegated agents, and launched no B or
C sessions. Its measured authoring/baseline interval was 425 s, from 17:31:10 to
17:38:15 UTC on 2026-10-05, excluding input reading and final report preparation.
The retained attempts and checks below describe that original work.

The owner supplied B round1 **REJECT** with three concrete test defects in S10,
S13, and S23, and confirmed B otherwise found the S01-S33 mapping, numerical
expectations, and independently checked tolerances sound. A fresh root correction
session changed only those defects and this report's historical-evidence format.
The revised-check evidence below belongs to that second A session. B round1
remains spent; only B round2 remains. The same checkpoint lineage, 33-scenario
scope, and initial C plus one C repair allowance are preserved without reset.

Known authorship/review spending at this correction handoff: two root A sessions,
no delegation, one completed B round (rejection), and no C work performed by
either A session. No fixed owner time/token cap was supplied. Billing and exact
model-token usage are unavailable; the original 425 s interval is retained rather
than replaced by the correction interval.

## Scenario mapping

Every product assertion calls the real public
`dphtools.utils.fitfuncs.spectrum_fit` interface. The solver-identity test calls
that same interface in a fresh Python process. Tests use synthetic point
observations and existing NumPy/SciPy, with no product model helpers, existing
test fixtures, or implementation oracles.

| Scenario | Public test(s), prefix `test_` | Expected behavior/source |
|---|---|---|
| S01 | `s01_full_gaussian_guesses_and_schema` | R3 Gaussian unit-height equation, full rows, sorted result schema; case-insensitive aliases and list/array inputs |
| S02 | `s02_full_lorentzian_guesses` | R3 Lorentzian unit-height equation and gamma as half width at half maximum; aliases |
| S03 | `s03_full_voigt_guesses_independent_widths` | R3 normalized Voigt equation; both independently identifiable positive widths |
| S04 | `s04_center_guesses_joint_components`; `s04_center_guesses_identifiable_overlap_with_one_maximum` | R3 center-only starts, exact component count, joint fitting; two-component Gaussian mixture has only one local maximum |
| S05 | `s05_automatic_discovery_controls_and_index_default` | R3/SciPy local maxima, prominence in data units, distance in samples; omitted x means sample indices; fractional distance is valid |
| S06 | `s06_constant_background_and_covariance` | R3 default constant background, jointly estimated coefficient, full physical covariance |
| S07 | `s07_linear_background_first_coordinate_origin` | R3 `b0+b1*(x-x[0])`, with nonzero x origin and nonzero slope |
| S08 | `s08_no_background_multicomponent_sum` | R3 empty background parameters and no added coefficient |
| S09 | `s09_nonuniform_units_full_sorted_covariance_and_storage` | R3 physical units, unweighted objective on nonuniform samples, sorting of all covariance rows/columns, caller storage preservation |
| S10 | `s10_non_1d_data` | R3 1-D data domain: matrix/3-D reshapes of a valid 41-sample Gaussian with matching valid x and full guesses; scalar rejection also overlaps insufficient observations |
| S11 | `s11_nonfinite_data` | R3 finite observations: NaN and both infinities rejected |
| S12 | `s12_complex_data` | R3 real data domain, including complex dtype with zero imaginary part |
| S13 | `s13_coordinate_shape_length` | R3 1-D matching coordinates: scalar, strictly increasing x reshaped as a matrix, shorter and longer arrays; valid Gaussian/full guesses |
| S14 | `s14_coordinate_real_finite_domain` | R3 real finite coordinates, including zero-imaginary complex dtype |
| S15 | `s15_strict_coordinate_order` | R3 duplicate, descending and one reversed adjacent pair rejected |
| S16 | `s16_empty_or_malformed_guesses`; `s16_wrong_full_guess_row_width` | R3 nonempty 1-D centers or 2-D rows of family-specific width; reject ragged, scalar, 3-D, empty and wrong-column arrays |
| S17 | `s17_guess_real_finite_domain`; `s17_nonfinite_amplitude_or_width` | R3 finite real centers, amplitudes and every width, including both Voigt widths |
| S18 | `s18_nonpositive_initial_amplitude` | R3 strictly positive initial heights, every family |
| S19 | `s19_nonpositive_initial_width` | R3 strictly positive initial widths, including both Voigt widths |
| S20 | `s20_free_initial_and_fitted_edge_tail_centers` | R3 free centers on either side of observed interval; start inside or outside; identifiable Gaussian/Lorentzian edge tails |
| S21 | `s21_invalid_profile` | R3 allowed profile names, unsupported names/types rejected |
| S22 | `s22_invalid_background` | R3 allowed background choices, unsupported names/types rejected |
| S23 | `s23_invalid_prominence` | R3 finite nonnegative scalar prominence; sequence `[0, 10]` retains the fixture peak if incorrectly forwarded as a SciPy interval |
| S24 | `s24_invalid_distance` | R3 finite sample distance at least one |
| S25 | `s25_explicit_guesses_reject_discovery_controls` | R3 no ignored controls with either center-only or full guesses; includes boundary-valid controls |
| S26 | `s26_no_local_maxima`; `s26_no_peaks_meet_prominence` | R3 no-eligible-peaks ValueError, without baseline-only success |
| S27 | `s27_empty_data`; `s27_positive_residual_degrees_of_freedom` | R3 `n>p`; test `n=p` and `n=p-1`, one/two components, every family/background |
| S28 | `s28_invalid_evaluation_limit` | R3 positive integer evaluation limit; reject nonpositive, fractional, nonfinite and nonnumeric values |
| S29 | `s29_exhausted_default_optimizer_preserves_inputs`; `s29_failed_custom_status_preserves_inputs` | R3 exhausted real default solve or unsuccessful custom status raises RuntimeError; arrays unchanged |
| S30 | `s30_default_is_real_lm_with_positive_domain_and_free_centers` | R3 actual SciPy least_squares method is LM for omitted/explicit optimizer; same positive physical result, center leaves interval |
| S31 | `s31_custom_optimizer_physical_protocol_and_covariance` | R3 custom physical initial/residual/order/bounds/limit; successful real TRF adapter has no returned Jacobian; fitter computes full sorted physical covariance |
| S32 | `s32_invalid_optimizer_selection` | R3 only default `"lm"` string or callable; other noncallable choices rejected |
| S33 | `s33_malformed_custom_optimizer_output`; `s33_invalid_final_positive_domain_and_input_storage` | R3 required result attributes, real finite 1-D correct-length x, Boolean success, strictly positive final heights/widths; failure preserves input arrays even if optimizer uses initial as working storage |

Variants exercise the approved rows; they do not add contract scenarios.

## Independent oracle and tolerance rationale

The simple `_model` helper evaluates exactly the three R3 point-sample equations.
Gaussian and Lorentzian expectations use elementary expressions. Voigt uses the
permitted SciPy definition, divided by its value at zero. No peak-width measurement
routine supplies the expected fitted width: half prominence is a starting-estimate
quantity and need not equal the model's full width at half maximum.

The [SciPy Voigt definition](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.voigt_profile.html)
specifies sigma as Gaussian standard deviation and gamma as Cauchy half width at
half maximum. The [peak discovery definition](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.find_peaks.html)
supports the synthetic local-maximum counts and sample-distance interpretation.
The [width definition](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.peak_widths.html)
explains why half-prominence estimates are not numerical truth for fitted widths.
These contract-linked pages were opened during A authoring.

All core numeric fixtures have order-one peak heights and widths in the supplied
x units, multiple samples per width, wide observation intervals, and perturbed
starts near the identifiable solution. Noiseless comparisons use `rtol=3e-5` and
`atol=2e-5` for heights, centers, widths and background coefficients, and absolute
`2e-7` data units for fitted samples. These allow optimizer stopping and floating
point differences well above float64 roundoff, while requiring the exact sampled
model to be reconstructed accurately. Relative tolerances apply to each physical
parameter; absolute tolerances carry that parameter's units.

The unit-height sanity check is direct: each isolated profile equals one at its
center. A Gaussian with sigma `0.83` would have area/height factor
`sqrt(2*pi)*0.83 ≈ 2.08`; a Lorentzian with gamma `1.27` has factor
`pi*1.27 ≈ 3.99`. These order-one differences greatly exceed the parameter
tolerances. Gaussian full width divided by sigma is approximately `2.355`, and
Lorentzian full width divided by gamma is `2`. The chosen Voigt fixture has sigma
`0.72` and gamma `0.46`, so swapping the widths also exceeds tolerance. Incorrect
families are constrained by both the recovered parameters and the complete
sampled profile, including tails.

Internal consistency comparisons use `rtol=2e-10, atol=2e-10` for reconstruction
from returned parameters and `rtol=2e-12, atol=2e-12` for `data-fitted`. These are
arithmetic checks at identical physical parameters, not optimizer convergence
checks. Covariance symmetry permits `rtol=2e-10, atol=2e-12` and eigenvalues down
to `-1e-12` for numerical roundoff on these well-conditioned fixtures. This finite
covariance requirement applies only to the identifiable numeric fixtures; the
contract does not promise finite uncertainties for singular fits.

Noisy fixtures add the fixed deterministic sequence
`0.018*(sin(1.63*i)+0.6*cos(0.71*i))` in data units. Assertions against the generating
parameters allow absolute `0.04` or `0.05` in each peak parameter's own units,
and `0.01` for background coefficients. The generating parameters are not
claimed to be the noisy optimum. Stronger checks use the returned optimum's
independent model, unweighted stationarity, and independently scaled covariance.

For a sorted physical parameter vector p, the independent five-point derivative
is

`J[:,j] = (m(p-2h)-8m(p-h)+8m(p+h)-m(p+2h))/(12h)`.

Each h is `eps**(1/5)*max(1,abs(p[j]))`. Float64 epsilon gives a relative step
approximately `7.4e-4`, balancing fourth-order truncation and roundoff. All
fixture widths remain well above these perturbations. The covariance expectation
is

`s² = residual.T @ residual / (n-p)`; `C = s² * inv(J.T @ J)`.

The [SciPy covariance convention](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.curve_fit.html)
supports this residual-variance scaling. `_check_covariance_and_objective` checks
the dimensionless condition number of the column-normalized Jacobian is below
`1e4` before trusting the inverse. It compares every covariance entry after
division by `sqrt(C_expected[ii]*C_expected[jj])`, with relative `0.01` and
absolute `2e-4`. This tests cross-component and background correlations as well
as diagonal variances, and tolerates approximate numerical differentiation.
For the 81-sample constant-background fixture, omitting residual degrees of
freedom changes the variance scale by at least about 5%; it cannot fit this 1%
tolerance. Covariance is evaluated in the actual sorted physical units, so a
private log-parameter covariance or an inconsistent permutation cannot pass.

The unweighted optimum requires `J.T @ residual = 0`. The maximum columnwise
dimensionless projection `abs(J.T @ residual)/(norm(J_column)*norm(residual))`
must be below `3e-4`. Nonuniform coordinates deliberately create unequal sample
spacing; integration-weighted objectives must still satisfy this unweighted
stationarity check. No fit is required to find a global optimum for arbitrary
input data.

Automatic discovery uses three separated exact Gaussian peaks. The small middle
peak is removed by higher prominence or larger sample distance. When it is
removed, the reduced model is misspecified, so its fitted amplitudes/widths are
not compared to the original three-component truth. The test checks candidate
count, centers (absolute `0.003` x units), reconstruction, covariance and local
unweighted stationarity. Physical coordinates scaled by `0.01` distinguish sample
distance from distance in x units. No endpoint/noisy/unresolved automatic peak
identification guarantee is imposed.

The custom optimizer adapter checks residual sign, profile normalization,
background origin, row order and physical bounds at two explicitly constructed
physical vectors before calling SciPy's real trust-region reflective solver.
No background starting estimator is specified or assumed. The adapter returns
only x, Boolean success and a message. The fitter must supply its own covariance.
The default-method test wraps the actual SciPy solver in a fresh interpreter,
before the first product import, and executes the real numerical solve. The
[SciPy optimizer documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html)
supports the method distinction and LM's inability to accept explicit bounds.
It is not used to prescribe a private positivity transform.

All tolerances above were fixed before the first public baseline execution.
No product output, source, history, or prior tests informed the oracle or tolerance.

## Initial A baseline execution and classification

Initial A's last exact baseline command, run from the packet's worktree:

```sh
set -o pipefail
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/spectrum-fitting-001-a-mpl /Users/davidhoffman/Documents/GitHub/dphtools/.venv/bin/python - <<'PY' 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-baseline-final.log
import warnings
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f'{category.__name__}: {message} ({filename}:{lineno})\n'
)
import sys
import traceback
sys.excepthook = lambda kind, value, tb: traceback.print_exception(kind, value, None, limit=0)
import pytest
status = pytest.main([
    'tests/test_spectrum_fitting.py', '-p', 'no:warnings', '--tb=no',
    '-q', '-ra', '--capture=tee-sys',
])
print('BASELINE_PYTEST_EXIT_STATUS=', status, sep='')
raise SystemExit(status)
PY
```

Result: **1 failed, 262 errors in 1.64 s; pytest and shell exit status 1**.
All 263 parametrized cases were collected. The 262 fixture setup errors arise
from importing the missing public entry point. The fresh-process LM probe also
exited 1 with this source-free diagnostic:

```text
ImportError: cannot import name 'spectrum_fit' from 'dphtools.utils.fitfuncs' (/Users/davidhoffman/.codex/worktrees/spectrum-fitting/dphtools/dphtools/utils/fitfuncs.py)
```

**Classification: product defect — missing R3 public API**, high confidence.
The module path identifies the approved worktree, not an unrelated installed
package. This supplies product-red evidence for the absent entry point only.
The baseline cannot establish failures of the numerical, validation or optimizer
behaviors, because they were not reached. No environment/tooling defect or
unresolved requirement was observed; independent B still must assess test validity.

Retained attempts, without deleting earlier evidence:

| Attempt | Exact command difference | Result | Full log outside Git |
|---|---|---|---|
| Initial | Same launcher; tee path ends in `A-baseline.log` | 1 failed, 262 errors, 12.33 s, exit 1; same missing API | `/private/tmp/SPECTRUM-FITTING-001-A-baseline.log` |
| Final | Exact launcher above | 1 failed, 262 errors, 1.64 s, exit 1; same missing API | `/private/tmp/SPECTRUM-FITTING-001-A-baseline-final.log` |

Between attempts, a static test review removed an unsupported assumption that
the custom optimizer's initial array must be writable. S33 now mutates that
working array only when it is writable; either storage choice is valid under
R3. This clarification did not alter any oracle, tolerance, product expectation,
or scenario count. The final baseline was rerun on the revised exact test file.

No warnings, diagnostic messages, counts, exit status, exclusions, or skips were
hidden by the launcher. `--capture=tee-sys` retained live diagnostic output;
source-free rendering suppressed source excerpts. No warning was emitted in
either baseline. Full repository statement/branch coverage and platform-matrix
verification were not run in this narrow A execution; no coverage or merge
readiness claim is made.

The source-free helper check used the specified Python, `MPLBACKEND=Agg`, and
`MPLCONFIGDIR=/private/tmp/spectrum-fitting-001-a-mpl`. It parsed the authored test,
checked the one-maximum overlap fixture, and compared the Gaussian/Lorentzian
five-point derivatives with their elementary analytic derivatives
(`rtol=2e-6, atol=2e-8`). Exit status was zero. Dimensionless Jacobian condition
numbers at the physical nonuniform truth were Gaussian `5.3078`, Lorentzian
`5.28439`, and Voigt `21.8762`, all comfortably below the fixture limit.

Environment observed through that source-free launcher: Darwin arm64, Python
`3.13.12`, NumPy `2.5.3`, SciPy `1.18.1`, pytest `9.1.1`. These versions are execution
evidence; the approved primary definitions remain the contract-linked references.

## Initial A blind-input and allowance disclosure

Read only the A packet, approved public contract R3, root AGENTS.md, selected
`design-tests` skill, permitted test/check sections of setup.cfg, and the five
approved primary-reference pages. The two output paths were checked for existence
without reading prior content; both were absent. No product source/history,
existing tests/oracles, PROJECT.md, task state, existing lessons, or other role
conversation was read. No delegation, Superpowers, optional memory, configuration
editing, commit, push, PR, merge, or publication occurred.

The helper and both baseline launchers configured source-free warnings and
exception rendering before imports. Both baselines used `-p no:warnings --tb=no`;
the default-method child also set source-free rendering before its imports.
**Source exposure: none observed.** The diagnostic showed a product filename,
which is permitted; no product source excerpt or source-bearing traceback was
rendered. Public imports execute product code but did not expose its source.
A/B retain the packet's single checkpoint lineage and at most two
review rounds; initial C plus one C repair is unchanged. A has not accepted its
own tests and has not spent a B round or C repair.

## A correction after B round1 rejection

The owner-supplied B1 findings are **test defects**, not unresolved requirements.
This correction preserves every other numerical fixture, model/covariance oracle,
and tolerance, and leaves the S01-S33 scenario count and 263 case count unchanged.

| B1 defect | Correction | Why an unrelated rejection no longer satisfies the case |
|---|---|---|
| S10 matrix/3-D data also lacked detectable peaks | Reshape the existing valid Gaussian spectrum to `(41, 1)` and `(1, 41, 1)`; supply matching valid x, explicit full Gaussian guesses, and no background | Flattening yields the exact valid 41-sample Gaussian with 3 physical parameters; automatic discovery is bypassed |
| S13 matrix x also violated strict ordering when flattened | Reshape the fixture's strictly increasing x to `(41, 1)`; use the matching valid Gaussian and explicit full guesses with no background | Flattening restores the original valid ordered coordinates with matching sample count |
| S23 `[1, 2]` could legitimately remove the peak | Replace only that invalid sequence with `[0, 10]` | SciPy forwarding retains the fixture peak, so no-eligible-peaks rejection cannot satisfy this case |

The scalar S10 case remains a rejection check, with a matching one-coordinate x
array and valid full guesses. It also has too few observations (`1 < 3`) and does
**not** independently establish dimension validation. Only the matrix/3-D cases
isolate the dimension defect. S13 scalar and length-mismatch cases retain their
approved rejection expectations.

The approved [SciPy find_peaks documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.find_peaks.html)
was reopened during correction. It defines a two-element prominence sequence as
minimum/maximum bounds. The source-free correction sanity check measured one
retained peak at index 21 with prominence `2.999596352246759` data units for
`[0, 10]`, while `[1, 2]` retained none. This checks fixture discrimination, not
product behavior.

Correction syntax/fixture checks passed, exit status 0. They parsed and compiled
the revised test without writing bytecode, checked finite real 41-sample data,
increasing matching coordinates and lossless reshapes, and solved the flattened
fixture with SciPy's real LM solver using the unchanged independent `_model`.
The recovered physical parameters were
`[3.0, 0.19999999999999993, 0.9]`, within the unchanged parameter/model tolerances.
These checks did not import the product. Full check output is retained outside
Git at `/private/tmp/SPECTRUM-FITTING-001-A-correction-checks.log`.

The exact revised test file used in the baseline has SHA-256
`f568ad84499aa7dea8a061bc5935fad09fbf96caee733bd78664cdb907d411c0`.
The report's final hash is returned outside the tracked tree to avoid a recursive
self-hash. The packet-supplied approved runtime revision remains
`a835a490373835fa01e4460ce2159365150d1946`; no product history or diff was inspected.

Exact correction baseline command, run from the packet's worktree:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/spectrum-fitting-001-a-correction-mpl /Users/davidhoffman/Documents/GitHub/dphtools/.venv/bin/python - <<'PY' 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-baseline-correction.log
import warnings
warnings.formatwarning = lambda message, category, filename, lineno, line=None: (
    f'{category.__name__}: {message} ({filename}:{lineno})\n'
)
import sys
import traceback
sys.excepthook = lambda kind, value, tb: traceback.print_exception(kind, value, None, limit=0)
import pytest
status = pytest.main([
    'tests/test_spectrum_fitting.py', '-p', 'no:warnings', '--tb=no',
    '-q', '-ra', '--capture=tee-sys',
])
print('BASELINE_PYTEST_EXIT_STATUS=', status, sep='')
raise SystemExit(status)
PY
```

Correction result: **1 failed, 262 errors in 12.08 s; pytest and shell exit status
1**. All 263 cases were collected. The 262 fixture errors occurred when importing
the absent `spectrum_fit`; the unchanged fresh-process LM test printed child exit
status 1 and the same source-free missing-entry-point ImportError recorded above.
No numerical or validation product assertion was reached. No warnings or skips
were reported; diagnostics, failure/error counts, and exit status were retained.
The new correction log does not replace either retained initial-A attempt.

**Classification: product defect — missing R3 public API**, high confidence.
This remains product-red evidence for the missing entry point only. The corrected
tests still require fresh independent B round2 review; A grants no acceptance.
No unresolved requirement or environment/tooling defect was observed. The earlier
B1 test defects are recorded and corrected, not relabeled as product failures.

Correction input disclosure: read the A packet, contract R3, session-supplied
AGENTS instructions, selected design-tests skill, setup.cfg test/check conventions,
the specifically permitted prior test/report, and the approved find_peaks page.
No product source/history/diff, other tests/oracles, PROJECT.md, task record,
lesson entries, or A/B conversations were read. Only the two authorized output
files were edited; logs and Matplotlib configuration used temporary paths.
No delegation, Superpowers, optional memory, commit, push, PR, or full verification
was performed.

Source-free warning and exception rendering was configured before the correction
sanity check and baseline; pytest used `-p no:warnings --tb=no`, and the unchanged
LM child installed source-free rendering before imports. **Source exposure: none
observed.** The ImportError disclosed only a filename, not product source or a
source-bearing traceback. Correction environment: Darwin arm64; Python 3.13.12,
NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1. The measured correction authoring/check
interval was 189 s, from 17:45:45 to 17:48:54 UTC on 2026-10-05, excluding earlier
input reading and final report preparation. The prior 425 s interval and all
attempts remain recorded; these partial measurements total 614 s. Exact token
usage/billing are unavailable. Only B round2 remains in the existing window;
the C allowance and scope are unchanged.

## A formatting correction after the accepted B2 checkpoint

The owner supplied B2 acceptance of all approved scenarios at test SHA-256
`f568ad84499aa7dea8a061bc5935fad09fbf96caee733bd78664cdb907d411c0`.
The later canonical fast format failure was classified as a **test formatting
defect**. This fresh root A session was authorized only to apply Black to the
test file, prove unchanged Python AST, run a permitted revised source-free
baseline, and append this historical evidence. All earlier report text, attempts,
oracle values, tolerances, and spending remain preserved.

Preflight confirmed that the worktree test and the specifically permitted exact
B2-accepted source at `/private/tmp/dphtools-spectrum-delivery/B2-accepted-tests.py`
were byte-identical and both had the accepted hash above. The only test mutation
was this command, run from the supplied worktree:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/SPECTRUM-FITTING-001-A-format-sourcefree /private/tmp/dphtools-spectrum-delivery/venv/bin/python -m black --line-length 99 tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-format-black.log
```

The temporary `sitecustomize.py` in that PYTHONPATH installs the same source-free
warning formatter and exception hook used by the approved launcher, before Black
imports. It changes no repository configuration. Black 26.5.1 reported one file
reformatted; shell exit status was 0. The formatted test SHA-256 is
`4f762faf63839e4301038a91d02abee9b2f6c8748dc671d19a48aec1ffb674d2`.

Both sources were parsed with `ast.parse(..., type_comments=True)` and compared
using `ast.dump(..., annotate_fields=True, include_attributes=False)`. The dumps
are **exactly equal**, including all constants, oracle values, tolerances,
decorators, and the embedded child-process script; location attributes are omitted.
Both AST dumps have SHA-256
`5a7a1039ce49c886f75739f5b7dc4436acc38f55e44a5e4ab3939f9857f90855`
and 6,583 AST nodes under Python 3.13.12. AST proof exit status was 0. This session
did not redesign cases or repeat numeric oracle calibration.

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/SPECTRUM-FITTING-001-A-format-sourcefree /private/tmp/dphtools-spectrum-delivery/venv/bin/python -m black --check --line-length 99 tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-format-black-check.log
```

Black check reported one file would be left unchanged; shell exit status was 0.
**Formatting-defect correction: verified**, high confidence, with exact AST
equality to the supplied accepted source.

The revised baseline reused the approved source-free launcher and exact pytest
arguments recorded above: `tests/test_spectrum_fitting.py -p no:warnings --tb=no
-q -ra --capture=tee-sys`. It used
`/private/tmp/dphtools-spectrum-delivery/venv/bin/python`,
`PYTHONDONTWRITEBYTECODE=1`, `MPLBACKEND=Agg`, and
`MPLCONFIGDIR=/private/tmp/spectrum-fitting-001-a-format-mpl`, with `set -o pipefail`
and output retained at
`/private/tmp/SPECTRUM-FITTING-001-A-baseline-format-correction.log`.
Source-free hooks preceded all probe imports. Added identity diagnostics recorded
the test hash, HEAD, environment, and imported public module filename; the test
hash and HEAD were unchanged after pytest.

Observed baseline HEAD was `8361239543449f1afe2d7464a27790c5467c780b`, obtained
only through `git rev-parse HEAD`; no history or implementation was read. Execution
was in `/Users/davidhoffman/.codex/worktrees/spectrum-fitting/dphtools`, and the
imported public module filename was that worktree's `dphtools/utils/fitfuncs.py`.
HEAD identifies the observed checkout; this blind session makes no claim about
uninspected working-tree product changes. Environment: Darwin arm64, Python
3.13.12, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1, Black 26.5.1.

Result: **1 failed, 262 errors in 12.34 s; pytest and shell exit status 1**.
All 263 cases were collected. The fixture errors arose from the absent public
`spectrum_fit` entry point. The unchanged LM child exited 1 and retained the
same source-free missing-entry-point ImportError shown in the prior baseline.
No numerical or validation product assertion was reached. No warning or skip
was reported; the full log retains all diagnostics, counts, and failure status.
**Baseline classification: product defect — missing R3 public API**, high
confidence. This is evidence for the missing entry point only and does not
authorize product repair by A.

Additional retained evidence outside Git:

| Evidence | Path | Result |
|---|---|---|
| Preflight source hashes and identity | `/private/tmp/SPECTRUM-FITTING-001-A-format-preflight.log` | Accepted/current bytes equal; expected hash confirmed |
| AST proof | `/private/tmp/SPECTRUM-FITTING-001-A-format-ast.log`; `/private/tmp/SPECTRUM-FITTING-001-A-format-ast-proof.json` | Exact equality excluding locations; exit 0 |
| Historical report preimage | `/private/tmp/SPECTRUM-FITTING-001-A-format-report-before.md`; `/private/tmp/SPECTRUM-FITTING-001-A-format-report-preimage.log` | Prior report retained for prefix-preservation verification |

Correction disclosure: read only the session packet/instructions, selected
design-tests skill, approved contract R3, permitted test/report, and exact
B2-accepted public test source. This session also read its own generated logs and
current HEAD identity. No implementation/history, other tests, PROJECT.md, task
state, lessons, or other conversations were read. Only the test and this report
were edited in the tracked tree. No delegation, Superpowers, optional memory,
commit, push, PR, full verification, or budget expansion occurred. **Source
exposure: none observed.** Diagnostics disclosed a public module filename, with
no implementation excerpt or source-bearing traceback. This appendix adds no
competing Current state and grants no test acceptance.

Allowance evidence supplied by the owner: the prior review window spent 2/2 B
rounds and was accepted/closed. This later evidenced, authorized correction opens
a new section9 test-correction window with at most 2 B reviews, 0 used at this
handoff. All prior attempts/spending remain retained; no C allowance is reset.
This A session launched no reviewers or C sessions and performed no C repair.

Metrics: 33 unchanged scenarios / 263 unchanged cases; 1 fresh root A launch
(3 known A launches including the two historical authors); 0 reviewer launches
in this session; prior B window 2/2 accepted/closed, correction window B 0/2;
no C allowance reset. The observed formatting/check interval was 88 s, from
17:57:09 to 17:58:37 UTC on 2026-10-05, excluding final report preparation.
Together with the retained 425 s and 189 s intervals, measured partial spending
totals 702 s. Exact token usage/billing remain unavailable.

## Post-implementation public coverage correction — new review window

This fresh independent root A session was authorized for three evidenced public
coverage gaps after initial C. These additions are **post-implementation
correction evidence**, not original test-first red evidence. The owner approved
R3 and S01-S33 as one slice; no scenario or behavior approval was added or reset.
The packet supplied accepted test SHA-256
`4f762faf63839e4301038a91d02abee9b2f6c8748dc671d19a48aec1ffb674d2`, historical report
SHA-256 `0f9e2b90b7f6d96a016073bc2dd0807c3888a9d257f781c568d097f221d23674`, and
checkpoint `1f9050ccd1d7e6b0ae86fd9eabc3600fcf29a268`. Both input hashes were
confirmed before editing. The earlier original review window was 2/2
accepted/closed; the Black-only correction window was 1/2 accepted/closed.
Their attempts and spending remain recorded above. The coordinator's supplied
narrow public gap descriptions were used; no raw implementation or coverage
logs were read.

### Added-case mapping and independent expectations

Three new public test functions add ten cases, bringing the focused file from
263 to 273 cases while retaining all 33 approved scenarios.

| Existing scenario(s) | Added public test, prefix `test_` | Cases and expected behavior |
|---|---|---|
| S04/S08; S01-S03 numerical definitions | `s04_s08_center_guess_broad_component_truncated_half_height` | 6: each family, left/right truncation; center-only guesses fit one broad component with no background, recovering physical height, center, width(s), model and residuals despite an absent half-height crossing |
| S31; S01-S03/S08 model and R3 covariance boundary | `s31_coincident_components_warn_without_finite_individual_height_uncertainties` | 3: each family; public physical custom optimizer returns an exact minimizer for two coincident equal-width components; covariance warning remains observable and both individual height variances are nonfinite |
| S29; S03 normalized Voigt and finite-domain definitions | `s29_finite_subnormal_voigt_numerical_boundary_preserves_inputs` | 1: finite strictly increasing coordinates, finite data and strictly positive subnormal initial widths; numerical inability raises RuntimeError, or a robust solver returns an independently verified finite exact-model fit; either outcome preserves caller inputs |

**Truncated broad component.** The physical row is `(3.4, 7.3, 2.5)`, or
`(3.4, 7.3, 2.5, 1.25)` for Voigt. Height has data units; center and widths have
x units. The 241 coordinates are `center + sigma_or_gamma * offsets`, with
offsets `[-0.4, 3]` for left truncation and `[-3, 0.4]` for right truncation.
The boundary on the truncated side lies above half height, while the opposite
tail lies below half height. Continuity and monotonicity of these isolated
profiles place one half-height crossing outside the observation interval.
For example, the Gaussian half-height offset is
`sqrt(2*log(2))*2.5 = 2.9435` x units; the truncated boundary is only `1.0`
x unit from the center. Lorentzian half height is at offset `2.5` x units.
The endpoint ordinate assertions check this property for every family.

An independent source-free fixture check, using the existing five-point oracle,
found column-normalized physical Jacobian condition numbers `4.60255` for
Gaussian, `3.21364` for Lorentzian, and `19.60241` for Voigt, on either side.
Thus the samples retain width and height information; no initialization
algorithm or internal estimate is specified. Each result uses the unchanged
parameter tolerances `rtol=3e-5, atol=2e-5` in the parameter's own units,
model/residual absolute tolerance `2e-7` data units, and existing reconstruction
and residual consistency checks. Wrong height units, a stuck fallback width,
a shifted center, dropped components or an added background coefficient cannot
satisfy the combined physical-parameter, shape and complete-model checks.

**Coincident component uncertainty.** The two identical rows are
`(2, 0.25, 0.9)`, or `(2, 0.25, 0.9, 0.4)` for Voigt, on 161 coordinates
from `-4` to `4` x units. With a common unit-height profile P,

`m(x) = a1*P(x-c) + a2*P(x-c) = (a1+a2)*P(x-c)`.

For any `t` retaining positive heights, `(a1+t, a2-t)` yields the same model.
The individual-height Jacobian columns are identical; the null direction is
`(1, 0, ..., -1, 0, ...)`. No finite individual-height uncertainty follows from
this rank-deficient local linear approximation, even when residual sum of
squares is zero. The independent fixture check verified the unchanged model
at height split `(2.5, 1.5)` and exactly equal numerical height columns.
The public custom callable returns the known physical exact minimizer, checks
its supplied residual against zero with absolute `2e-12` data units, and returns
no Jacobian/covariance. This removes optimizer convergence as a confounder.

The result must preserve the correct peak rows, empty background, finite model
and residuals, real full covariance shape, and an observable warning. Captured
warnings are explicitly rendered, including category, message, filename and
line, even if the call fails. Both individual-height diagonal entries must be
nonfinite; either infinity or NaN is allowed. No particular warning category,
wording, covariance algorithm, or encoding is required, and no assertion
requires nonfinite uncertainty for an identifiable combined quantity. The
unchanged identifiable-fit covariance helper is deliberately not used here.
The [contract-linked SciPy covariance reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.curve_fit.html)
was opened; R3's prohibition on fabricated finite uncertainties and the exact
height-exchange invariant determine this expectation, rather than a particular
SciPy covariance implementation. Finite fabricated height variances, suppressed
warnings, a wrong model, or dropping a component fail independently.

**Finite numerical boundary.** Let `s = finfo(float).tiny/1024`, observed as
`2.1729236899484e-311` x units, and take `x = s*linspace(-4,4,81)`.
The [NumPy floating-point limits definition](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
was opened to establish the positive subnormal domain. The approved
[SciPy Voigt definition](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.voigt_profile.html)
was also opened. Its normalized convolution has the scale property

`V(s*d; s*sigma, s*gamma) = V(d; sigma, gamma)/s`.

Consequently the unit-height ratio is unchanged by s. Finite exact data are
formed in order-one coordinates as `3*V(x/s;1,1)/V(0;1,1)`; the physical full
initial row is the exact generating minimizer `(3,0,s,s)`. The test checks
finite inputs, positive widths and strictly increasing coordinates explicitly.
The independent scaled fixture has normalized Jacobian condition number
`11.07356`. Its center sample is exactly `3` data units. Direct public SciPy
normalization at physical s returned infinite center density and 81 nonfinite
ratios, with an unsuppressed RuntimeWarning. This is evidence of a real floating
point limitation, not an invalid input or a patched production optimizer.

The outcome is intentionally conditional. A numerical fit failure must be
RuntimeError; ValueError, other leaked exceptions or mutated caller inputs fail.
A robust finite solution is allowed, so the test does not force solver failure.
Success must return one real finite positive Voigt row, empty background,
correctly shaped real covariance, finite fitted samples/residuals, and the
verified generating model. Rescale returned center/widths by s before comparing
to `(3,0,1,1)` with the unchanged `rtol=3e-5, atol=2e-5`; this prevents the
ordinary absolute width tolerance from accepting zero or arbitrary subnormal
widths. Independent scale-normalized reconstruction uses `rtol=2e-10,
atol=2e-10`; fitted data and zero residuals use `atol=2e-7` data units, and
`data-fitted` uses the existing `rtol=2e-12, atol=2e-12`. Physical covariance
may itself encounter numerical limits, so no finite-covariance assertion is
added for this outcome. The starting row is already an exact minimizer; the
success alternative imposes no recovery requirement from a distant local basin.
All tolerance choices and both allowed outcomes preceded product execution.
The [contract-linked optimizer reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html)
was opened for the public objective/status definitions; R3 determines the
public failure exception type.

### Focused execution, preservation and classification

All commands ran from the supplied worktree, using
`/private/tmp/dphtools-spectrum-delivery/venv/bin/python`, with
`PYTHONDONTWRITEBYTECODE=1`, `MPLBACKEND=Agg` and writable
`MPLCONFIGDIR=/private/tmp/dphtools-spectrum-delivery/mplconfig`. No environment
or repository configuration was changed. The known Anaconda-base tooling
repair remained outside this session.

Retained evidence directory, outside Git:
`/private/tmp/SPECTRUM-FITTING-001-A-coverage-20261005/`.
The authored `launch.py` sets `warnings.formatwarning` to category/message/file/
line only and sets the uncaught-exception hook to exception-only rendering
before importing pytest. Every fixture probe and product test uses
`-p no:warnings --tb=no -q -ra --capture=tee-sys`. The launcher's explicit exit
status and `set -o pipefail` retain failures. The existing LM child independently
sets the same source-free hooks before its imports.

Exact focused full-spectrum command:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-delivery/mplconfig /private/tmp/dphtools-spectrum-delivery/venv/bin/python /private/tmp/SPECTRUM-FITTING-001-A-coverage-20261005/launch.py tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-coverage-20261005/spectrum-tests.log
```

The authored `format.py` installs the same source-free hooks before importing
Black and invokes Black's public command entry point. Formatting was applied
once to the new additions. Exact final format-check command:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-delivery/mplconfig /private/tmp/dphtools-spectrum-delivery/venv/bin/python /private/tmp/SPECTRUM-FITTING-001-A-coverage-20261005/format.py --check --line-length 99 tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-coverage-20261005/black-check.log
```

| Check/evidence | Result | Retained file in evidence directory |
|---|---|---|
| Independent fixture/domain checks | 10 passed in 0.87 s; exit 0; no product import | `fixture-checks.log`; `test_fixture_checks.py` |
| Focused full spectrum | 273 passed in 2.21 s; pytest/shell exit 0; no failures, errors or skips | `spectrum-tests.log`; `launch.py` |
| Numerical boundary outcome | RuntimeError; RuntimeWarning remained visible; input preservation passed | `spectrum-tests.log` |
| Coincident-component outcomes | All three returned warnings and nonfinite individual-height variances | `spectrum-tests.log` |
| Black application/check | One file formatted; final check leaves one file unchanged; both exit 0 | `black-format.log`; `black-check.log`; `format.py` |
| Original-test preservation | All original module AST nodes remain exactly unchanged excluding locations; only a warnings import and three functions were added; exit 0 | `preservation-environment.log`; `tests-before.py` |
| Historical report preservation | Original report bytes retained as an exact prefix | `report-before.md`; `final-identity.log` |

The spectrum run retained one RuntimeWarning from the finite numerical boundary
and three OptimizeWarnings from covariance estimation. The standalone fixture
check retained its own RuntimeWarning for direct public SciPy normalization.
Diagnostics were not suppressed, and no exact message assertion was used.
Execution environment: Darwin arm64; Python 3.13.12, NumPy 2.5.3, SciPy 1.18.1,
pytest 9.1.1, Black 26.5.1. The final test SHA-256 is
`1ca6c122fa10d628af4e4c00f6c19b3aef948bbff38bcabec147397061bac764`.
The report's own final hash is returned outside this tracked tree.

**Classification: authorized post-implementation test coverage correction;
focused public checks pass**, high confidence. The observed numerical failure
is contract-permitted RuntimeError, not an input-validation or product defect.
The rank-deficient warnings and nonfinite individual-height variances satisfy
the approved uncertainty boundary. No unresolved requirement was found. A does
not accept its own tests. Fresh independent B round1 review is the next handoff
in this new window. No canonical full coverage measurement or platform matrix
was run, and this report makes no coverage-completion or merge-readiness claim.

Read only root AGENTS.md, the narrow owner packet, R3, selected design-tests
skill, setup.cfg conventions, the permitted current test/historical report,
the cited primary-reference pages, and this session's own source-free evidence.
No product implementation, private-helper source, history/diff, other tests/
oracles, PROJECT.md, task state, lesson entries or role conversations were
read. Only the two authorized tracked paths were edited; temporary launchers,
preimages and logs reside outside Git. No Superpowers, optional memory,
delegation, repository rule/config edit, commit, push, PR, broad full
verification, merge or release occurred. **Source exposure: none observed.**
Warnings disclosed product filenames/line numbers as required, but no source
excerpt or source-bearing traceback was rendered.

Metrics: 33 unchanged scenarios / 273 cases / 10 added cases; 1 fresh root A
launch (4 known A launches including preserved historical sessions), 0 reviewer
or C launches in this session; original B window 2/2 accepted/closed, Black-only
window 1/2 accepted/closed, new correction B window 0/2 used with round1 next;
initial C used, C-repair counter 0/1 unchanged. Measured fixture-authoring/check
interval: 136 s, 18:31:42–18:33:58 UTC on 2026-10-05, excluding earlier permitted
input/reference reading and final report preparation. Added to the retained
702 s of earlier partial measurements, partial measured spending totals 838 s.
No fixed owner time/token cap was supplied; exact token usage and billing are
unavailable. Earlier attempts, spending and allowances were not reset. This
appendix is historical evidence and creates no competing Current state.

## Post-implementation smallest-positive-width regression — new review window

Fresh independent root A was authorized to add only the evidenced Gaussian and
Lorentzian numerical-boundary regression. This is **post-implementation
product-red evidence**, not original test-first evidence. The supplied accepted
checkpoint is `9e3962dcfaa2fc60beb67803446e642b5b87ff17`, with test SHA-256
`1ca6c122fa10d628af4e4c00f6c19b3aef948bbff38bcabec147397061bac764` and historical
report SHA-256 `a6dd1d690176ddc159dbd2479f8a024fcd4c86eda78602dfaf2bfc53f341aec3`.
Both input hashes were confirmed before editing. R3 and all 33 scenarios remain
approved; earlier windows and spending remain retained. This report provides
no independent acceptance or competing Current state.

### Mapping, fixture legitimacy and allowed outcomes

One parametrized public test adds two cases, bringing the spectrum file to 275
cases. The only test additions are `import math`, the independent stable helper,
and the new test; every preexisting test, comment, oracle and tolerance remains
byte-for-byte unchanged after removing those additions.

| Existing scenarios | Added public test, prefix `test_` | Expectations |
|---|---|---|
| S29; S01/S02 unit-height definitions, S08 no background, S17/S19 finite positive domain, S30 default LM | `s29_smallest_positive_width_point_samples_preserve_inputs` | Two cases: `gauss` and `lorentz`; numerical failure must raise RuntimeError and preserve inputs, or success must independently reproduce the sampled model with valid public outputs and applicable uncertainty warnings |

Both calls use precisely `x=linspace(-1,1,41)`, zero data except `y[20]=2`,
full list guesses `[[2.0,0.0,nextafter(0.0,1.0)]]`, `background="none"`, and an
omitted optimizer. Height is in data units; center, width and spacing are in x
units. Coordinates, samples and guesses are finite; amplitude and width are
positive; `N=41>P=3`, giving 38 residual degrees of freedom. The float64 width
is `2**-1074`, approximately `4.9406564584124654e-324` x units. NumPy distinguishes
this positive subnormal from the smallest normal value. Its square rounds to
zero; that arithmetic limitation does not invalidate the width. The
[NumPy floating-point limits reference](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
was opened independently.

R3's point-observation convention and elementary unit-height equations establish
the fixture. At the center, both profiles equal one, so the modeled sample is
2. At any other sample, the distance is at least approximately 0.05 x units.
For Lorentzian, `L(d) <= (width/abs(d))**2`, less than approximately `1e-644`;
for Gaussian, `G(d)=exp(-0.5*(d/width)**2)` is still smaller. Both off-center
models round to zero in float64. This is a component narrower than the sampling
distance, not a bin-integrated component or an identifiable-width fixture.
The independent checks reproduced the complete supplied data exactly for both
families and found sampled sensitivity rank one at the starting row.

The test calls only the real public `spectrum_fit` through the unchanged
input-preservation helper. RuntimeError is allowed without requiring specific
wording or a warning. Any other exception fails, including LinAlgError/ValueError.
Input preservation is checked in `finally`, including failed calls. A successful
return must have one finite real positive physical row, an empty real background,
a real 3-by-3 covariance, and finite real fitted/residual arrays of length 41.
Missing attributes, None, incorrect shapes, nonpositive parameters, inconsistent
or fabricated fitted arrays, and inconsistent residuals fail. No mock, custom
optimizer, production seam, optimizer-failure mandate, or comparison of returned
width/center to the initial row is introduced.

### Stable reconstruction, tolerances and uncertainty

The new helper evaluates the approved equations independently without squaring
the physical width. Gaussian uses `exp(log(amplitude)-z**2/2)` with `z=d/width`;
division is bounded before evaluating z. Beyond `abs(z)=64`, even the largest
finite float64 amplitude times the profile rounds to zero, since
`log(max_float)-64**2/2 < -1338 < log(smallest_subnormal)`, approximately -744.
This bound applies only to oracle evaluation, not to accepted physical widths.
Lorentzian uses `scale=max(abs(d),width)`, `t=d/scale`, `u=width/scale`, and
`exp(log(amplitude)+2*(log(width)-log(scale))-log(t*t+u*u))`.
The scaled denominator is at least one. Log evaluation also retains a finite
height-times-profile product when the separate profile would underflow.

All tolerances were fixed before public product execution. Existing reconstruction
`rtol=2e-10, atol=2e-10` and residual-consistency `rtol=2e-12, atol=2e-12` are
reused. Complete fitted data and zero residuals use existing absolute `2e-7`
data units, allowing solver stopping far above float64 roundoff while rejecting
loss of the height-2 sample or spurious tails. There is no physical-parameter
recovery tolerance for this unidentifiable starting fixture. Covariance symmetry
reuses `rtol=2e-10, atol=2e-12`, allowing symmetric NaN/infinity encodings;
finite diagonal variances must be nonnegative.

Applicable uncertainty checks use the **returned** row. The helper also evaluates
analytic physical Jacobian columns multiplied by `(amplitude,width,width)`,
then normalizes each nonzero column by its maximum magnitude. These positive
column scalings preserve rank and individual-parameter estimability. For Gaussian
the scaled columns are `(m,m*z,m*z*z)`. For Lorentzian they are
`(m,2*m*t*u/(t*t+u*u),2*m*t*t/(t*t+u*u))`. Rank uses NumPy's documented singular
value decomposition (SVD) threshold `largest_singular_value*max(shape)*eps`,
approximately `9.10e-15*largest_singular_value` here. The
[NumPy rank reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)
was opened. A parameter is locally unidentifiable when removing its column
leaves rank unchanged: the remaining columns can compensate its local change.
At a deficient returned solution, those diagonal variances must be nonfinite
and an observable warning must exist. Any nonfinite covariance also requires
a warning. A wider full-rank returned solution is allowed. No width is declared
identifiable merely because it is positive, and no singular result is required.

These conditions follow R3's local-linear uncertainty convention and prohibition
on fabricated finite uncertainties. The opened
[contract covariance reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.curve_fit.html)
documents covariance's local-linear limits and warnings; it does not specify
the product's internal covariance algorithm. Seven independent source-free
checks, with no product import, verified the environment, exact boundary samples,
ordinary profile values, analytic scaled derivatives against independent
five-point differences, full rank at an ordinary row, and finite products when
a profile alone underflows. Derivative sanity tolerances reuse `rtol=2e-6,
atol=2e-8`; ordinary-model sanity uses `2e-14` relative/absolute tolerance.

### Execution, preservation and failure classification

Evidence resides outside Git at
`/private/tmp/SPECTRUM-FITTING-001-A-boundary-lwd67_xk/`.
Every Python launch installs source-free warning and exception hooks before
probe/test imports. `launch.py` supplies `-p no:warnings --tb=no -q -ra
--capture=tee-sys`, prints the pytest exit status, and propagates it. The temporary
`sitecustomize.py` also supplies the hooks to child interpreters; the unchanged
LM child retains its own hooks. Warnings retain category, message, filename and
line. Captured regression warnings are rendered even when the call fails.

Exact final full-spectrum command from the supplied worktree:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/SPECTRUM-FITTING-001-A-boundary-lwd67_xk MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/mplconfig PIP_CACHE_DIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/pip-cache /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3 /private/tmp/SPECTRUM-FITTING-001-A-boundary-lwd67_xk/launch.py tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-boundary-lwd67_xk/spectrum-tests-final.log
```

The focused command adds `-k smallest_positive_width` and uses
`regressions-final.log`. Fixture checks replace the test argument with the
evidence directory's `test_fixture_checks.py` and use `fixture-checks.log`.
Black uses the same environment with `format.py --check --line-length 99
tests/test_spectrum_fitting.py` and `black-check-final.log`.

| Check | Result | Evidence file |
|---|---|---|
| Independent fixture/oracle/environment | 7 passed in 0.12 s; exit 0; no product import | `fixture-checks.log`; `test_fixture_checks.py` |
| Final focused regressions | 2 failed, 273 deselected in 0.80 s; pytest/shell exit 1 | `regressions-final.log` |
| Final existing spectrum suite plus additions | 2 failed, 273 passed in 1.33 s; no errors/skips; pytest/shell exit 1 | `spectrum-tests-final.log` |
| Black 99 | One file would be left unchanged; exit 0 | `black-check-final.log` |
| Exact old-test bytes and abstract syntax tree (AST) preservation | Removing only the three additions reconstructs the exact preimage; every old AST node is unchanged | `preservation-final.py`; `final-identity.json` |
| Historical report preservation | All 47,184 preimage bytes remain an exact prefix | `report-before.md`; `final-identity.json` |

Both new public calls emitted numerical RuntimeWarnings and leaked
`LinAlgError: SVD did not converge`; the unchanged input checks completed
successfully before the exception was re-raised. Independent runtime inspection
of the public NumPy exception type confirmed it subclasses ValueError. Each
regression retained ten overflow warnings and one invalid-value warning. The
full suite also retained the prior Voigt numerical warning and three covariance
OptimizeWarnings; the real LM child exited 0. No failure, warning, count or
exit status was hidden. The seven fixture checks and Black emitted no warning.

**Classification: product defect — leaked numerical exception for valid public
inputs**, high confidence, subject to fresh independent B validation of the
tests. R3 permits verified success or RuntimeError for numerical failure; it
does not permit an input-validation exception here. The tests reached that
exact boundary, so these are valid post-implementation product-red observations.
No unresolved requirement or environment defect was observed. The opened
[R3 optimizer reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html)
supports the residual objective; R3 determines the public exception type.
Fresh B round1 review is required before C repair. A grants no acceptance and
performs no repair.

The initial focused/full runs returned the same two failures with 273
deselected/passing cases in 0.85/1.34 s, respectively; their logs remain
`regressions.log` and `spectrum-tests.log`. Initial Black also passed.
After clarifying only the new helper's docstring to describe physical column
scaling precisely, focused/full/Black checks were rerun on the final test bytes.
No numerical oracle or tolerance was calibrated from product results.
The final test SHA-256 is
`8bcfb0a747c6572403d57122dd16e60e08668472b68879014d95aa1c94889b09`.
The old 7,806-node AST has SHA-256
`df2f48ffe10ada93cb0b3a2fd9d81d2e6341c0795b9916bd4a2bc8efdc5de6f5`.
The report's final hash is returned outside its tracked tree.

Environment: macOS 27.0 arm64, confirmed non-Conda CPython 3.12.14 at the supplied
path; NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1, Black 26.5.1. Interpreter prefix
and base prefix both point to the clean supplied Python. These tests exercised
the working-tree public entry point; warning filenames identify the supplied
worktree. The supplied checkpoint is prior acceptance identity, not a claim
about uninspected product working-tree bytes.

Read only root AGENTS.md, the owner packet, R3, design-tests skill, setup.cfg,
the permitted spectrum test/report, cited primary-reference pages, and this
session's own source-free evidence. No implementation/private-helper source,
history/diff, other tests/oracles, PROJECT.md, task state, lessons, role
transcripts or raw implementation logs were read. Only the two permitted tracked
paths were edited. No Superpowers, memory, delegation, commit/push/PR, broad full
verification, merge or release occurred. **Source exposure: none observed.**
Required diagnostics revealed filenames/line numbers but no source excerpt or
source-bearing traceback. Full coverage and platform-matrix verification were
not run or claimed.

Metrics: 33 unchanged scenarios / 275 cases / 2 added cases; 1 fresh root A
launch (5 known A launches), 0 reviewer/C launches in this session; original B
2/2, format B 1/2, coverage B 1/2 all accepted/closed; new regression window
B 0/2 used, round1 next. Initial C used; C repairs remain 0/1 used. Partial
measured authoring/check interval: 192 s, 18:51:37–18:54:49 UTC on 2026-10-05,
excluding earlier permitted reading and final report preparation. Retained prior
838 s plus this interval totals 1,030 s of partial measured spending. No fixed
owner time/token cap was supplied; exact token usage/billing are unavailable.
No scenario approval, review allowance or C repair allowance was reset.


## Supplementary physical-covariance unit regression — before the same B round1

This fresh independent root A was authorized to supplement the still-unreviewed
A5 numerical-regression correction before one combined B round1. This is
**post-implementation regression evidence**, not original test-first evidence.
R3 and all 33 scenarios remain approved. The supplied accepted checkpoint remains
`9e3962dcfaa2fc60beb67803446e642b5b87ff17` (273 cases). The prior unreviewed A5
preimages were confirmed exactly before editing: test SHA-256
`8bcfb0a747c6572403d57122dd16e60e08668472b68879014d95aa1c94889b09`; report SHA-256
`dd4459a66b1cd9cf6c8f53556da98033c4a0d6c1753eaeada283367ef632efa2`.
All 275 authored cases and all historical report bytes are preserved. This
supplement neither opens another window nor resets scenarios, reviews, spending,
or the C-repair allowance. It grants no independent acceptance or Current state.

### Mapping and analytic expectation

One new public test adds one collected case, calling the real fitter under two
x-unit systems; the file now contains 276 cases.

| Existing scenarios | Added public test, prefix `test_` | Expected behavior |
|---|---|---|
| S09/S31 | `s09_s31_gaussian_physical_covariance_transforms_with_x_units` | One identifiable Gaussian and no background; a legitimate physical custom optimizer returns the independently established local minimizer without a Jacobian/covariance; fitted samples and residuals agree under both units, and the fitter's physical covariance matches the analytic expectation and transforms consistently |

For `m=A*exp(-z**2/2)`, `z=(x-c)/sigma`, the physical model-Jacobian columns are

`J_A=exp(-z**2/2)`, `J_c=m*z/sigma`, `J_sigma=m*z**2/sigma`.

The residual Jacobian is `-J`; this sign cancels in its Gram matrix. R3 gives

`v=RSS/(N-P)`; `C=v*inv(J.T@J)`.

The opened [contract-linked SciPy covariance reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.curve_fit.html)
supports the local-linear approximation and residual-degrees-of-freedom scale.
R3 additionally requires the custom path's Jacobian and covariance in physical
coordinates. These contract terms determine the expectation; no unresolved
units, estimator, or requirement was found.

Changing x units by a positive factor `s` maps `(A,c,sigma)` to
`(A,s*c,s*sigma)`. With `D=diag(1,s,s)`, the model and residuals are unchanged,
`J'=J*inv(D)`, and therefore

`C'=v*inv(inv(D)*(J.T@J)*inv(D))=D*C*D`.

Height variance has squared data units; center/width variances have squared x
units. Their covariance with height has data-times-x units. Every physical
covariance entry is converted back by division by `outer((1,s,s),(1,s,s))`.
The test compares that matrix with the analytic base-unit C, and finally compares
the two converted returned matrices. It specifies neither a finite-difference
step nor a decomposition or internal positivity transform for production.

### Fixture legitimacy and independent precision checks

The fixed fixture is `x=linspace(-4,4,201)`, one Gaussian row `(3,0,1)`, and no
background. All inputs are finite, coordinates strictly increase, and the
positive height/width are interior to the public domain. `N=201`, `P=3`, and
`N-P=198`. There are 25 sampling intervals per sigma in either unit system.
This is an ordinary identifiable component, not a narrow unresolved peak.
The second scale is the ordinary finite `s=1e-6`; its sigma is `1e-6` x units
and its sampling interval is `4e-8` x units.

Small deterministic high-frequency noise is
`u_i=0.01*(sin(1.63*i)+0.6*cos(0.71*i))` in data units. With Q an orthonormal
basis for the analytic Jacobian columns, set `r=(I-Q*Q.T)*u` and `y=m+r`.
Projection gives `J.T*r=0` algebraically. The test uses the represented `y-m`,
including addition roundoff, both to prove stationarity and to scale covariance.
Observed RSS is `0.013791440787719707` squared data units, and the maximum noise
magnitude is `0.015960383757952064` data units. Nonzero noise makes covariance
scale meaningful and discriminating. The normalized gradient projection is
`5.932776029801998e-16`, below the independently fixed `1e-12` bound.

For half the residual sum of squares, the objective Hessian is
`H=J.T@J-K`, `K=sum_i(r_i*Hessian(m_i))`. At `(3,0,1)`, analytic second derivatives
in `(A,c,sigma)` order are

`m_AA=0`, `m_Ac=profile*x`, `m_A_sigma=profile*x**2`,
`m_cc=m*(x**2-1)`, `m_c_sigma=m*(x**3-2*x)`,
`m_sigma_sigma=m*(x**4-3*x**2)`.

Symmetry supplies the other entries. The test requires
`norm(K,2)<0.01*lambda_min(J.T@J)`. By the eigenvalue perturbation bound, H is
positive definite. The observed ratio is `6.750085189827774e-5`; Gram eigenvalues
are approximately `(28.014426,199.400969,315.396977)` and H eigenvalues are
`(28.014499,199.400969,315.398190)`. Thus the stationary physical row is a strict
local least-squares minimizer to floating-point precision. The scaled objective
Hessian is related by congruence with `inv(D)` and remains positive definite.
R3 does not require a global optimum. The column-normalized Jacobian condition
number is `1.931855195700513`, below the fixed bound 3 in both systems; the
unscaled base Jacobian condition number is `3.3553505359410005`.

Five separate temporary fixture checks ran without importing product code.
They verified the clean interpreter, gradient/curvature proof, analytic first
and second derivatives, exact covariance transform, attainable numerical
precision, and tolerance discrimination. Complex-step derivatives of the
independent elementary Gaussian agree with the analytic columns at relative/
absolute `2e-14`. Central second differences with base-unit step `1e-4` agree
with all six second derivatives at relative/absolute `2e-6`; their weighted
sum agrees with K within absolute `2e-15`. The exact unit-transform oracle agrees
within relative/absolute `2e-14`. A dimensionless three-point derivative sanity
check at both scales gives normalized covariance agreement within `2e-8`.
These are independent feasibility checks, not prescribed production algorithms.

The custom callable checks its physical initial vector, bounds, limit, and
supplied residual at the known solution and a nearby physical trial. It makes
two residual calls under `max_nfev=10`, then returns only a copied physical x
and Boolean success. It supplies no Jacobian or covariance. This is a legitimate
minimizing callable under the opened [R3 optimizer reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html)
and public protocol; it removes iterative convergence as a covariance confounder.
No production seam or supplied production result is introduced.

### Tolerances, discrimination and allowed outcome

All fixture/oracle/tolerance choices preceded the first product call. No tolerance
was calibrated to the owner-supplied bad-output description or observed output.
The existing generic result/schema and input-preservation helpers are reused.
Since the custom callable returns the known row, parameter checks after unit
conversion, exact fitted samples, and nonzero residuals use `rtol=2e-12,
atol=2e-12` in the respective base parameter/data units. These are arithmetic
checks at identical physical parameters, not iterative convergence tolerances.

Every covariance comparison divides by `sqrt(C_expected[ii]*C_expected[jj])`.
It reuses the existing `rtol=0.01, atol=2e-4` in dimensionless uncertainty units.
For each variance this allows about 1.02% error; nearly zero cross terms have a
small absolute allowance in their natural uncertainty units. This exceeds the
independently demonstrated derivative/oracle roundoff by many orders of magnitude
while retaining physical variance and height-width correlation discrimination.
For this fixture, using RSS/N instead of RSS/(N-P) changes each variance by
`3/201=1.492537%`, beyond that allowance. Independent synthetic wrong expectations
also fail: zero covariance, missing residual scaling, ignored x units, a private
log-amplitude covariance, and deleted height-width correlation. These examples
come from analytic expectation errors, not production output.

The valid outcome is a successful public result with the verified physical row,
model, residuals and finite identifiable covariance satisfying both analytic and
unit-transform comparisons. No numerical failure or nonfinite covariance is
required or accepted as a substitute here. The simple interior minimizer is
already supplied; unlike the retained A5 extreme cases, this is not a failure-
handling or nonidentifiability fixture. No warning category/message or particular
covariance method is imposed.

### Source-free execution, preservation and classification

Evidence is retained outside Git at
`/private/tmp/SPECTRUM-FITTING-001-A-covariance-vgcernwb/`.
Every Python probe/check installs source-free warnings and exception rendering
before probe imports. Temporary `sitecustomize.py` also supplies those hooks to
child interpreters. Warnings show only category/message/filename/line; uncaught
exceptions show only type/message. The pytest launcher uses `-p no:warnings
--tb=no -q -ra --capture=tee-sys`, prints and propagates the exit status, and its
diagnostic plugin emits only failure exception/assertion messages, without
traceback/source excerpts. The unchanged LM child retains its own hooks.

Exact full-spectrum command from the permitted worktree:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/SPECTRUM-FITTING-001-A-covariance-vgcernwb MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/mplconfig PIP_CACHE_DIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/pip-cache /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3 /private/tmp/SPECTRUM-FITTING-001-A-covariance-vgcernwb/launch.py tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-covariance-vgcernwb/spectrum-tests.log
```

The targeted command adds `-k physical_covariance_transforms_with_x_units` and
uses `covariance-regression.log`. Independent checks substitute the evidence
file `test_fixture_checks.py` for the spectrum file and use `fixture-checks.log`.
Black uses the same environment and interpreter with `format.py --check
--line-length 99 tests/test_spectrum_fitting.py`, retaining `black-check.log`.
Preservation uses the same environment with `preservation.py` and retains
`preservation-final.log` plus `final-identity.json`.

| Check | Observed result | Evidence file(s) |
|---|---|---|
| Independent fixture/oracle/environment | 5 passed in 0.13 s; pytest/shell exit 0; no product import | `fixture-checks.log`; `test_fixture_checks.py`; `analytic-preflight.json` |
| New targeted covariance regression | 1 failed, 275 deselected in 0.83 s; pytest/shell exit 1; no warnings | `covariance-regression.log` |
| Full spectrum file | 3 failed, 273 passed in 1.79 s; no errors/skips; pytest/shell exit 1 | `spectrum-tests.log` |
| Black 99 application/check | File already formatted; final check leaves one file unchanged; both exit 0 | `black-format.log`; `black-check.log` |
| Exact original bytes and abstract syntax tree (AST) | All 41,547 preimage test bytes are an exact prefix; removing only the appended function restores all 275 cases and every prior AST node | `tests-before.py`; `preservation.py`; `final-identity.json` |
| Historical report | All 60,731 preimage bytes are an exact prefix | `report-before.md`; `preservation.py`; `final-identity.json` |

Both unit systems passed real returned parameter/model/residual checks, custom
protocol checks and caller input preservation. Base-unit analytic covariance
passed. At `s=1e-6`, five of nine covariance entries failed the independently
fixed analytic comparison after conversion back to base units. Normalized
height variance was `1.2122417284` instead of 1; center variance was
`18380.1989992` instead of 1; width variance was `1.3814470827` instead of 1;
height-width correlation was approximately `-0.86815035` instead of
`-0.57735149`. The intended covariance assertion was reached. The final direct
unit-to-unit assertion was not reached because this analytic comparison failed.
These numbers are diagnostic observations only; they supplied no expected value.

**Classification: product defect — physical covariance changes inconsistently
with x units**, high confidence, subject to fresh B validation. This supplies
valid supplementary S09/S31 product-red evidence within the same unreviewed
window. There is no observed unresolved requirement, environment defect or
fixture defect. A performs no product repair and grants no acceptance.

The full spectrum file retains both A5 failures: valid smallest-positive-width
Gaussian and Lorentzian calls leak `LinAlgError: SVD did not converge` and retain
ten overflow warnings plus one invalid-value warning each. The 273 prior
accepted cases still pass; their earlier Voigt RuntimeWarning and three
covariance OptimizeWarnings remain visible, and the real LM child exits 0.
No diagnostic, warning, failure count or exit status was suppressed. Fixture and
Black checks emit no warning. No broad full coverage/platform check was run.

Final test SHA-256:
`c5e51bab70912a674d7242541591d1844939dca981c5b78cbce9de70617accca`.
The preserved 8,881-node A5 AST has SHA-256
`3091be61d7a8f8000178785c63c7129921d82360bce526cfd025ee3a13ecd4c5`
under CPython 3.12.14, excluding location attributes. Exactly one new top-level
function is appended; no import, original AST node, oracle, tolerance or case
was changed. The report's final hash is returned outside its tracked tree.

Environment: macOS 27.0 arm64, non-Conda CPython 3.12.14 at the supplied clean
path; interpreter/base prefixes point to that clean Python. NumPy 2.5.3,
SciPy 1.18.1, pytest 9.1.1, Black 26.5.1. The public imports execute the current
worktree product, without source inspection. The supplied checkpoint identifies
prior test acceptance; this session makes no identity claim about uninspected
product working-tree changes.

Read only root AGENTS.md, the owner packet, contract R3, selected design-tests
skill, setup.cfg conventions, permitted current spectrum test/historical report,
the two cited primary references, and this session's own source-free evidence.
No product source/private helpers/history/diffs, other tests/oracles, PROJECT.md,
task state, lessons, role conversations or raw implementation logs were read.
Only the two authorized tracked files were edited; all supporting evidence lives
in the temporary directory. No Superpowers, memory, delegation, commit/push/PR,
broad full verification, merge or release occurred. **Source exposure: none
observed.** Filename/line diagnostics were retained; no implementation excerpt
or source-bearing traceback was rendered.

Metrics: 33 unchanged scenarios / 276 cases / 1 supplementary case; 1 fresh root
A launch (6 known A launches), 0 reviewer/C launches in this session. Original
B 2/2, format B 1/2 and coverage B 1/2 remain accepted/closed. The current
numerical-regression window remains B 0/2, round1 next for combined A5 plus this
supplement; C repairs remain 0/1 used. Partial measured authoring/check interval:
217 s, 19:00:37–19:04:14 UTC on 2026-10-05, excluding earlier permitted
input/reference reading and final report/preservation preparation. Retained
1,030 s plus this interval totals 1247 s of partial measured spending. No fixed
owner time/token cap was supplied; exact token usage/billing are unavailable.
No approval, review allowance, or C allowance was reset. One fresh independent
B review is the next handoff before any C repair.


## Post-repair numerical-failure coverage correction — new B window

This fresh independent root A was authorized for the three narrow public needs
in the owner packet. R3 and all 33 scenarios remain approved. This is
**post-repair test-correction evidence**, not original test-first evidence.
The owner supplied accepted checkpoint
`f70661a9ed1813312e5bc409fae8a288c8c3d025`, test SHA-256
`c5e51bab70912a674d7242541591d1844939dca981c5b78cbce9de70617accca`, and report
SHA-256 `2bd7819d6c6112d2a6660fd03cd5f71685613565f11081fa8b340f3647a8ef3c`.
Both preimage hashes were confirmed before editing. The 276 old cases and all
76,378 historical report bytes remain preserved. No implementation, private
coverage evidence, or other role conversation supplied the new expectations.
This appendix adds no competing Current state and grants no acceptance.

### Mapping and independently allowed outcomes

Three appended public test functions add five cases, for 281 total. All map to
existing S29/S31, using the R3 Gaussian, no-background, finite-domain and
physical-covariance conventions. No new scenario or behavior was added.

| Added test, prefix `test_` | Cases | Independent expectation |
|---|---:|---|
| `s29_s31_extreme_finite_height_exact_custom_minimum` | 2 | Heights `1e200` and largest finite float, nine integer coordinates from -4 to 4, exact Gaussian row; RuntimeError numerical failure or independently checked exact-model success, with valid covariance and input preservation |
| `s29_s31_large_noise_units_unrepresentable_height_variance` | 1 | Well-conditioned stationary Gaussian with finite data and a data-unit factor `1e160`; RuntimeError or verified success with a warning and nonfinite height variance, without fabricated finite uncertainties |
| `s29_s31_public_numerical_backend_failure` | 2 | A documented NumPy decomposition fails on valid finite operands inside the selected custom optimizer or during covariance calculation; RuntimeError with input preservation, or an independently verified mathematically valid successful result |

All product calls use only `dphtools.utils.fitfuncs.spectrum_fit`, through the
unchanged `_preserving_call` helper. It checks data, coordinates and full guesses
in `finally`, on success and failure. No production helper, module patch, seam,
supplied covariance, or optimizer-supplied Jacobian is used. A None successful
return, an input-validation exception, a leaked backend exception, or mutation
cannot satisfy the numerical-failure helper. No product timeout is imposed.
The owner-supplied earlier stalled probe remains an incomplete observation;
this session does not claim that it completed or supplied a result.

### Exact-height and noise-projection proofs

For the extreme-height cases, `P=exp(-x*x/2)`, `y=A*P`, and the full initial
row is `(A,0,1)`. Every sample is finite because `0<P<=1` and A is finite.
There are nine observations and three parameters, hence six residual degrees
of freedom. The exact generating row achieves zero squared residual sum and
is an exact minimizer. The custom callable checks physical initial values,
bounds, the limit, and the residual at that row, then returns a copied physical
vector and Boolean success. It makes one residual call under `max_nfev=10`.
No iterative convergence or arbitrary parameter vector supplies the oracle.

Using relative amplitude coordinates, the model Jacobian divided by A is
`Jrel=(P,P*x,P*x*x)`. It is full rank with condition number below 3. All three
columns have magnitudes at most one. Physical derivative values remain finite,
although squaring the largest can exceed float representation; the fixture
checks this in logarithms without causing an oracle overflow. Successful model,
residual and parameter checks divide data-valued quantities by A first.
Their tolerances cannot accept an arbitrary width, a lost center sample, or
wrong peak-height units just because the data are large.

For the ordinary stationary fixture, `x=linspace(-4,4,201)`, `(A,c,sigma)=(3,0,1)`
and `m=3*P`. The analytic physical model-Jacobian columns are
`J=(P,m*x,m*x*x)`. Let Q span J's columns and
`u_i=0.01*(sin(1.63*i)+0.6*cos(0.71*i))`. Define `y=m+u-Q*(Q.T*u)` and use the
represented `r=y-m`. Algebraically, `J.T*r=0`. The measured normalized gradient
projection is `1.7330390199790092e-15`, below the independently fixed `1e-12`
bound. The column-normalized Jacobian condition number is `1.931855195700513`.

For half the squared residual objective, `H=J.T*J-K`, where
`K=sum_i(r_i*Hessian(m_i))`. At the chosen row, the independent second derivatives
are `m_AA=0`, `m_Ac=P*x`, `m_A_sigma=P*x*x`, `m_cc=m*(x*x-1)`,
`m_c_sigma=m*(x**3-2*x)`, and `m_sigma_sigma=m*(x**4-3*x*x)`.
The fixture requires `norm(K,2)<0.01*lambda_min(J.T*J)`; the measured ratio is
`6.75008518981806e-5`. H's eigenvalues are approximately
`(28.014499,199.400969,315.398190)`, all positive. This establishes a strict
local minimum to floating-point precision, without reading or trusting a
product optimizer result. R3 requires minimization, not a global optimum.

Complex-step checks of the independent elementary Gaussian validate the first
derivatives at relative/absolute `2e-14`. Central second differences with step
`1e-4` validate the six analytic second derivatives at relative/absolute `2e-6`.
Real unpatched NumPy least squares gives rank 3 and a step below absolute
`2e-12` in these base parameter units. Real SVD reconstructs J and agrees with
the independent inverse Gram matrix within relative/absolute `2e-14`.
These checks do not prescribe a production differentiation/decomposition method.

R3 gives `C=(r.T*r)/(201-3)*inv(J.T*J)`. Observed base squared residual sum is
`0.013791440787719714` squared data units and height variance is
`2.357879943719217e-6` squared base data units. Multiplying data and amplitude
by finite `s=1e160` maps parameters through `D=diag(s,1,1)`. Since
`Jphysical=s*J*inv(D)` and `rphysical=s*r`, covariance becomes `D*C*D`.
Thus physical height variance is approximately `2.35788e314` squared physical
data units, while all inputs and model samples remain finite.
The logarithmic calculation gives `log10(C_AA_physical)=314.37252168833004`,
well above `log10(max_float)`, approximately 308.255. This is a physical
uncertainty-representation limit, not loss of identifiability. Scaling represented
data back preserves stationarity below `1e-12` and covariance within relative
`2e-12`, absolute `2e-18`. The positive-curvature proof is unaffected by this
roundoff margin.

The large-unit custom callable returns the proved stationary physical vector,
after one residual evaluation and physical-protocol checks. A successful return
must have the correct finite positive row, empty background, finite fitted values
and residuals, the independently checked model, and real symmetric covariance.
Its height variance must be nonfinite and an observable warning must exist.
NaN or infinity and broader nonfinite covariance markings are allowed. No exact
warning category/text is imposed. Every finite covariance entry must agree with
the physical analytic convention after sequential unit conversion; finite
diagonal entries must be nonnegative. An honest all-nonfinite uncertainty matrix
with a warning is allowed; a finite substituted height variance is rejected.
This follows the R3 prohibition on fabricated finite uncertainty and leaves
nonfinite encoding unspecified.

The extreme-height success path computes covariance expectations from the
returned, independently checked residuals in relative amplitude coordinates.
This accommodates arithmetic roundoff at enormous heights. For exactly zero
represented residuals the identifiable covariance is zero. Otherwise the same
logarithmic overflow test, warning checks, and finite-entry comparison apply.
No unavoidable failure is inferred from a particular arithmetic implementation.

### Numerical fault boundaries, discrimination and limits

Opened primary references document a `LinAlgError` on nonconvergence at
[NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
and [NumPy singular value decomposition (SVD)](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).
The [floating-point limits reference](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
defines the largest representable value. The opened
[R3 covariance reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.curve_fit.html)
supports local linear scaling; the opened
[optimizer reference](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html)
supports the residual objective. R3 supplies public RuntimeError classification.

The ordinary fixture's real decompositions converge. No reproducible natural
backend nonconvergence on this valid, well-conditioned finite fixture was
established. Making operands invalid would confound numerical failure with input
validation. A narrow fault injection instead models the documented numerical
nonconvergence, without depending on a platform-specific failing matrix.

In `optimizer_lstsq`, the selected custom callable evaluates the real physical
residual, then performs a real analytic Gauss-Newton stationarity check through
`np.linalg.lstsq(J,residual,rcond=None)`. If available, its negligible step confirms
the already proved minimum before the adapter returns the physical vector.
Only this documented dependency is replaced with a nonconvergence exception.
The injection verifies the exact independently known finite J and residual
operands before raising. This proves propagation/classification of a numerical
failure inside a legitimate selected optimizer, not arbitrary callback-programming
exceptions or default LM backend behavior.

In `covariance_svd`, the custom callable returns the stationary vector without a
decomposition. Only `np.linalg.svd` is patched during the public fit call. Each
injected matrix must be real, finite, nonempty and two-dimensional. No covariance
algorithm or operand scaling/shape beyond that documented input domain is required.
The patch is restored before all result checks. A conforming implementation may
use another valid algorithm, an earlier imported binding, or recover from failure;
it may return success with the independently verified row/model/residuals and
analytic finite covariance. No assertion requires this backend to be called on
success. A failed public call must have actually reached the injected boundary,
so an unrelated rejection cannot satisfy the test. This case proves only the
public failure boundary when the chosen dependency is reached; it is not a
universal covariance-method or backend-coverage guarantee.

Arithmetic comparisons reuse `2e-12` relative/absolute tolerances in normalized
base units; reconstruction uses relative `2e-10`, absolute `2e-12` in scaled data
units. Covariance comparisons reuse relative `0.01`, absolute `2e-4` after division
by independent uncertainty units. Covariance symmetry reuses relative `2e-10`,
absolute `2e-12`, with symmetric nonfinite encodings allowed for extreme cases.
These tolerances and both outcome alternatives were fixed before product calls.
The base 201-sample degrees-of-freedom correction is 1.492537%, beyond the variance
tolerance. Independent synthetic checks reject zero/unit/max-float fabricated
finite height variance, missing warnings, doubled finite width variance, and
nonzero covariance for an exact zero-residual identifiable model. They accept
correct finite entries and honest warned NaN/infinity encodings. No expectation
or tolerance was tuned to product output; no unresolved requirement was found.

### Source-free checks, preservation and classification

All evidence resides outside Git at
`/private/tmp/SPECTRUM-FITTING-001-A-postrepair-xm4t1pf2/`.
Every Python launch installs category/message/filename/line-only warning formatting
and exception-only rendering before imports. Temporary `sitecustomize.py` supplies
the hooks to child interpreters; the unchanged real LM child has its own hooks.
Pytest uses `-p no:warnings --tb=no -q -ra --capture=tee-sys`. The diagnostic plugin
prints only failure identifiers and exception type/message. Recorded warnings
are explicitly rendered even when a fit fails. Pytest exit status is printed
and propagated through `set -o pipefail`; no diagnostics or failure status are hidden.

Exact final full-spectrum command from the permitted worktree:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/SPECTRUM-FITTING-001-A-postrepair-xm4t1pf2 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/mplconfig PIP_CACHE_DIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/pip-cache /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3 /private/tmp/SPECTRUM-FITTING-001-A-postrepair-xm4t1pf2/launch.py tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-postrepair-xm4t1pf2/spectrum-tests-exact-final.log
```

Black uses the same interpreter/environment with `format.py --check --line-length
99 tests/test_spectrum_fitting.py`, and `black-check-exact-final.log`. Independent
checks substitute the evidence file `test_fixture_checks.py` for the spectrum
file, retaining `fixture-checks.log`. The initial focused command adds
`-k 'extreme_finite_height or large_noise_units or public_numerical_backend_failure'`
and retains `new-cases.log`. Preservation uses `preservation.py`, retaining
`preservation-final.log`, `preflight.json` and `final-identity.json`.

| Check | Result | Evidence file |
|---|---|---|
| Independent fixture/environment/oracle checks | 6 passed in 0.64 s; exit 0; no product import or warnings | `fixture-checks.log`; `analytic-preflight.json` |
| Initial new cases | 1 failed, 4 passed, 276 deselected in 0.67 s; exit 1 | `new-cases.log` |
| Final full spectrum on exact formatted test bytes | 1 failed, 280 passed in 1.38 s; pytest/shell exit 1; no errors/skips | `spectrum-tests-exact-final.log` |
| Black 99 on exact final test bytes | One file would be left unchanged; exit 0 | `black-check-exact-final.log` |
| Original tests | All 45,853 preimage bytes are an exact prefix; original AST nodes/constants/oracles/tolerances are unchanged | `tests-before.py`; `preservation-final.log`; `final-identity.json` |
| Historical report | All 76,378 preimage bytes remain an exact prefix | `report-before.md`; `preservation-final.log`; `final-identity.json` |

The `1e200` case returned a verified exact-model fit with zero covariance. Largest
finite height retained an overflow RuntimeWarning and returned RuntimeError.
Large noise units returned RuntimeError for unrepresentable covariance.
The optimizer boundary reached the verified finite `(201,3)` J and residual,
then leaked `LinAlgError: Numerical decomposition did not converge (injected)`.
Caller-input checks completed before the leak. The covariance boundary reached
a finite `(201,3)` operand and returned RuntimeError. Its alternative successful
path is independently checked but was not the observed product outcome.

**Classification: product defect — numerical failure inside a selected custom
optimizer leaks LinAlgError instead of RuntimeError**, high confidence, subject
to fresh independent B validation of the injection and tests. This is valid
post-repair product-red evidence at S29/S31. The other new outcomes satisfy R3.
All 276 previously accepted cases pass on this run. No test, fixture or environment
failure explains the optimizer leak; A performs no product repair. The full run
retains ten RuntimeWarnings and three OptimizeWarnings, including the new height
warning and previous numerical/uncertainty warnings. The real LM child exits 0.

All attempts remain retained. An initial Black check passed. Adding an injection
diagnostic print subsequently caused a format-only test defect; the intermediate
full run still had 1 failed/280 passed in 1.45 s, and intermediate Black exit was
1 (`spectrum-tests-final.log`, `black-check-final.log`). Formatting only appended
bytes corrected it; exact complete AST equality was asserted, then full-spectrum
and Black checks were rerun on the final bytes as above. No oracle or tolerance
changed. Final test SHA-256 is
`b241e8d23eda41ecdc3f95ed6d9641007d58e5e000a78947243a8e43e9ed9417`.
The report's final hash is returned outside its tracked tree.

Environment: supplied non-Conda CPython 3.12.14, NumPy 2.5.3, SciPy 1.18.1,
pytest 9.1.1, Black 26.5.1; interpreter/base prefixes both identify the supplied
clean Python. No package, environment or repository configuration edits occurred.
Public imports execute the worktree entry point; no source bytes were inspected.
The supplied checkpoint identifies prior acceptance, not the uninspected current
product candidate. Full coverage/platform-matrix verification was not run or
claimed. These passing additions alone do not prove complete coverage.

Read only root AGENTS.md, the owner packet, contract R3, selected design-tests
skill, setup.cfg conventions, the permitted spectrum test/historical report,
opened primary references, and this session's own source-free evidence. No
implementation/private helper source, source/history/diff, private coverage data,
other tests/oracles, PROJECT.md, task state/role packet, existing lessons, or role
conversation was read. Only the two authorized tracked files were edited. No
Superpowers, optional memory, delegation, commit/push/PR, broad full verification,
merge or release occurred. **Source exposure: none observed.** Diagnostics retain
filenames/line numbers without implementation excerpts or source-bearing tracebacks.

Metrics: 33 unchanged scenarios / 281 cases / 5 added cases; 1 fresh root A launch
(7 known A launches), 0 reviewer/C launches in this session. Owner-supplied original
B 2/2, format B 1/2, earlier coverage B 1/2 and numerical B 1/2 are accepted/closed.
This new post-repair correction window has B 0/2 used, round1 next. Initial C and
C repair 1/1 are used; this work grants no further repair. Partial measured
authoring/check interval: 192 s, 19:32:59–19:36:11 UTC on 2026-10-05, excluding
earlier input/reference reading and final report/preservation preparation. Retained
1,247 s plus this interval totals 1,439 s of partial measured spending. No new
budget cap was supplied; exact token usage/billing remain unavailable. No approval,
attempt, spending, scenario or allowance was reset. Fresh independent B round1
review is the next handoff; any further C repair needs owner allowance extension.

## Supplementary large-origin narrow-width public boundary — same post-repair window

This fresh independent root A was authorized for one S29 boundary supplement,
concurrently with A7 authoring other numerical boundaries. This is supplementary
post-implementation public-boundary evidence. It is not original test-first
red evidence or independent acceptance. Owner approval of R3 and all 33 scenarios
is unchanged. The supplied accepted checkpoint is
`f70661a9ed1813312e5bc409fae8a288c8c3d025` (276 cases), with accepted public test
SHA-256 `c5e51bab70912a674d7242541591d1844939dca981c5b78cbce9de70617accca` and
accepted public report SHA-256
`2bd7819d6c6112d2a6660fd03cd5f71685613565f11081fa8b340f3647a8ef3c`.
Both snapshots were hashed before any public probe. Original, format, coverage,
and prior numerical windows remain accepted/closed. This supplement shares A7's
existing post-repair test-correction window, before combined B round1/2. Initial
C and the sole C repair are used (repair1/1). No allowance/window/spending reset,
new budget cap, repair authorization, or competing Current state is introduced.

### Mapping and independent mathematical proof

One parametrized public test, `test_s29_large_origin_narrow_width_point_boundary`,
adds four variants of existing S29: Gaussian/Lorentzian, each with a legitimate
exact-minimizer custom callable or the omitted default LM optimizer. Supporting
expectations are R3's S01/S02 point-model definitions, S08 no-background schema,
S17/S19 finite positive domain, S30 default LM, S31 physical callable protocol,
and local-linear covariance/warning convention. No scenario is added.

The fixture is precisely `x=1e12+arange(9)`, full row `(2,x[4],1e-6)`, and
`background="none"`. Height has data units; origin, center, width and coordinate
spacing have x units. The test independently checks all finite inputs, strictly
increasing coordinates, positive height/width, and `N=9>P=3` (six residual
degrees of freedom). Coordinate spacing is 1 x unit. Around this origin, float64
representable spacing is `2**-13=0.0001220703125` x units, approximately 122
widths; adding or subtracting `1e-6` x units from the center leaves its represented
value unchanged. This establishes a real numerical boundary without choosing any
production derivative step, method, or failure wording.

Let `d=x-center`, `w=1e-6`, and `A=2`. R3 gives

`m_G=A*exp(-0.5*(d/w)**2)`; `m_L=A*w**2/(d**2+w**2)`.

At `d=0`, both samples are exactly 2 data units. Every off-center sample has
`abs(d)>=1`. The Gaussian exponent is at most `-5e11`, so its float64 point
samples are zero. Lorentzian adjacent samples are approximately `2e-12` data
units; endpoint samples are approximately `1.25e-13`. They remain positive and
finite. Data use the accepted independent `_stable_narrow_component` equation
helper. An independent 80-digit Decimal calculation of the Lorentzian model
and analytic scaled sensitivities agrees within relative `2e-14`; observed
maximum model relative error is `2.625329092575691e-15`. Gaussian samples and
sensitivities are checked exactly against the analytic zero-tail result. No
product helper or optimizer output supplies numerical truth.

The full row generates these point observations. Thus mathematical residuals
are zero and RSS=0, the global lower bound of the unweighted sum of squares.
It is a minimizer even when parameters are weakly identifiable. The real custom
optimizer checks the physical initial vector, bounds `(0,-inf,0)` / all positive
infinities, evaluation limit 10, and finite correctly shaped supplied residuals.
It evaluates one physical residual vector and returns only a copied physical
minimizer and Boolean success. It supplies no derivative or covariance. The
other two cases call the real omitted default optimizer. No mocks, production
seams, private coordinates, private derivative recipe or synthetic failure are
used.

### Tolerances, discrimination and allowed outcomes

All proposal equations, data, tolerances, assertions and allowed outcomes were
fixed before public execution. The custom residual at the known minimizer must
satisfy `abs(r_i)<=2e-12*abs(y_i)+8*smallest_subnormal` in data units. This accepts
arithmetic roundoff while resolving the Lorentzian tails instead of comparing
them with an order-one absolute allowance.

Success reconstruction uses the accepted independent stable helper at the
**returned** physical row, with `rtol=2e-10` and absolute floor
`8*smallest_subnormal`, approximately `3.95e-323` data units. These are model
arithmetic checks at identical parameters; an 80-digit sanity check establishes
precision well within this allowance. Existing absolute `MODEL_ATOL=2e-7`
data units applies to fit quality and zero residuals, allowing solver stopping.
Existing residual consistency `rtol=2e-12, atol=2e-12` checks `data-fitted`.
No exact returned center/width/height recovery is required. Independent Gaussian
rows with widths `1e-6`, `1e-4` and `1e-2` produce identical point samples and
rank-one sampled sensitivity, demonstrating why width recovery is inappropriate.

Temporary independent discriminator checks reject zero Lorentzian tails in the
reconstruction check, a lost center sample, Lorentzian width `1e-3` with spurious
fitted tails, peak area substituted for peak height, and the wrong residual sign.
Ordinary equivalent direct Lorentzian arithmetic passes the fixed reconstruction
tolerance. These are mathematical incorrect-output examples, not production
mutations or tuning to observed output.

A numerical fit inability may raise RuntimeError. There is no mandatory failure,
warning category/message, or exception wording. Other exceptions fail, including
ValueError and leaked linear-algebra errors. Input preservation is checked in
`finally` on both outcomes. A successful return must have one finite real positive
row, empty real background, finite real fitted samples/residuals, real 3-by-3
covariance, independently consistent point model/residuals, symmetry with
`rtol=2e-10, atol=2e-12` including symmetric nonfinite encodings, and nonnegative
finite diagonal entries.

The reused helper analytically evaluates physical model-Jacobian columns scaled
by `(A,w,w)`. Gaussian columns are `(m,m*z,m*z*z)` with `z=d/w`; Lorentzian
columns are `(m,2*m*d*w/(d*d+w*w),2*m*d*d/(d*d+w*w))`. The residual sign cancels
in the covariance Gram matrix. Positive column scaling and normalization preserve
rank/estimability. Rank is assessed at the **returned** row. If deficient,
parameters whose column removal leaves rank unchanged need nonfinite diagonal
uncertainties and an observable warning. Any nonfinite covariance also needs an
observable warning. These checks reuse the accepted public uncertainty boundary.

At the initial row, Gaussian rank is 1: center and width sensitivities vanish,
so their individual local-linear uncertainties cannot be finite. Lorentzian
normalized rank is 3 with condition number `1.4688443954432964`: its tiny positive
tails carry formal sensitivity even though center perturbations can be below
coordinate representation. A robust analytical/scaled calculation may succeed;
finite uncertainty is not forbidden solely by small physical units or large
origin. Returned alternatives are assessed without requiring either initial
rank or a particular covariance computation. The identifiable covariance
helper demanding finite entries is deliberately not used here.

### Sequential boundary and retained checks

All initial proposal work and probes lived in
`/private/tmp/SPECTRUM-FITTING-001-A-origin-r8f18rb3/`. Only the accepted B5 public
snapshots were read before A7's completion marker. The frozen proposal SHA-256
is `ae4a51de1fb0391a94dbd71f0d71f078d83299a4417f12b5b0bb40edca42df3d` (4,608
bytes), recorded before the public probe at 2026-10-05 19:37:30 UTC. Mutation
waited for `/private/tmp/dphtools-spectrum-delivery/A7-editing-complete.json`;
exact marker/live hashes and preimages are retained in this evidence directory.

The marker identified A7 test SHA-256
`b241e8d23eda41ecdc3f95ed6d9641007d58e5e000a78947243a8e43e9ed9417` and report
SHA-256 `270b0d4b6eaa1daed5f02fa5ddef6cf1c61a7816a6d33eebe5dbdef8727b6d9e`.
Both exact live identities were verified before interpreting current public
content or mutating either file. Marker bytes were retained and rechecked.
The 57,441-byte A7 test preimage and 94,978-byte A7 report preimage remain exact
prefixes; each also retains its B5 accepted snapshot as an exact prefix.
Appending only the frozen proposal plus two separator newlines added 4,610 test
bytes and one top-level function. Removing only that function restores the
complete 12,276-node A7 abstract syntax tree (AST), excluding location attributes,
SHA-256 `16fe12017fed883d88f5e62e9082e89cae98d74a996b4ebe55186bae263eb4ad` under
CPython 3.12.14. Every prior decorator, constant, helper, oracle, tolerance and
case is unchanged. The final file contains 285 cases: 276 accepted, A7's five,
and this supplement's four. No unexplained changed input was observed.

Every Python probe/check installed category/message/file/line-only warning
rendering and exception-only rendering before imports. All pytest runs used
`-p no:warnings --tb=no -q -ra --capture=tee-sys`. A diagnostic plugin printed
failure identifiers and exception/assertion type/message without traceback or
source. Captured boundary warnings were explicitly rendered in `finally`.
Pytest and pipeline failure statuses were retained. The existing real LM child
has its own source-free hooks. No temporary sitecustomize or environment repair
was needed in this session; only task-local launchers and command-local variables
were used.

Exact final spectrum command from the authorized worktree:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/mplconfig PIP_CACHE_DIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/pip-cache /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3 /private/tmp/SPECTRUM-FITTING-001-A-origin-r8f18rb3/launch.py tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-origin-r8f18rb3/spectrum-tests-final.log
```

Black used the same command-local environment/interpreter, with
`format.py --check --line-length 99 tests/test_spectrum_fitting.py`, retaining
`black-check-final.log`. Independent fixture checks and the proposal probe used
`launch.py` with respectively `test_fixture_checks.py` and `test_proposal.py`
under the evidence directory and `--confcutdir` set to that directory.
Preservation used `preservation.py`, retaining `preservation-initial.log`,
`preservation-final.log`, and `final-identity.json`. Black formatting was applied
only to new temporary proposal/check files; tracked preimages were never reformatted.

| Check | Actual result | Evidence in the temporary directory |
|---|---|---|
| Independent fixture/geometry/oracle/discrimination checks | 5 passed in 0.59 s; pytest/shell exit 0; no product import or warnings | `fixture-checks.log`; `gauss-proof.json`; `lorentz-proof.json`; `environment.json` |
| Smallest frozen public proposal, before live access | 4 passed in 0.63 s; pytest/shell exit 0; all four RuntimeError outcomes; one retained RuntimeWarning each | `proposal-probe.log`; `proposal-frozen.json` |
| Combined temporary candidate format preflight | Black 99 exit 0; one file unchanged; no tracked mutation | `candidate-black-check.log` |
| Whole final spectrum file | 1 failed, 284 passed in 1.77 s; pytest/shell exit 1; no errors/skips | `spectrum-tests-final.log` |
| Exact final tracked Black 99 | Exit 0; one file unchanged | `black-check-final.log` |
| Sequential inputs / exact prefix and AST proof | Exact A7 marker/live identities; all accepted and A7 bytes/AST preserved; one function/four cases appended | `sequential-preflight.json`; preimages; `preservation-final.log`; `final-identity.json` |

All four supplementary cases reach a contract-permitted RuntimeError, preserve
caller inputs, and retain an invalid-value RuntimeWarning. Both robust success
alternatives are specified independently but were not observed against the
product. No failure or warning is required merely because the current candidate
fails numerically. **Classification for this supplement: authorized public
numerical-boundary coverage, passing initially**, high confidence, subject to B's
independent expectation/tolerance review. No product-red claim is manufactured.

The single combined-file failure remains A7's
`test_s29_s31_public_numerical_backend_failure[optimizer_lstsq]`: a valid finite
custom-optimizer decomposition boundary leaks
`LinAlgError: Numerical decomposition did not converge (injected)`.
**Classification: product defect — leaked selected-optimizer numerical failure**,
subject to independent B validation of A7's fixture/injection and expectations.
All 276 accepted cases pass, four other A7 cases pass, and all four supplementary
cases pass. The full run retains 14 RuntimeWarnings and three OptimizeWarnings;
the real LM child exits 0. No failure, diagnostic, warning, count, skip, or status
was hidden. A performs no product repair; the supplied C repair1/1 is exhausted.

All local attempts remain retained. Initial temporary formatting of two files
failed, exit 123 (`proposal-black.log`), because this session's Black wrapper
lacked a multiprocessing main guard. This is a **temporary tooling-script defect**,
not a product or environment defect. The task-local wrapper was corrected; the
next formatting attempt passed (`proposal-black-corrected.log`). No product
probe, oracle/tolerance change, repository/environment edit, or implementation
source exposure occurred in that failed attempt. Independent/public checks then
ran once on the frozen proposal. No later test edits invalidated them.

Final test SHA-256:
`c3151ae5fd74a315ad185b6fbc3d4b137909f87a275ce927e5ff4137dcd6f43c`.
The report's final SHA-256 is returned outside its tracked tree. Public imports
exercise the working-tree entry point; the accepted checkpoint identifies prior
test acceptance and makes no identity claim about uninspected product bytes.
Environment evidence: supplied non-Conda CPython 3.12.14, macOS 27.0 arm64,
NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1, Black 26.5.1. Interpreter and base
prefixes both identify the supplied clean Python.

Read only root AGENTS.md, selected design-tests skill, owner packet, R3, accepted
B5 public snapshots, the coordinator completion marker, current permitted public
tests/report only after exact marker verification, and this session's own evidence.
No implementation/private-helper source, history/diff, coverage report, other
preexisting tests/oracles, PROJECT.md, task state/role packet, lessons or role
transcript was read. Only the two authorized tracked paths were edited. No
Superpowers, memory, delegation, config/rules/environment edit, commit/push/PR,
full verification, merge or release occurred. **Source exposure: none observed.**
Warning and failure diagnostics retained filenames/line numbers without product
source excerpts or source-bearing traceback. Mathematical sensitivity probes
constrain no production derivative algorithm. Full coverage and platform-matrix
measurement were neither run nor claimed; B must review the combined additions.

Metrics: 33 unchanged scenarios / 285 cases / 4 supplementary cases; 1 fresh root
A launch (8 known including six in B5 history and A7), 0 reviewer/C launches in
this session. Original/format/coverage/prior numerical windows remain
accepted/closed. This is the same post-repair correction window as A7, B 0/2
used, combined round1/2 next. Initial C and repair1/1 remain used. Measured partial
authoring/probing/sequential-check interval: 288 s, 19:35:22–19:40:10 UTC on
2026-10-05, including initial draft preparation and the marker boundary, excluding
earlier permitted input reading and final report completion. Prior recorded
1,439 s plus this interval totals 1,727 s of partial measured spending; overlapping
root sessions are counted as agent effort, not elapsed wall time. All prior
attempts/spending are retained. No fixed owner time/token cap or new cap is
introduced; exact model-token usage/billing is unavailable. Combined fresh B
round1 is the next handoff; A grants no acceptance or further C repair allowance.

## Correction of the two B6 test defects — same post-repair window

Fresh independent root A was authorized only to correct the two evidenced B6
observers. This is post-repair test correction, not original test-first evidence.
The owner packet reports B6 round1/2 rejection of these restrictions, a spectrum
run of 284 passed/1 failed, 16 independent checks and passing Black 99. It also
confirms that the selected custom optimizer's LinAlgError leak is valid product-red
evidence. That report/transcript was not read. R3 and the 33 approved scenarios
remain unchanged. B round1 is spent; fresh independent B round2 is next. C
repair1/1 remains exhausted. The coordinator's extra-repair request supplies no
authorization in this packet. No window, allowance, spending or budget cap resets.
This appendix is historical evidence, without acceptance or competing Current state.

Confirmed rejected preimages: test SHA-256
`c3151ae5fd74a315ad185b6fbc3d4b137909f87a275ce927e5ff4137dcd6f43c`; report SHA-256
`d001e99a794331c5e6031814c0cca2f05673a8e57b0a7cdfd450db94bccb3cf7`.
Both exact preimages and all local attempts are retained outside Git. The B5
accepted snapshots retain hashes `c5e51bab70912a674d7242541591d1844939dca981c5b78cbce9de70617accca`
and `2bd7819d6c6112d2a6660fd03cd5f71685613565f11081fa8b340f3647a8ef3c`.

### Explicit historical corrections and unchanged mapping

The earlier A7 restriction to two-dimensional covariance SVD operands was a
**test defect**. Its statement that this described the documented input domain
was incorrect. [NumPy's SVD documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html),
opened in this session, permits operands with at least two dimensions and applies
decomposition to stacked matrices. The injection now accepts `ndim>=2`, while
retaining real, finite, nonempty operand checks. The custom optimizer's independently
derived least-squares context still requires a two-dimensional matrix exactly
equal to the analytic Jacobian and the independently known residual. Its adapter
and those exact comparisons are unchanged. Input preservation, reached-boundary
failure checks, RuntimeError classification, recovery/alternate-algorithm success,
and the physical model/covariance oracle are unchanged.

The A7 large-unit statement that height variance must always be nonfinite with
a warning was also a **test defect**. The physical variance exceeds float64 range;
R3 does not promise a float64 covariance dtype. That historical mapping, overflow
wording and mandatory-warning claim are superseded by this correction: nonfinite
height variance and an observable warning are required when its physical variance
exceeds the actual returned covariance dtype's maximum. A correct representable
finite variance may succeed without a warning. Every finite covariance entry still
has to satisfy the physical analytic expectation, symmetry and nonnegative
variances; all nonfinite encodings still require an observable warning. Exact
zero-residual identifiable covariance still has to be exactly zero.

The observer now uses `np.log(np.finfo(covariance.dtype).max)`. It takes the logarithm
before any narrowing conversion, so a valid wider maximum is not first converted
to float64 infinity. [NumPy's finfo documentation](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html),
opened here, defines limits for the supplied dtype and documents platform variation
in longdouble. The success diagnostic now says physical covariance rather than
claiming variance overflow for every possible dtype. This changes no assertion
outside the two rejected observers. Earlier A7/A8 attempts, restrictions and
claims remain in their historical bytes; this appendix explicitly corrects them.

| Existing mapping | Corrected observer | Cases/tolerances |
|---|---|---|
| S29/S31, `test_s29_s31_public_numerical_backend_failure[covariance_svd]` | Documented stacked SVD domain; legitimate optimizer matrix context retained | Same two boundary cases and all physical failure/success tolerances |
| S29/S31, the extreme-height and large-noise-unit tests | Actual returned covariance dtype's representable range | Same three cases; unchanged finite-entry `rtol=0.01, atol=2e-4` in independent uncertainty units |

No functions, parameterizations, fixtures or scenarios were added to the public
suite: 285 cases remain, including all 276 accepted cases and all nine A7/A8
additions. The four A8 boundary variants and their oracles/tolerances are unchanged.

### Host-independent range proof and both-direction observer checks

The unchanged stationary Gaussian yields base height variance
`C_AA=2.357879943719217e-6` squared base data units. For data-unit factor `s=1e160`,
`D=diag(s,1,1)` and `C_physical=D*C*D`, so

`V=C_AA*s*s=2.3578799437192171e314` squared physical data units.

A 90-digit Decimal calculation uses exact conversions of the represented base
variance and unit factor. It establishes `log10(V)=314.372521688330045...`, whereas
`log10(float64_max)=308.254715559916744...`. Thus float64 cannot represent V.
For a mathematical binary format with 113 significant bits and maximum exponent
16384, the largest finite value is `(2-2**-112)*2**16383`, with decimal logarithm
`4932.075448958667902...`. Hence `float64_max < V < max_wide`. Rounding at relative
precision `2**-112` is far below the fixed 1.02% diagonal-variance allowance.
The old threshold would reject that mathematically valid finite covariance;
the corrected threshold permits it, subject to the same physical-entry comparisons.
This is a mathematical possible-format proof, not a claim about local availability.

Local ARM longdouble has maximum exponent 1024, equal to float64. No local wider
result, product wide-dtype success, or additional runtime dependency is claimed.
The retained proof includes a real wider-longdouble observer check only when
such a dtype exists; that conditional branch did not execute here and no pytest
skip hides it. Separate narrow instrumentation verifies that the observer queries
the actual dtype and passes its maximum directly to NumPy logarithm without
float64 narrowing; it is explicitly instrumentation, not a synthetic host dtype.

The 25 independent observer checks do not import product code. They demonstrate:
valid SVD reconstruction and analytic covariance for `(201,3)`, `(1,201,3)` and
`(2,1,201,3)`; successful recovery after injected failure; legitimate RuntimeError
translation at both boundaries; valid covariance through an alternate algorithm;
continued rejection of an optimizer LinAlgError leak and vector/empty/nonfinite/
complex decomposition operands; preserved exact optimizer matrix/residual context;
actual float32/float64/longdouble limit lookup and valid finite physical covariance;
honest warned nonfinite encodings when range is exceeded; rejection of missing
warnings, saturated maximum/zero/unit fabricated height variance, doubled width
variance and nonzero covariance for exact zero residual; and the mathematical
range/nonnarrowing proofs. All equations and tolerances precede product execution.
These checks are evidence outside the public suite, not new contract scenarios.

### Exact checks, retained attempts and classification

Evidence directory: `/private/tmp/SPECTRUM-FITTING-001-A-B6-correction-20261005/`.
Every Python launch/probe/test installs category/message/filename/line-only
`warnings.formatwarning` and exception-only rendering before probe imports. Pytest
uses `-p no:warnings --tb=no -q -ra --capture=tee-sys -o addopts=`. The diagnostic
plugin prints only failed identifiers, exception type and message. Captured
warnings and the unchanged LM child retain source-free rendering. Exit statuses
are propagated directly; stdout/stderr are retained without a masking pipeline.
No sitecustomize, environment, repository config or rule file was edited.

Exact whole-spectrum command, from the authorized worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/mplconfig PIP_CACHE_DIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/pip-cache /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3 /private/tmp/SPECTRUM-FITTING-001-A-B6-correction-20261005/launch.py /private/tmp/SPECTRUM-FITTING-001-A-B6-correction-20261005/spectrum-results.json tests/test_spectrum_fitting.py > /private/tmp/SPECTRUM-FITTING-001-A-B6-correction-20261005/spectrum-tests.log 2>&1
```

Black uses the same interpreter and command-local variables with
`run.py black --check --line-length 99 tests/test_spectrum_fitting.py`.
Observer checks use `launch.py observer-results.json test_observer_proofs.py`
with absolute evidence paths and `--confcutdir` set to the evidence directory.
Preservation and count checks use `run.py preservation.py` and
`run.py outcome_summary.py`; all scripts/logs are retained in that directory.

| Check | Actual result | Evidence |
|---|---|---|
| Independent observer/proof checks | 25 passed in 0.67 s; pytest/process exit 0 | `observer-checks.log`, `observer-results.json`, `mathematical-range-proof.json` |
| Exact whole spectrum | 284 passed, 1 failed in 1.43 s; pytest/process exit 1; no errors/skips | `spectrum-tests.log`, `spectrum-results.json` |
| Exact Black 99 | One file would be left unchanged; exit 0 | `black-check.log` |
| Accepted case membership/outcomes | All 276 accepted cases passed; total 285 | `outcome-summary.log`, `outcome-summary.json` |
| Preservation | Accepted 45,853 test bytes, every accepted AST node/oracle/tolerance unchanged; only authorized observer AST changes; all other A7/A8 AST unchanged | `preservation-corrected.log`, `final-identity.json` |
| Historical evidence | All 111,341 rejected report bytes retained as exact prefix, including all 76,378 accepted report bytes | `report-before.md`, `final-identity.json` |

The sole public failure is still
`test_s29_s31_public_numerical_backend_failure[optimizer_lstsq]`:
`LinAlgError: Numerical decomposition did not converge (injected)` on the valid
finite `(201,3)` analytic Jacobian and independent residual. Input-preservation
checks complete before propagation. **Classification: product defect — leaked
selected-optimizer numerical failure instead of RuntimeError**, high confidence;
fresh B round2 must review the corrected candidate. Neither covariance observer
defect explains or weakens this separate red evidence. No production repair occurred.

The other appended cases satisfy R3. Extreme height `1e200` returns a verified
exact fit; maximum finite height and large-noise units return RuntimeError.
Covariance SVD reaches a finite `(201,3)` operand and returns RuntimeError. All
four A8 variants return RuntimeError. The full log retains 14 RuntimeWarnings
and three OptimizeWarnings; the real LM child exits 0. No warning category/message,
failure count, exception or status was hidden.

One initial preservation attempt failed with source-free `IndexError: list index
out of range`, exit 1 (`preservation-initial.log`). **Classification: temporary
tooling-script defect**: the local launcher dropped the script's argument while
forwarding it to runpy. Correcting only that launcher yielded preservation exit 0
(`preservation-corrected.log`). The failed attempt remains retained. This changed
no product, public test, oracle, tolerance, environment or repository configuration.

Final test SHA-256:
`4c40f4cf5bac99b3d47d4f3bb1fcf18ae366941b23428dc0bc9fbfa7bf81cafe`.
The final report hash and exact final preservation results are returned outside
its tracked tree. Accepted module AST hash under CPython 3.12.14 is
`ee2bc688f9c90d1b4044f21223d23f72237b36b31b5b64d1b28d9863f0053b80`
(9,790 nodes, locations excluded). No tracked formatting was applied.

Observed host: supplied clean non-Conda CPython 3.12.14, macOS 27.0 arm64,
NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1 and Black 26.5.1. Interpreter/base prefixes
identify the supplied clean Python. Public execution exercises the uninspected
worktree product; accepted checkpoint identity is prior test acceptance, not a
claim about current product bytes. Wider-range runtime execution, full coverage
and Linux/macOS/Windows verification were not performed or claimed.

Read only root AGENTS.md, design-tests skill, owner packet, R3 public contract,
current permitted spectrum tests/historical report, B5 accepted public snapshots,
the two opened NumPy references, and this session's own evidence. No implementation,
private helpers/source/history/diffs, private coverage, other preexisting tests/
oracles, PROJECT.md, task state/role packets, lessons or role transcripts were read.
Only the two authorized tracked files changed. No Superpowers, memory, delegation,
environment/config/rules edits, commit/push/PR, full verification, merge or release
occurred. **Source exposure: none observed.** Diagnostics disclosed filenames/line
numbers without implementation excerpts or source-bearing tracebacks.

Metrics: 33 unchanged scenarios / 285 unchanged public cases / 0 added cases;
1 fresh root A launch (9 known), 0 reviewer/C launches in this session. Same
post-repair B window: 1/2 used and rejected, one remaining. C repair1/1 exhausted;
extra repair requested but not authorized. All earlier attempts/spending and closed
windows remain retained. Partial measured authoring/proof/check interval: 243 s,
19:50:18–19:54:21 UTC on 2026-10-05, excluding earlier input/reference reading
and final report preparation. Prior recorded 1,727 s plus this interval totals
1,970 s of partial measured agent spending. No fixed owner time/token cap or new cap is supplied;
exact model-token usage/billing are unavailable. Fresh independent B round2 is next.
After another nonacceptance, no extra rewrite is authorized without diagnosed
resolution or explicit extension.

## Supplementary complex SVD operand observer correction — same post-repair window

This fresh independent root A (A10) corrects one further independently evidenced
test-observer defect before combined B round2/2. R3, all 33 scenarios, and the
285 public cases remain unchanged. This is a test correction and independent
counterexample, not original test-first evidence, product-red evidence, or
self-acceptance. B6 round1/2 rejected the dimensional restriction and float64
output-range assumption; A9 corrected those before this sequential supplement.
The current owner packet explicitly authorizes one additional C repair for the
valid custom-optimizer numerical exception defect. That later authorization
supersedes the earlier report's statements that the extra repair was merely
requested. Original repair1/1 remains spent; no review window, scenario, spending
or budget cap resets. Only B round2/2 remains in this correction window.

### Sequential identity and narrow correction

All proposal work initially used only the supplied B5 accepted and B6 rejected
public snapshots, R3, root AGENTS.md, and design-tests skill. No live test/report
was read or written before `/private/tmp/dphtools-spectrum-delivery/A9-editing-complete.json`
existed. Both live hashes exactly matched that marker before live access:

- A9 test: `4c40f4cf5bac99b3d47d4f3bb1fcf18ae366941b23428dc0bc9fbfa7bf81cafe`.
- A9 report: `6391435c246b4c3933c2f7437587b505aad4d669ead710b4efae315dcc50d375`.

Exact copies are retained in
`/private/tmp/SPECTRUM-FITTING-001-A-complex-2LPa2aaY/A9-tests-before.py`
and `A9-report-before.md`. A9 had removed the two-dimensional covariance
restriction but still asserted `np.isrealobj(matrix)` for covariance SVD operands.
The only additional test change moves this real-only check into the test-owned
`optimizer_lstsq` context, whose operand is specifically the independently
derived real Jacobian. Covariance operands retain `ndim>=2`, nonempty and finite
checks. The exact least-squares matrix/residual checks, injected LinAlgError,
required reached-boundary RuntimeError outcome, input preservation, permitted
verified success, public real outputs and all mathematical expectations/tolerances
remain unchanged. A9's dtype-range correction and other bytes remain intact.

Earlier A7/A9 descriptions of real-only covariance SVD inputs, including A9's
claim that complex operands must be rejected, are superseded here. They describe
an unsupported **test defect**. [NumPy's SVD documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html),
opened independently in this session, allows real or complex input with at least
two dimensions and stacked decomposition. R3 requires real public observations,
parameters, model and physical covariance. It specifies no dtype for an internal
numerical operand. No complex public output is required or permitted by this
correction. No production algorithm or binding is mandated.

### Independent counterexample and outcome discrimination

The unchanged stationary fixture has `N=201`, physical `(A,c,sigma)=(3,0,1)` and
projected finite noise. Its analytic Jacobian is
`J=[g, m*(x-c)/sigma**2, m*(x-c)**2/sigma**3]`, with `g=exp(-z**2/2)` and
`m=A*g`. Independently observed normalized stationarity is `1.73304e-15`, the
column-normalized condition number is `1.93186`, and the fixture's unchanged
curvature inequality proves a strict local minimum. All inputs use ordinary
finite base units; no extreme-unit or covariance-output-range issue is involved.

A temporary conforming observer alternative calls the supplied optimizer via
its physical residual/initial/bounds/limit protocol, computes J at the returned
solution and casts J to complex128. With NumPy's documented decomposition
`J=U*S*V^H`, it forms `F=V/S` and real covariance
`C=real(F*F^H)*RSS/(N-P)`. Here `RSS=0.013791440787719714` squared data units and
`N-P=198`. Because this J is physically real, `J^H*J=J.T*J`; the Hermitian factor
product yields the same real physical covariance. Unpatched covariance agrees
with the unchanged analytic expectation at `rtol=1e-12, atol=1e-14`; maximum error
in independent uncertainty units is `1.21242e-15`. Public parameters, fitted
values, residuals and covariance satisfy all existing real-output checks.

The B6 observer rejects this `(201,3)` complex128 path with AssertionError solely
at its real-only check. A9's completed observer also rejects complex128 2-D and
4-D paths. The corrected live observer accepts real and complex inputs with
shapes `(201,3)`, `(1,201,3)` and `(1,2,201,3)`, including legitimate RuntimeError
translation, successful recovery and a correct earlier binding/alternate path
that does not reach the injection. Independent negative alternatives still fail
for a leaked LinAlgError, unrelated early RuntimeError, mutated data or guesses,
doubled covariance, incorrect fitted model, complex public covariance, scalar/
vector/empty/nonfinite operands. The unchanged real optimizer-lstsq context still
accepts a valid translated numerical failure. No probe imports production code.

The initial frozen-snapshot proof passed 38 checks in 0.68 s, exit 0. Expanded
proof against the exact corrected live observer passed 44 checks in 0.70 s,
exit 0, with no skips or warnings. Both attempts are retained as
`observer-proof.log` and `live-observer-proof.log`; the public suite gains no case.
The alternatives are independently derived discriminating witnesses for this
fixture, not a general fitter, production repair or universal backend-coverage
claim. A covariance implementation may use any conforming numerical algorithm.

### Exact verification evidence

Evidence and all temporary scripts reside outside Git at
`/private/tmp/SPECTRUM-FITTING-001-A-complex-2LPa2aaY/`. Every Python probe/test
installs category/message/filename/line-only warning rendering and exception-only
rendering before imports. Pytest uses `-p no:warnings --tb=no -p no:cacheprovider
-q -ra --capture=tee-sys -o addopts=`. The diagnostic plugin retains exception
type/message, case outcomes and process status without source or traceback.
Recorded warnings and the existing LM subprocess retain their own source-free
rendering. Pipeline exit status is preserved with `set -o pipefail`.

Exact corrected test SHA-256:
`50bb168312355b5c3035f534195260f4a0081a6f171cdd6a6de52235774837dd`.
The report's own final hash is returned outside the tracked tree.

The whole-spectrum command, from the authorized worktree, is:

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 SPECTRUM_A10_RESULTS=/private/tmp/SPECTRUM-FITTING-001-A-complex-2LPa2aaY/spectrum-results.json MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/mplconfig PIP_CACHE_DIR=/private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/pip-cache /private/tmp/dphtools-spectrum-clean-env-w0q0h1ko/python/bin/python3 /private/tmp/SPECTRUM-FITTING-001-A-complex-2LPa2aaY/launch.py tests/test_spectrum_fitting.py 2>&1 | tee /private/tmp/SPECTRUM-FITTING-001-A-complex-2LPa2aaY/spectrum-tests-final.log
```

Black uses the same interpreter and command-local variables, with
`format.py --check --line-length 99 tests/test_spectrum_fitting.py` and
`black-check-final.log`. Exact preservation/case outcomes use `launch.py
test_preservation.py --confcutdir=/private/tmp/SPECTRUM-FITTING-001-A-complex-2LPa2aaY`,
with absolute evidence paths, retaining `preservation-final.log`,
`final-identity.json` and `outcome-summary.json`. Final results follow below.

| Check | Actual result | Retained evidence |
|---|---|---|
| Initial independent snapshot proof | 38 passed in 0.68 s; exit 0 | `observer-proof.log`, six `unpatched-*.json` mathematical witnesses |
| Corrected live observer and discriminating negatives | 44 passed in 0.70 s; exit 0; no skips/warnings; no product import | `live-observer-proof.log`, `test_observer_proof.py` |
| Exact whole spectrum | 284 passed, 1 failed in 1.62 s; pytest/shell exit 1 | `spectrum-tests-final.log`, `spectrum-results.json` |
| Exact Black 99 | One file would be left unchanged; exit 0 | `black-check-final.log` |

The sole spectrum failure remains
`test_s29_s31_public_numerical_backend_failure[optimizer_lstsq]`, with
`LinAlgError: Numerical decomposition did not converge (injected)` at the valid
finite `(201,3)` analytic Jacobian and independently known residual. This is
**product defect — leaked custom-optimizer numerical exception**, high confidence,
subject to final independent B review. Neither covariance observer correction
explains or weakens that red evidence. A did not repair production code.

The covariance case reaches the finite `(201,3)` boundary and returns RuntimeError;
the success alternative remains independently checked. The suite retains 14
RuntimeWarnings and three OptimizeWarnings; the real LM subprocess exits 0.
There are no errors or skips. Accepted-case membership/outcomes and preservation
are recorded by the final source-free preservation check outside the tracked
candidate. Its assertions require all 276 accepted cases to pass, unchanged
accepted test prefix/AST, exact equality to the A9 preimage except the two moved
assertions, and the A9/rejected/accepted report preimages as exact prefixes.
All 45,853 accepted test bytes, all 125,173 A9 report bytes, all 111,341 rejected
report bytes and all 76,378 accepted report bytes are retained. The final report
and test hashes are in `final-identity.json` and the handoff conversation.

Environment: supplied clean non-Conda CPython 3.12.14, NumPy 2.5.3, SciPy 1.18.1,
pytest 9.1.1, Black 26.5.1; interpreter and base prefixes identify the supplied
clean Python. Only command-local supplied environment values and task-local
scripts were used. No environment/config/rules change occurred. Full coverage
and platform-matrix verification were neither run nor claimed. No local
counterexample prescribes an internal dtype, decomposition or covariance algorithm.

Read only the authorized public inputs/snapshots, marker and permitted live
test/report after exact hash verification, the opened NumPy reference, and this
session's own evidence. Directory filename inventory was used for permitted
input discovery; no forbidden file content was read. No product implementation,
private helpers/source/history/diffs, coverage, other preexisting tests/oracles,
PROJECT.md, task state/role packets, lessons or role transcripts were inspected.
Only the authorized two tracked files changed. No Superpowers, memory,
delegation, commit/push/PR, full verification, merge or release occurred.
**Source exposure: none observed.** Diagnostics contain category/message/file/line
and exception types/messages, without product source excerpts or traceback.

Metrics: 33 unchanged scenarios / 285 unchanged public cases / 0 added cases;
1 fresh independent root A launch (10 known), 0 reviewer/C launches in this
session. Same post-repair B window: round1/2 rejected, combined final round2/2
next. Initial C and original repair1/1 remain spent; the owner explicitly approved
one additional repair for the valid custom numerical exception defect. Partial
measured authoring/proof/sequential-check interval: 225 s, 19:54:17–19:58:02 UTC
on 2026-10-05, excluding earlier input/reference reading and final report/hash
completion. Prior recorded 1,970 s plus this interval totals 2,195 s of partial
measured agent spending; concurrent effort is not elapsed wall time. All local
attempts remain retained. No fixed owner time/token cap or new cap was supplied;
exact model-token usage/billing remain unavailable. No allowance or spending reset
is claimed. Fresh independent combined B round2 is the next handoff; A grants no
acceptance and launches no reviewer or product repair.
