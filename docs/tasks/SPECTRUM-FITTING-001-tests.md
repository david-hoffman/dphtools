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
