# SPECTRUM-FITTING-001 public contract

**Version 1.0. Contract revision R3 — draft pending owner approval.**
This is the public behavior packet. It contains no implementation or task status.

## Purpose and scope

Add one public function for a finite real one-dimensional spectrum. Fit positive
Gaussian, Lorentzian, or Voigt components jointly, using either detected local
maxima, supplied centers, or supplied full parameters. All components in one call
use the selected profile family. Support no background, a constant, or a line.

Reuse the existing NumPy/SciPy dependencies and Python >=3.8 package metadata.
Reuse existing fitting functions where they satisfy this contract. If they are
deficient for this task, add new fitting functions or helpers. Do not change
existing public signatures or behavior. New additive interfaces are allowed.
Weighted/Poisson objectives, absorption dips, mixed profile families, plotting,
batch spectra, automatic information-criterion model selection, and release
publication are outside this slice. A new additive function is the release-note
impact.

## Public interface

```python
from dphtools.utils.fitfuncs import spectrum_fit

result = spectrum_fit(
    data, xdata=None, *, peak_type="gauss", guesses=None,
    background="constant", prominence=None, distance=None, max_nfev=10000,
    optimizer="lm",
)
```

- `data`: a nonempty real finite 1-D array-like of sample values. It is not mutated.
- `xdata`: optional real finite 1-D coordinates matching `data`, strictly
  increasing. Nonuniform spacing is allowed. Omission means sample indices
  `0, ..., len(data)-1`. Sample values are point observations, not bin integrals.
- `peak_type`: `"gauss"`, `"lorentz"`, or `"voigt"`, case insensitive. Gaussian and
  Lorentzian are aliases `"gaussian"` and `"lorentzian"`. The correct name is Voigt.
- `guesses=None`: detect local maxima, estimate starting parameters, and jointly
  fit the resulting components and background.
- A nonempty 1-D `guesses` sequence supplies centers in x units. Estimate the
  other starting parameters without detecting or adding components.
- A nonempty 2-D `guesses` sequence supplies one full parameter row per component:
  Gaussian `(amplitude, center, sigma)`, Lorentzian `(amplitude, center, gamma)`,
  or Voigt `(amplitude, center, sigma, gamma)`. Fit exactly this many components.
- Initial amplitudes and widths must be strictly positive and finite; centers
  must be finite but may lie outside the observed x interval. Guesses are not
  mutated.
- `background`: `"none"`, `"constant"`, or `"linear"`. Default is constant.
  A line is `b0 + b1*(x-xdata[0])`; `b0` is the value at the first coordinate.
- `prominence`: optional finite nonnegative minimum prominence in data units.
  `distance`: optional finite value >=1, interpreted as minimum separation in
  samples. These controls apply only when guesses are absent; explicit guesses
  combined with either control raise `ValueError`, rather than ignoring it.
- `max_nfev`: a positive integer optimizer evaluation limit. Exhaustion is a fit
  failure. Counting follows the selected optimizer; this is not a uniform limit
  on every model evaluation, including covariance calculations.
- `optimizer`: `"lm"` by default, or a custom callable with the protocol below.
  Other optimizer choices can be supplied as callables or configured adapters.

`result` exposes these public attributes:

| Attribute | Meaning |
|---|---|
| `peak_parameters` | 2-D array of rows in the above profile order, sorted by fitted center |
| `background_parameters` | Empty array, `(b0,)`, or `(b0, b1)` for none/constant/linear |
| `covariance` | Approximate full parameter covariance, ordered as flattened sorted peak rows followed by background coefficients |
| `fitted` | 1-D fitted spectrum at the input coordinates, including background |
| `residuals` | 1-D array `data - fitted` |

Sorting must reorder covariance consistently, including cross-component and
background correlations. Returned array storage must not modify caller inputs.

## Numerical convention and objective

For `d=x-center`, use unit-height profiles:

- Gaussian: `G(d) = exp(-d**2/(2*sigma**2))`.
- Lorentzian: `L(d) = gamma**2/(d**2+gamma**2)`.
- Voigt: `P(d) = V(d; sigma, gamma)/V(0; sigma, gamma)`, where `V` is the
  normalized Gaussian/Cauchy convolution defined by SciPy's `voigt_profile`.

The model is `background(x) + sum(amplitude_i * profile_i(x-center_i))`.
Each profile equals one at its center, so `amplitude` is the individual
component's peak height in data units, not integrated area. For overlapping
components, the total spectrum at a component's center also includes the other
components and background. Centers and widths have x units; Gaussian `sigma` is
standard deviation and Lorentzian `gamma` is half width at half maximum. Gaussian
full width at half maximum is `2*sqrt(2*log(2))*sigma`; Lorentzian full width is `2*gamma`.
Voigt has two independent width parameters. Both are strictly positive in this
first fitting interface; select Gauss or Lorentz for the pure limiting family.

Fit unweighted nonlinear least squares: minimize the sum of squared residuals
at the supplied samples. Nonuniform x spacing does not introduce integration
weights. Amplitudes and widths remain strictly positive. Centers and background
coefficients are unconstrained, including centers outside the observed x interval.

Covariance follows the local linear approximation with residual-variance scaling
by residual sum of squares divided by `number_of_samples-number_of_parameters`.
Require more samples than fitted parameters. Covariance is an approximate
uncertainty estimate, not a guarantee of identifiability or a global optimum.
Preserve applicable numerical warnings; do not fabricate finite uncertainties.
Numerical test tolerances are chosen independently by A and reviewed by B.

## Optimizer interface

The default is SciPy's Levenberg-Marquardt solver through
`scipy.optimize.least_squares(..., method="lm")`. It minimizes the stated
unweighted residual objective. The public fitting result and peak-height units
remain the same when a different optimizer is supplied.

A custom callable has this signature:

```python
optimizer(residual, initial, *, bounds, max_nfev) -> optimizer_result
```

- `residual(parameters)` returns `data - model(parameters)` as a 1-D array.
- `initial` and the returned `optimizer_result.x` use physical parameters, not
  private transformed coordinates: flattened peak rows followed by background
  coefficients. Parameter ordering stays fixed throughout the optimizer call;
  the fitting result is subsequently sorted by center.
- `bounds` is a pair of lower/upper bound arrays in the same order and units:
  amplitude/width lower bounds are zero and their upper bounds are infinity;
  centers/background coefficients have infinite bounds in both directions.
  The physical domain requires strictly positive amplitudes/widths, so a final
  zero value is invalid even though zero marks their bound. The custom optimizer
  must respect this domain and minimize the supplied residual sum of squares.
  Configure additional solver options with a wrapper or `functools.partial`.
- `max_nfev` is forwarded to the callable, which must honor its evaluation limit.
- The returned object must expose a finite real 1-D `x` of the correct length
  and a Boolean `success`. A `message` is optional. A failed convergence status,
  malformed result, or final parameters outside the physical domain produces
  `RuntimeError`; caller input arrays remain unchanged.
- The fitter calculates covariance from the residual Jacobian in physical
  coordinates at the final solution. A custom optimizer need not return its own
  Jacobian or covariance. Sorting still reorders the full covariance.

SciPy's LM method does not accept explicit bounds. The default path therefore
uses internal parameter transforms for strictly positive amplitudes and widths.
Centers remain free, including outside the observed interval. Returned parameters
and covariance remain in physical units. Transformation details are internal
implementation choices; they are not part of the custom-optimizer protocol.

## Discovery and failure behavior

Automatic discovery supplies local-maximum candidates. It does not promise
identification of endpoints, unresolved overlapping components, or every physical
peak in noisy data. Prominence and distance let callers control candidate count;
full or center guesses supply components without local maxima. Starting estimates
and internal optimization choices remain delegated.

Raise `ValueError` for invalid inputs/options, insufficient observations, or no
eligible detected peaks. Raise `RuntimeError` when fitting fails or reaches the
evaluation limit. Do not return a failed fit as a successful result. Do not
silently drop nonfinite samples or return a baseline-only fit for no peaks.

## Expectation sources

- Owner request on 2026-10-05: 1-D spectrum, automatic discovery or supplied
  guesses, and selectable Gauss/Lorentz/Voigt.
- Owner clarification on 2026-10-05: positive peaks; selectable constant/linear
  background; centers and full parameters; optional x, joint least squares,
  parameters/covariance/fitted values/residuals.
- Owner correction on 2026-10-05: amplitude means component peak height, not
  integrated area. This convention is settled.
- Owner clarification on 2026-10-05: preserve existing public interfaces; add new
  fitting functions/helpers when existing functions are deficient. Use a default
  Levenberg-Marquardt optimizer and allow a custom optimizer.
- Owner constraint choice on 2026-10-05: positive amplitudes/widths, with freely
  moving centers; retain positivity through internal transforms for default LM.
- Proposed width conventions, schema, custom-optimizer protocol, and failure
  choices above need owner approval of this R3 before test authoring or
  implementation.
- [SciPy 1.15.3 Voigt definition](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.voigt_profile.html)
  gives the normalized Gaussian/Cauchy convolution and sigma/gamma convention.
- [SciPy peak discovery](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.find_peaks.html)
  defines local maxima, prominence, sample-distance controls, and limitations.
- [SciPy measured peak widths](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.peak_widths.html)
  distinguishes half prominence from model full width at half maximum.
- [SciPy least-squares/covariance convention](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.curve_fit.html)
  describes the objective, scaling, covariance approximation, and failures.
- [SciPy optimizer methods and result interface](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html)
  defines LM's bounds limitation, residual objective, result status, and
  method-dependent evaluation counts.
- Invariants: unit-height profile normalization, positive components, covariance
  ordering, input preservation, and `residuals == data-fitted`.

## Contract scenarios

This is one coherent fitting slice with 33 explicit scenarios, exceeding the
five-scenario default. Owner approval of the size exception is pending. Profile
and data variants within a row may exercise that same approved behavior. A must
supply public-entry-point mappings and justified tolerances; B reviews both.

| ID | Input/context and observable outcome | Expectation source |
|---|---|---|
| S01 | Full Gaussian guesses recover a well-conditioned positive component model and correctly shaped outputs | Gaussian model/objective |
| S02 | Full Lorentzian guesses recover a well-conditioned positive component model | Lorentzian model/objective |
| S03 | Full Voigt guesses recover a well-conditioned model with both widths positive | Voigt model/objective |
| S04 | Supplied centers estimate other starts and jointly fit exactly the requested components, including identifiable overlap | Owner guess mode; objective |
| S05 | Automatic mode fits separated eligible local maxima with discovery controls and sample-index x default | Owner discovery mode; SciPy discovery |
| S06 | Constant background is fitted jointly and its coefficient/covariance are represented correctly | Approved background model |
| S07 | Linear background is fitted jointly in the stated coordinate convention | Approved background model |
| S08 | No-background mode adds no coefficient and returns the correct fitted sum | Optional background |
| S09 | Physical/nonuniform coordinates preserve stated center, amplitude, width, and covariance units; unsorted guesses return consistently sorted rows | Units and covariance ordering |
| S10 | Non-1-D data is rejected with ValueError | Data shape |
| S11 | Nonfinite spectrum samples are rejected with ValueError | Finite sample domain |
| S12 | Complex spectrum samples are rejected with ValueError | Real sample domain |
| S13 | Coordinate dimension or length mismatch is rejected with ValueError | Coordinate shape |
| S14 | Nonfinite or complex coordinates are rejected with ValueError | Real finite coordinate domain |
| S15 | Duplicate or decreasing coordinates are rejected with ValueError | Strict coordinate ordering |
| S16 | Empty/malformed/wrong-width guess arrays are rejected with ValueError | Guess schema |
| S17 | Nonfinite or complex guess values are rejected with ValueError | Real finite parameter domain |
| S18 | Nonpositive initial amplitudes are rejected with ValueError | Positive initial amplitude |
| S19 | Nonpositive initial widths are rejected with ValueError | Positive width convention |
| S20 | Finite initial and fitted centers may lie outside the observed interval; an identifiable edge-tail model can be fitted without center bounds | Owner free-center choice |
| S21 | Unsupported profile selection is rejected with ValueError | Profile choices |
| S22 | Unsupported background selection is rejected with ValueError | Background choices |
| S23 | Invalid prominence is rejected with ValueError | Discovery control domain |
| S24 | Invalid distance is rejected with ValueError | Discovery control domain |
| S25 | Explicit guesses combined with discovery-only controls are rejected with ValueError | No ignored options |
| S26 | No eligible automatic peaks produces ValueError | No-peak outcome |
| S27 | Empty data or too few samples for positive residual degrees of freedom produces ValueError | Sample count and covariance scaling |
| S28 | Invalid evaluation limit produces ValueError | Positive integer limit |
| S29 | Exhausted/nonconvergent optimization produces RuntimeError and preserves caller inputs | Fit failure outcome |
| S30 | Omitted optimizer selects Levenberg-Marquardt, retains positive amplitudes/widths, and leaves centers free | Owner optimizer default and constraints |
| S31 | A supplied custom callable receives physical residuals, initial values, bounds, and the evaluation limit; a successful fit returns the same result/covariance convention | Owner optimizer override; callable protocol |
| S32 | A noncallable optimizer other than the supported default is rejected with ValueError | Optimizer selection domain |
| S33 | Malformed or physically invalid custom optimizer output produces RuntimeError without mutating caller inputs | Optimizer result contract |

## Delivery constraints supplied to roles

Use the specialist A/B/C/D route because this introduces a new scientific fitting
contract and independently checked numerical expectations. Blind A/B may read
this contract, their narrow packet, the applicable skill, permitted public
instructions/test conventions, and these references. They must not inspect
product implementation/history, task execution state, other role transcripts,
or implementation-bearing project records/lesson entries.

A authors meaningful public-function tests. B reviews their expectation sources,
tolerances, and failure discrimination. Record the accepted checkpoint and valid
baseline evidence before C. C edits only approved product code/docstrings and
preserves reviewed tests, fixtures, configuration, workflows, and delivery rules.
D independently reviews the exact passing candidate and real check evidence.

Default allowances are two A/B review rounds and initial C plus one corrective C
cycle. No fixed owner time/token budget has been supplied; record measurable
usage and missing billing rather than claiming a new unlimited allowance. A scope
or allowance extension requires owner approval. Preserve exact global/per-file
100% statement and branch coverage, including subprocesses and never-imported
owned runtime. Full local reference and protected Linux/macOS/Windows CI are
required before submission/merge under the governing instructions. Merge needs
separate owner authorization; no release publication is authorized.
