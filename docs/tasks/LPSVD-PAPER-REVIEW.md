# LPSVD paper review

**Version 1.0** Research evidence from 2026-09-28 against product candidate `8ca6cf2`. This is not delivery role D approval. No numerical repair is authorized by a paper alone.

## Sources actually read

The exact module-header citation is Tufts and Kumaresan, *Singular Value Decomposition and Improved Frequency Estimation Using Linear Prediction*, IEEE ASSP 30(4), 671–675 (1982), DOI `10.1109/TASSP.1982.1163927`. The complete article text was read through [Kumaresan's uploaded copy](https://www.researchgate.net/publication/3177158_Singular_value_decomposition_and_improved_frequency_estimation_using_linear_prediction); some OCR matrix symbols are degraded. The [University of Rhode Island record](https://digitalcommons.uri.edu/ele_facpubs/740/) confirms the citation. Its model concerns unit-magnitude sinusoidal modes.

The directly applicable damped-signal reference is Kumaresan and Tufts, *Estimating the Parameters of Exponentially Damped Sinusoids and Pole-Zero Modeling in Noise*, IEEE ASSP 30(6), 833–840 (1982), DOI `10.1109/TASSP.1982.1163974`. Its [original PDF hosted by UC Davis](https://www.math.ucdavis.edu/~saito/data/sonar/KumaresanTufts.pdf) was read in full, with equations (2)–(4) visually checked. Local PDF SHA-256: `e261414613954d94a05c738f6db9ccaefa0f118b4046a154e7da37056d8fc099`.

The separately cited Barkhuijsen papers from [1985](https://www.sciencedirect.com/science/article/abs/pii/0022236485901878) and [1986](https://www.sciencedirect.com/science/article/abs/pii/0022236486904464) remain inaccessible in full after publisher, author, and repository searches. Their publisher records were checked; their full contents and uncertainty formulas were not validated. The PDFs are not republished in Git.

## What the Hankel edit establishes

The December paper's equation (2), p.833, requires

\[
A_{ij}=\overline{s_{i+j+1}},\quad h_i=\overline{s_i},
\quad 0\leq i<N-L,\quad 0\leq j<L.
\]

The corrected `hankel(rollsig[:N-L], signal[N-L:])`, followed by conjugation, has exactly those entries and dimensions. The old second argument gave the wrong width except when `N=2L`.

The root convention is also consistent. Reversing `[1,b1,...,bL]` for `np.roots` solves `Q(q)=1+b1*q+...+bL*q**L`, where `q=1/z` relative to the paper's polynomial. A mode `s_n=a*exp(lambda*n)` has `q=exp(conj(lambda))`, so `conj(log(q))` recovers its exponent modulo sampling aliases. Damped modes therefore have these reciprocal roots inside the unit circle. The implementation's comparison is appropriate; its comment about removing inside roots describes the opposite polynomial convention.

This validates the matrix indexing and reciprocal-root conversion, not automatic order selection, denoising performance, or error calibration.

## Findings and scope

- **Confirmed default-path defect:** for `n=arange(16)` and `2*exp(-0.08*n)*cos(2*pi*0.125*n+0.2)`, `LPSVD(signal)` prints estimated order 10, clamps it to 8, averages an empty noise tail during bias correction, and raises `ValueError` after NaN coefficients. With default `L=8`, adding eight to any nonnegative estimated order guarantees exhaustion of the singular values. This does not require assuming the estimator must return exactly two. The `+8` heuristic is not justified by the original papers read here; replacing order selection needs its own supported contract.
- **Complex behavior and unresolved support:** the public fitter raises `UFuncTypeError` during uncertainty calculation for the December paper's complex two-exponential example. Separately, public reconstruction with a `complex64` template discards the imaginary component; a `complex128` template does not. No explicit real-only API promise was found, but this evidence alone does not approve expanding the supported model or inventing a complex-noise uncertainty rule.
- **Tests are useful, but incomplete:** all nine existing LPSVD tests pass. The eight reconstruction cases use explicit order, even length 48, real signals, and the package's own reconstruction function. Paired convention errors could cancel. The remaining test checks finite/nonnegative error fields, not calibrated uncertainty. These are gaps in what the tests establish, not evidence that their existing assertions are wrong.

## Independent mathematical tests authorized by existing real-signal scope

The owner's request to validate the tests against the paper permits additional baseline tests of the existing real damped-sinusoid behavior. For sample index `n`, a component `A*exp(-d*n)*cos(2*pi*f*n+phi)` with `A>0`, `d>0`, and `0<f<0.5` has two exponential terms. Each has amplitude `A/2`, damping `-d` per sample, frequencies `+f` and `-f` in cycles per sample, and phases `+phi` and `-phi` modulo `2*pi` radians. This follows from Euler's identity and the cited model, independently of the implementation.

Fresh A/B should check those coefficients and independently evaluated held-out samples for explicit, identifiable model order; include odd sample counts and Hankel lengths below and above half the sample count. Direct reconstruction with analytically supplied coefficients tests its convention separately. Do not silently add complex support, prescribe an MDL replacement, calibrate errors, or change existing assertions. No noise-free coefficient test establishes noisy estimator performance.

The source-reading audit helper ran separate diagnostic probes, including one process that bypassed uncertainty calculation to isolate core pole estimation. Those probes are research evidence, not public end-to-end success, independent acceptance tests, or a committed test checkpoint. Detailed retained evidence is under ignored `reports/lpsvd-paper-audit/`.

## Supplemental tests and sensitivity check

Fresh A `01a0e8c7-a2e7-7013-ad39-ce1c770a7e59` added four cases in `tests/test_lpsvd_paper.py`. Fresh B `01a0e8d0-f236-70f2-abcf-d808d0fdb39d` independently accepted them and passed all four new plus nine existing LPSVD cases, with zero warnings. Both had memory/delegation disabled and reported no implementation exposure; each disclosed an over-broad excerpt of adjacent public documentation. Neither is claimed perfectly instruction-compliant or a final candidate review.

The tests compare literal known coefficients and independent held-out real samples for `N=63` with `lfactor=1/3`, `1/2`, and `2/3`; a separate case reconstructs supplied analytic coefficients without fitting. As a coordinator sensitivity check, a disposable source copy passed all four tests, while restoring only the old Hankel slice in that copy caused all three fitted-coefficient cases to fail; direct reconstruction still passed. The imported copy's location was verified. An initial fixture path check stopped before tests because a macOS temporary-directory symlink needed normalization. This is a controlled mutation check, not historical red evidence or another independent session. Product source and reviewed tests were not changed by the probe. Evidence: `reports/lpsvd-paper-audit/hankel-mutation.json`.

## Model-order follow-up

[Lin's 1998 Berkeley thesis](https://escholarship.org/content/qt50f4m6fr/qt50f4m6fr.pdf), §3.4 pp.57–59 and Appendix A pp.179–181, supplies readable primary evidence. Its MATLAB implementation minimizes the same minimum description length (MDL) criterion over `k=0,...,L-1`, with no `+8`; its example uses `L=floor(N/3)`. The separate LPSVD routine uses discarded singular values for bias compensation. This supports direct MDL for positive spectra of the expected dimensions. It does not settle zero singular values, `L>N/2`, zero selected order, or explicit full-rank bias correction. Those policies remain unresolved; no numerical change was made. The complete 1997 article, its correction, and the Scharf book were not obtained.
