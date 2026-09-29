# Real-signal LPSVD baseline contract

**Version 1.0** Narrow source-free packet for the owner's 2026-09-28 request to validate LPSVD and its tests against the original mathematics, within the already approved baseline-test scope. Product implementation and research findings are deliberately absent.

Use explicit model order and real, nondegenerate damped sinusoidal signals. Sample index `n` is dimensionless. A component `A*exp(-d*n)*cos(2*pi*f*n+phi)`, with `A>0`, `d>0`, and `0<f<0.5`, is the sum of two conjugate exponential terms. Each has amplitude `A/2`, damping `-d` per sample, frequencies `+f` and `-f` in cycles per sample, and phases `+phi` and `-phi` modulo `2*pi` radians. The total model order counts exponential terms, twice the number of distinct real cosines. Do not choose zero/Nyquist frequencies or indistinguishable components.

This oracle follows from Euler's identity and the uniformly sampled damped-exponential model in Kumaresan and Tufts, *Estimating the Parameters of Exponentially Damped Sinusoids and Pole-Zero Modeling in Noise* (1982), [original paper](https://www.math.ucdavis.edu/~saito/data/sonar/KumaresanTufts.pdf), equations (1)–(4), pp.833–834. These equations give a backward-prediction matrix of `N-L` rows and `L` columns, with `L=floor(N*lfactor)`. The method needs enough rows and columns for the explicit signal rank. Use strictly more singular values than the model order when exercising noise-bias removal. Rank constraints do not require an even sample count or `L=N/2`.

Public functions already supplied by the library:

```python
LPSVD(signal, M=None, lfactor=1/2, removebias=True)
reconstruct_signal(LPSVD_coefs, signal, ampcutoff=0, freqcutoff=0, dampcutoff=0)
```

`LPSVD` returns a dataframe whose coefficient columns are `amps`, `freqs`, `damps`, and `phase`. Additional error fields do not establish uncertainty calibration and are outside this packet. `reconstruct_signal` evaluates coefficients on the sample indices represented by its `signal` argument. For this packet, use real floating templates and leave all cutoff arguments at zero. More sample indices may be supplied to test an independently known continuation.

Required useful additions: compare fitted coefficients directly with analytic expected terms; include odd sample counts and `L` below and above `N/2`; independently evaluate held-out samples; check reconstruction separately using supplied known coefficients. Compare phases modulo `2*pi` and match terms by frequency rather than row order. Choose numerically stable examples and justify tolerances. Preserve existing tests. A round trip using two production functions is supplementary, not the sole correctness oracle.

Automatic order selection, unsupported or ambiguous input policies, complex-signal support, parameter-error calibration, and statistical denoising/bias performance are not decided by this packet. Do not invent their contracts, add arbitrary tolerances, mock the numerical algorithm, or call private helpers solely for coverage. Return unresolved behavior to the coordinator.
