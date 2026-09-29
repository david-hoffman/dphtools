# PowerLaw automatic lower-cutoff proposal

**Version 1.0** Status: owner-approved contract; implementation held for coverage-first sequencing. Git versions revisions.

## Recommendation and meaning

Choose the lower cutoff whose retained observations most closely match a maximum-likelihood power-law fit, measured by ordinary Kolmogorov–Smirnov (KS) distance. In plain English: try possible starting points, fit the tail above each, and keep the best match. A higher cutoff discards more observations; it does not remove large observations or impose an upper bound.

This is the approach recommended in Clauset, Shalizi and Newman, *Power-law distributions in empirical data*, §3.3, Eq. 3.9. Their Eq. 3.11 also discusses a weighted alternative resembling the current calculation; weighted KS is not inherently invalid. Choosing ordinary KS is a proposed scientific contract, not proof that every current result is wrong. [Paper, full text](https://arxiv.org/pdf/0706.1062)

Observed at baseline `8ca6cf2`: floating data with `xmin=None` use 1 without searching; integer data search consecutive integer cutoffs below the cap. The proposed float search, observed-value candidate set, inclusive cap, and minimum tail size deliberately change those behaviors. Existing code alone does not approve them.

## Proposed public contract

Scope is `PowerLaw.fit(xmin=None, xmin_max=200, opt_max=False)` and the corresponding explicit-`xmin` likelihood. Preserve the call signature, `(C, alpha)` return, and fitted `xmin`, `C`, `alpha`, and `clipped_data` interface. The following policies were approved as recorded below; they are not all prescribed by the paper.

| Choice | Proposed behavior |
| --- | --- |
| Model | Preserve the documented dtype distinction: integer arrays use an integer-valued distribution; real floating arrays use a continuous distribution, even when their values happen to be whole numbers. Samples are unweighted observations. |
| Input | Require a nonempty, one-dimensional, finite, real numeric NumPy array with nonnegative values. Zero observations remain below every eligible cutoff. Reject negative, complex, boolean, or nonnumeric data. Do not mutate the input. |
| Lower boundary | Include every observation with `x >= xmin`, including repetitions exactly at the cutoff. `xmin` must be positive and must be an integer for discrete data. An explicitly supplied cutoff need not be observed. |
| Automatic candidates | Ascending distinct positive observed values `t <= xmin_max`, each retaining at least **50 observations** and at least one observation strictly above `t`. Preserve the default cap of 200; make its boundary inclusive. The cap limits candidate cutoffs, not observation values. Require a positive finite scalar cap. |
| Fitting | For every eligible candidate, fit its conditional tail by maximum likelihood with `alpha > 1`, using the continuous or exact discrete model below. Do not round floats to integers or substitute the continuous approximation for integer likelihood. |
| Selection | Minimize the ordinary KS distance below. If several distances are within an absolute `1e-10` of the minimum, select the smallest cutoff, retaining the most observations. This tolerance is a proposed reproducibility policy, not statistical uncertainty. |
| Small samples | The 50-observation floor applies only to automatic selection. An explicit cutoff may fit a smaller nondegenerate tail; its approximate uncertainty does not certify reliability. No extra public tuning parameter is proposed here. |
| Errors | Raise `ValueError` for invalid inputs, no eligible automatic candidate, an empty explicit tail, or a tail consisting entirely of its cutoff (no finite maximum-likelihood exponent). Raise `RuntimeError` if a mathematically eligible fit cannot converge to finite parameters; do not silently drop that candidate and select another. A failed call leaves prior fitted state unchanged. |
| Observability | `ks_statistics` contains ordinary distances in ascending eligible-candidate order; explicit fitting contains one distance. The returned parameters, retained samples, and fitted state refer to the selected cutoff. |

The 50-observation floor is a proposed practical guardrail. The paper's rule of thumb concerns exponent estimation; its more demanding cutoff experiments needed approximately 1,000 tail observations. Neither number guarantees a correct model or cutoff. [Paper, §§3.2–3.4](https://arxiv.org/pdf/0706.1062)

### Models and score

For a candidate `t`, retain `n` observations `x_i >= t`. The proposed continuous model and its likelihood solution are

\[
f(x)=C x^{-\alpha},\quad
\hat\alpha=1+\frac{n}{\sum_i\log(x_i/t)},\quad
C=(\hat\alpha-1)t^{\hat\alpha-1},\quad
F(x)=1-(x/t)^{1-\hat\alpha}.
\]

This is a Pareto distribution with shape `alpha - 1`, location zero, and scale `t`. [SciPy Pareto definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html)

For integers, use the normalized infinite-support model and minimize its negative log likelihood:

\[
p(k)=\frac{k^{-\alpha}}{\zeta(\alpha,t)},\quad
\mathrm{NLL}(\alpha)=n\log\zeta(\alpha,t)+\alpha\sum_i\log x_i,\quad
C=1/\zeta(\hat\alpha,t).
\]

Here `zeta(alpha, t)` sums `k**(-alpha)` for every integer `k >= t`. Consequently, `F(k) = 1 - zeta(alpha, k+1)/zeta(alpha, t)`. Do not renormalize at the largest observed value. [Hurwitz zeta definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zeta.html)

Use `D(t) = sup |F_empirical(x) - F_model(x)|` on the retained support. For continuous samples, compare both sides of every empirical jump, including tied values. For discrete samples, compare the step distributions over integer support, including gaps with no observed counts. Distances lie in `[0, 1]`; do not divide by `sqrt(F*(1-F))`. The two-sided continuous calculation follows the [NIST KS definition](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm).

### Observable examples for a later approved A/B packet

These are proposed expected behaviors, not executed acceptance tests or measurements of the current implementation.

- **Selection:** float data consisting of 70 ones and 20 each of 2, 4, and 8 have two eligible candidates, 1 and 2. Their distances are respectively `7/13` and `1/3`. Select `xmin=2`, retain 60 observations, and return `alpha = 1 + 1/log(2)`, approximately **2.44269504**. These deliberately coarse data illustrate selection; they do not demonstrate a plausible continuous power law.
- **Cap:** the same data with `xmin_max=1` select 1; `xmin_max=2` includes and selects 2. All samples above the cap remain in the fitted tail.
- **Inclusive cutoff:** floating `[2, 3, 8]` with explicit `xmin=2` retain all three values and give `alpha = 1 + 3/log(6)`, approximately **2.67433188**. Automatic selection on these three observations raises `ValueError` under the proposed floor.
- **Boundaries:** 49 positive observations cannot supply an automatic candidate. Fifty identical observations also cannot, because their only observed candidate has no finite exponent. NaN input fails rather than disappearing from the sample.
- **Models:** integer `[2, 3, 8]` with explicit `xmin=2` use the discrete likelihood above; the continuous answer is not its oracle. A later independent test designer must derive or verify the discrete optimum separately.
- **Invariants:** permuting observations preserves the result. For continuous data, multiplying every observation and the cutoff cap by the same positive factor scales the selected cutoff by that factor and leaves `alpha` and KS distances unchanged. Integer rescaling is not promised to preserve the discrete fit.

## Limits, alternatives, and approval

Minimized distance estimates a fitting region; it is not a p-value or evidence that a power law is scientifically preferable to alternatives. Ordinary tabulated KS significance levels do not apply when parameters were fitted from the same data. [NIST limitations](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm) Bootstrap goodness-of-fit, model comparison, revised uncertainty estimates, generators, plotting, and automatic upper-cutoff selection (`opt_max=True`) need separate scope. An upper-truncated likelihood is a different model; this proposal does not authorize dropping extreme observations under an unbounded model.

The main tradeoff is the hard 50-observation floor: it prevents very small automatically selected tails but can reject exploratory datasets or exclude their best-looking small tail. The alternative is a smaller mathematical minimum with a warning; that accepts more data at the cost of easier overinterpretation. Ordinary versus weighted KS is also a choice: retain weighted KS only if extra sensitivity at distribution extremes is wanted and its boundary behavior is separately specified.

The owner requested a proposal on 2026-09-28 and explicitly deferred `fit_ztp`. The approved architecture and unlimited setup budget remain in [PROJECT.md](../PROJECT.md). The original intake changed only this proposal document and ran arithmetic checks for the examples, not the product or an acceptance suite.

In the continuation from clean checkout `1fc9b5d9ee8d3fc8bd5bffd8a8a32bafed7fa710`, the coordinator asked whether to approve this proposal as written, explicitly including ordinary KS and the 50-observation floor. The owner's exact response was: “Sure do that. Stay on this branch and PR. Get coverage to 100% first then implement the cutoff plan”. This approves the complete lower-cutoff contract above, subject to that implementation order. Remain on `codex/scientific-maintenance` and retain PR #10; do not create a replacement PR or switch branches.

Prepare source-free behavioral inputs and fresh independent A/B tests before any C implementation. Cutoff implementation remains held while coverage work proceeds. A coverage percentage with failing tests is not a passing local gate. This approval does not choose upper-cutoff selection (`opt_max=True`), bootstrap/generation, other scientific conventions, or the deferred `fit_ztp` estimator. Those remaining decisions cannot be inferred from the coverage target. The final whole-project gate and fresh D review remain required before pushing; no merge or release is authorized.
