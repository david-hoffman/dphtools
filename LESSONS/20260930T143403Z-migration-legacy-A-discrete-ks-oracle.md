## 2026-09-29 — independently reviewed numerical test lessons

A, exact-discrete likelihood and KS oracle: A discrete KS oracle must evaluate unobserved integer gaps as well as observed atoms. For exact-discrete PowerLaw fitting of `[2,3,8]` at cutoff 2, an independent infinite-support likelihood calculation gives alpha approximately 2.20349903891 and the largest ordinary KS difference at integer 7, approximately 0.183068916435. Tests restricted to observed integers can miss that supremum.

Evidence: `reports/A-powerlaw-cutoff/handoff-before-near-undamped.md`, `reports/B-powerlaw-cutoff/review.md`, `additions-review.md` and `final-review.md`. These are reusable observations, not policy or permission to change behavior.
