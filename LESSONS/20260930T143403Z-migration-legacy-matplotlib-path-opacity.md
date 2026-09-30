### Matplotlib path opacity and calibrated geometry

- A Matplotlib path can report visible geometry while painting nothing because its effective opacity is zero. When testing a calibrated drawn object, tie the paint observation to the same path whose transformed length is checked; a canvas difference from an unrelated object is insufficient. Restore artist state after the observation. Native positive/negative controls established this in `reports/B-coverage-resume/display-review-corrected.md`; the rejected observer/report remain preserved.
