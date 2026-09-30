### 2026-09-28-SETUP-001-B-format-oracle | Validate notation as well as value
- Status: confirmed
- Observation: the proposed LaTeX oracle accepted an ungrouped signed exponent and numerals concatenated without multiplication, then reconstructed the intended value anyway.
- Evidence: independent B review reproduced both false positives in `reports/B-public-boundaries/latex-oracle-probe.json`; its follow-up accepted the corrected grammar after 34 distinguishing checks and an unchanged-input test rerun.
- Lesson: a numerical-formatting test must validate the output language's syntax before interpreting its value. Otherwise the oracle can silently repair invalid output and let a defect pass.
