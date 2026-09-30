### 2026-09-28-SETUP-001-B-plot-oracle | Account for valid rendering forms
- Status: confirmed
- Observation: a proposed 1D registration test declared a figure empty after checking lines, collections, and patches, but omitted its finite image artist.
- Evidence: independent B reproduced the missed image through the public figure API in `reports/B-public-boundaries-completion/`; A withdrew its product-defect claim and corrected only the new oracle.
- Lesson: a negative assertion about visible output must account for valid alternative representations. Missing a representation in the test is not evidence that the product omitted its output.
