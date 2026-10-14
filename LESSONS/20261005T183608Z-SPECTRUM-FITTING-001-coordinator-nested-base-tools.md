# A copied virtual environment can retain an unsuitable base interpreter

- ID: 20261005T183608Z-SPECTRUM-FITTING-001-coordinator-nested-base-tools
- Date: 2026-10-05T18:36:08Z
- Task/role: SPECTRUM-FITTING-001 / coordinator
- Status: confirmed
- Observation: A copied locked Python 3.13 environment passed dependency and copied-nested preflight checks, but real system-site nested environments inherited the original Conda base's Black 23.3.0 and unaudited startup inputs. The required Black version was 26.5.1.
- Evidence: Candidate dd31c2c237ac119d1ba1a3008f6b16e7e48b41f8 full receipt full-w9crt1ji reported 18 verification-reuse failures, including 12 Black version mismatches. The unchanged verification-reuse tests passed all 59 cases on a disposable non-Conda CPython 3.12.14 base with the same applicable hashed lock; tools/verification_reuse.py measured 78/78 statements and 46/46 branches, without exclusions. A real nested system-site environment inherited pinned tools and installed/imported a wheel.
- Lesson: Before an expensive full run, verify the actual base interpreter, its startup inputs, and the standard nested-environment creation path used by tests. Matching the selected environment's pins and a copied-nested probe does not establish those properties. Preserve the original environment and fix a disposable verification environment rather than weakening cache eligibility or version checks.
