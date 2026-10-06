# Keep worker scratch outside the checkout

- ID: 20261006T145753Z-TEST-SUITE-PERFORMANCE-001-author-worker-scratch
- Date: 2026-10-06T14:57:53Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: A private worker temporary directory inside its checkout still violates a genuine clean-install isolation assertion. On this host, the checkout also uses overlay storage while the operating-system temporary directory uses tmpfs.
- Evidence: Candidate `d28bcd5` failed `test_actual_probe_measurement_and_real_failure_controls` with “Disposable environment must be outside checkout”; its interrupted full receipt is retained under `reports/verification/performance-candidate/`.
- Lesson: Keep worker pytest and operating-system scratch directories outside the checkout, clean them after execution, and retain logs, receipts and raw coverage separately. Measure storage and base-interpreter scope when assessing suite performance.
