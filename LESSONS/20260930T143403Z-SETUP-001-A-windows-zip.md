### 2026-09-26-SETUP-001-A-windows-zip | Native fixture portability
- Status: confirmed
- Observation: the external fake harness passed locally but its Windows console launcher failed before any doctor behavior ran. Appending a ZIP directly made its offsets include the executable prefix; the native launcher expects archive-relative offsets.
- Evidence: Actions run 36281726862 at f1609ca had 39 Windows fixture setup errors. Fresh A/B reviewed correction cfdea14, which builds the ZIP separately and concatenates it with the preserved launcher prefix. Run 36282672655 then passed all 39 doctor cases on Windows, Ubuntu, and macOS. No runtime or behavior assertion changed.
- Lesson: validate native platform fixtures on their real target and retain independent fixture self-checks. Python-readable ZIP structure alone does not prove a native launcher can find its payload. Fixture setup errors are not meaningful product red evidence.
