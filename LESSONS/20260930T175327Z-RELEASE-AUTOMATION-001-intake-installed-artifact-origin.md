# Installing a wheel does not establish that tests import it

- ID: 20260930T175327Z-RELEASE-AUTOMATION-001-intake-installed-artifact-origin
- Date: 2026-09-30T17:53:27Z
- Task/role: RELEASE-AUTOMATION-001 / intake investigation
- Status: confirmed
- Observation: The canonical verifier installs the built wheel but launches module/test discovery with the repository root as its working directory. Source imports can shadow the installed distribution. The current gate does not assert installed-artifact import origin.
- Evidence: Baseline `e557c7d`; `tools/verification.py:140`, `tools/verification.py:147`, and `tools/verification.py:188`. No claim is made that a specific built distribution is broken.
- Lesson: Supplement checkout tests with clean-environment installation checks outside the source tree, asserting import locations and exercising the packaged public interfaces.
