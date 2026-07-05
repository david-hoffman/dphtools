# Downstream Compatibility Policy

No downstream projects or consumers were discovered during repository intake.

Current policy:

- `scripts/agent_harness/downstream_smoke.py` passes with an explicit no-known-consumer message.
- If downstream consumers are later identified, add a configured smoke target before treating downstream compatibility as covered.
- Breaking downstream behavior requires one of these outcomes:
  - Fix compatibility in this repository.
  - Open coordinated downstream pull requests.
  - Document a deliberate breaking change with versioning and release notes, then require human/admin approval.

For a future library downstream smoke target:

1. Build a local package artifact.
2. Create an isolated environment.
3. Install the downstream project with the local artifact.
4. Run a minimal smoke subset.
5. Record logs and versions.
