# Fixture Policy

Prefer small synthetic fixtures over large binary files.

New binary fixtures require metadata:

```yaml
name: <fixture file>
created_by: synthetic | hardware | external
creator: <person/tool>
generator: <path/to/script or explanation>
seed: <value or not applicable>
units: <units or not applicable>
license: <license or internal>
sha256: <hash>
expected_behavior:
  - <claim>
```

Large fixtures require a size budget and provenance note.

Do not update golden files in the same pull request as implementation changes unless the issue explicitly requires an oracle update and a reviewer approves it.
