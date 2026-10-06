# Exact interpreter pins also need platform and architecture availability

- ID: 20261006T164612Z-TEST-SUITE-PERFORMANCE-001-author-platform-patch-availability
- Date: 2026-10-06T16:46:12Z
- Task/role: TEST-SUITE-PERFORMANCE-001 / author
- Status: confirmed
- Observation: Python 3.10.21 existed for Linux x64 but the action had no build for macOS arm64 or Windows x64. A common pin stopped those platforms before collection.
- Evidence: CI run https://github.com/david-hoffman/dphtools/actions/runs/37497382992, setup jobs 112385313467 and 112385313663; actions/python-versions versions-manifest.json SHA-256 183f5392fbc06cd7ee039f040d132500f28b305554790aeeebea242da4b98ea6 confirms 3.10.11 availability for those platforms/architectures.
- Lesson: Verify exact version, platform and architecture together before pinning. Use the same available patch across each platform's collection, shards and aggregation; separate platform proofs need not share a patch version.
