# Release Notes - knowhere-rs-non-gpu-production

## [Unreleased]

### Added
- Initial long-task-codex scaffold
- Durable design doc for the non-GPU remote-x86 long-task workflow
- Durable implementation plan doc for initializing the long-task workflow
- First-pass feature inventory covering baseline, HNSW, IVF-PQ, DiskANN, production, and final acceptance
- Remote-first worker guide and bootstrap entrypoint
- Automated bootstrap test for `init.sh` injection hooks

### Changed
- Long-task workflow now treats the existing remote x86 machine as the sole authority for verification and performance claims
- Future Codex sessions now default to autonomous continuation and should only interrupt for real blockers or scope-changing decisions
- Remote bootstrap sync now excludes heavyweight local-only directories from rsync so Session bootstrap stays fast and predictable

### Fixed
- `baseline-remote-bootstrap` now passes with a real remote bootstrap, standalone sync, config summary, and automated injection test
