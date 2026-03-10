# Release Notes - knowhere-rs-non-gpu-production

## [Unreleased]

### Added
- Initial long-task-codex scaffold
- Durable design doc for the non-GPU remote-x86 long-task workflow
- Durable implementation plan doc for initializing the long-task workflow
- First-pass feature inventory covering baseline, HNSW, IVF-PQ, DiskANN, production, and final acceptance
- Remote-first worker guide and bootstrap entrypoint

### Changed
- Long-task workflow now treats the existing remote x86 machine as the sole authority for verification and performance claims
- Future Codex sessions now default to autonomous continuation and should only interrupt for real blockers or scope-changing decisions

### Fixed
- None yet
