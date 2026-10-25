# Changelog

All notable user-facing changes to mnemotree are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [PEP 440](https://peps.python.org/pep-0440/).

Add notes under **Unreleased** as changes land. The `prepare release`
workflow moves them into a dated section and refuses to run while
Unreleased is empty.

## [Unreleased]

### Added

- Each GitHub release ships a `constraints.txt` with the exact dependency
  resolution the release was tested against:
  `pip install mnemotree==X.Y.Z -c constraints.txt`.
- Python 3.14 is part of the tested matrix.

### Changed

### Fixed

### Removed

[Unreleased]: https://github.com/kurcontko/mnemotree/commits/main
