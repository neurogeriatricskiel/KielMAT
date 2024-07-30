# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide section),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.5]

Fifth release of KielMAT (formerly known as NGMT) for JOSS publication.

### Changelog for Merge #102

### Added:

- **Dataset Management:**
  - Fetch data from OpenNeuro, specifically for KeepControl and Mobilise-D datasets ([PR #99](https://github.com/neurogeriatricskiel/KielMAT/pull/99), [PR #101](https://github.com/neurogeriatricskiel/KielMAT/pull/101)).
- **Documentation:**
  - Revised various sections: Declaration of Helsinki, data usage per module, contributing guidelines, summary, statement of need, state of the field.
  - Added a new logo to the documentation ([PR #98](https://github.com/neurogeriatricskiel/KielMAT/pull/98)).
- **Tests:**
  - Added initial tests for loading recordings and the HDF5Decoder.

### Fixed:
- **Dependencies:**
  - Removed `tqdm` as a dependency.
- **Examples and Tutorials:**
  - Updated examples to ensure they work with publicly available datasets ([PR #100](https://github.com/neurogeriatricskiel/KielMAT/pull/100)).
  - Fixed spelling errors and other minor issues.

### Other Changes:
- **Dataset Management:**
  - Initial rewrite of the datasets submodule.
- **Documentation:**
  - Updated documentation for example data and tutorials.
  - Updated basic tutorial and examples, including gait sequence detection and loading data.
- **Project Renaming:**
  - Renamed the project and associated modules to Kiel Motion Analysis Toolbox (KielMAT) and various iterations of this name ([PR #97](https://github.com/neurogeriatricskiel/KielMAT/pull/97)).

## [0.0.4] 

Forth release of KielMAT for for JOSS publication.

### Fixed
- Gait sequence detection with datetime [[#61]](https://github.com/neurogeriatricskiel/KielMAT/pull/61)

### Changed
- Reworked documentation [[#60]](https://github.com/neurogeriatricskiel/KielMAT/pull/60)

## [0.0.3] - 2024-02-27

Third unofficial release of KielMAT for testing purposes.

### Added
- Pyarrow as dependency [[ADD]](https://github.com/neurogeriatricskiel/KielMAT/commit/22e401a5519cc9adde37b5c752a361a07d8166ac)
- Testing coverage [[ADD]](https://github.com/neurogeriatricskiel/KielMAT/commit/f6a919100e7a9d7319a4af77592a78bd6949bb69)

### Fixed
- Existing algorithms adapted to new dataclass structure [[FIX]](https://github.com/neurogeriatricskiel/KielMAT/commit/3adf7756d9998b36454dccc86d9e2283200d72ed)

## [0.0.2] - 2024-01-22

Second unofficial release of KielMAT for testing purposes.

### Added
- Physical acitivity monitoring algorithm [[#29]](https://github.com/neurogeriatricskiel/KielMAT/commit/a8d9067cde00f0c9a0dba8b7fc623ba4eeb32d0a)

## [0.0.1] - 2023-11-21

This is the first unofficial release of KielMAT.
Therefore, we do not have a proper changelog for this release.

### Added
- All the things :)

### Changed

### Deprecated

### Removed

### Fixed

### Migration Guide