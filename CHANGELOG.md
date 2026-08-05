# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-08-05

### Added

- `SmartSpimReader` accepts an optional `metadata_path=` kwarg that
  overrides the top-of-export-directory sidecar lookup. Motivating case:
  a read-only export mount paired with a sidecar that lives on a
  separate writable drive (a common LifeCanvas deployment shape). A
  missing override file raises `SmartSpimMetadataError` with a clear
  message. The zarrmony plugin entry point is unchanged — it still
  discovers the sidecar via the top-level `metadata*.json` glob.

## [0.1.1] — 2026-08-05

### Changed

- README cleanup now that the package is live on PyPI: removed the
  "not yet on PyPI" caveat and the `git+https://...` install fallback.
  No code changes.

## [0.1.0] — 2026-08-05

### Added

- Initial release. `SmartSpimReader` adapter satisfies zarrmony's
  `ReaderProtocol` and registers as `zarrmony-smartspim` via the
  `zarrmony.readers` entry point.
- **Directory matcher** that fires on a stitched-export directory
  containing at least one `Ex_<λ>_Ch<N>_stitched/` child. Cheap and
  side-effect-free (single `iterdir` pass, no full-tree scan).
- **Single-scene volume reader** (`scenes = ["volume"]`) with
  `layout_hint = "flat"`. `xarray_dask_data` lazily stacks the on-disk
  TIFF slices along `Z` in OME-NGFF `TCZYX` order.
- **Multi-channel support.** Every `Ex_<λ>_Ch<N>_stitched/` directory
  under the export root is stacked along `C` in directory-name order.
  `channel_names` returns real labels from the optional sidecar
  `wavelength_config` block; falls back to `Ex<λ>` labels when the
  block is absent.
- **Physical pixel sizes** populated from the sidecar
  (`session_config.µm/pix` for X/Y, `session_config.z_step_um` for Z).
  The sidecar is read as `latin-1` because SmartSPIM writes a raw
  `µ` byte (`0xB5`) into the `µm/pix` key.
- **Native `ome_types.OME` synthesis** exposed via
  `reader.ome_metadata`. Carries `Pixels/Channel` with
  `Name`, `Fluor`, `ExcitationWavelength`, and `EmissionWavelength`
  when the sidecar provides them. zarrmony consumes it directly,
  bypassing the XML round-trip. Per-channel audit entries (dye /
  fluor / excitation / emission) land under `attrs.zarrmony.audit`
  matching the shape used by CZI/LIF/ND2/OME-TIFF
  (closed `ferrinm/zarrmony#61`).
- **ADR-0008 instrument audit block.** Microscope model + serial,
  objective (magnification / NA / immersion / model), and acquisition
  date are folded into the OME's `<Instrument>` + `<Image>` and land
  under `attrs.zarrmony.audit.per_scene[0].{acquisition,objective}`
  via zarrmony's OME extractor. `imaging_method` is populated
  unconditionally with `["light_sheet"]` — a stitched SmartSPIM
  export is a light-sheet acquisition by construction. Liberal key
  aliasing (LifeCanvas has shipped several spellings across software
  versions); missing sidecar fields are omitted from the audit rather
  than nulled, per ADR-0008's omit-not-null rule.

### Known limitations

- Single-scene per SmartSPIM export (one Z-stack).
- Sidecar-driven identity — `wavelength_config` and instrument
  fields are optional; missing fields degrade to `Ex<λ>` labels
  and omitted audit entries respectively.
