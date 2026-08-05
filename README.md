# zarrmony-smartspim

LifeCanvas [SmartSPIM](https://lifecanvastech.com/products/smartspim/) reader
plugin for [zarrmony](https://github.com/ferrinm/zarrmony). Detects a
SmartSPIM stitched-export directory (one that contains one or more
`Ex_<λ>_Ch<N>_stitched/` channel subdirs) and converts it to OME-NGFF 0.5:

```bash
zarrmony convert /path/to/<sample-id> ./out
```

## Install

```bash
pip install zarrmony-smartspim
```

This pulls `zarrmony` from PyPI as a transitive dependency.

## Verify the plugin registered

```python
from zarrmony.readers.plugin import list_plugins

print([p.name for p in list_plugins()])
# -> [..., 'zarrmony-smartspim']
```

## Supported SmartSPIM exports

- **Export format**: LifeCanvas SmartSPIM stitched-export directory —
  one or more `Ex_<λ>_Ch<N>_stitched/` subdirs at the top level, each
  containing 2-D TIFF Z-slices, plus a top-level
  `metadata_<sample-id>.json` sidecar (Latin-1 encoded).
- **Acquisition software**: exercised against sidecars produced by
  LifeCanvas SmartSPIM acquisition software v5.x. Earlier and later
  versions are read on a best-effort basis via liberal key aliasing
  (see the "Instrument audit fields" table below) — the vendor has
  shipped several key spellings over the years and the parser accepts
  each.
- **Detection**: the matcher fires on the presence of at least one
  `Ex_<λ>_Ch<N>_stitched/` child. A missing metadata sidecar surfaces
  as `SmartSpimMetadataError` at read time with a message pointing at
  the expected filename.

## Scope

- Single-scene per SmartSPIM export (one Z-stack).
- All `Ex_<λ>_Ch<N>_stitched/` directories under the export root are stacked
  along `C` in directory-name order.
- Physical pixel sizes populated from the sidecar metadata JSON
  (`session_config.µm/pix` for X/Y, `session_config.z_step_um` for Z).
- Per-channel identity — name, dye, fluorophore, excitation and emission
  wavelengths — synthesised into a native `ome_types.OME` object exposed as
  `reader.ome_metadata`. Excitation is always known (from the directory
  name); the remaining fields come from an optional `wavelength_config`
  block in the sidecar (see below).
- The ADR-0008 instrument audit block (microscope, serial, objective,
  acquisition date) is folded into the OME's `<Instrument>` + `<Image>`
  and lands under `attrs.zarrmony.audit.per_scene[i].{acquisition,objective}`
  via zarrmony's OME extractor. Missing sidecar fields are omitted from
  the audit rather than nulled, per the ADR-0008 omit-not-null rule.

### Optional `wavelength_config` block

Keyed by excitation-wavelength string. Any subset of these keys is honored;
the whole block is optional and readers fall back to `Ex<λ>` labels when it
is absent:

```json
"wavelength_config": {
  "488": {
    "name": "GFP",
    "dye": "GFP",
    "fluor": "GFP",
    "emission_low_nm": 500,
    "emission_high_nm": 550
  },
  "561": {"name": "mCherry", "fluor": "mCherry", "emission_nm": 610}
}
```

A single `emission_nm` scalar is expanded to `emission_low_nm ==
emission_high_nm` per the ADR-0008 / zarrmony#61 uniform-band convention.

### Instrument audit fields

Populated from the sidecar with liberal key aliasing (LifeCanvas software
has shipped several spellings — first-writer-wins across each list).
Everything is optional; missing fields are omitted from
`attrs.zarrmony.audit`.

| Audit key                     | Accepted sidecar keys (session_config OR top-level)                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `microscope`                  | `microscope_model`, `microscope`, `system_model`, `system`, `model`, `instrument`            |
| `microscope_serial`           | `microscope_serial`, `serial_number`, `serial`, `machine_id`, `machine`, `system_serial`, `instrument_serial` |
| `date` (acquisition)          | `acquisition_date`, `date`, `start_time`, `acquisition_start`, `session_start`, `timestamp`  |
| `objective.nominal_magnification` | `obj_magnification`, `objective_magnification`, `magnification`, `nominal_magnification` |
| `objective.numerical_aperture`    | `NA`, `na`, `numerical_aperture`, `obj_NA`, `objective_NA`, `objective_na`               |
| `objective.model`                 | `obj_name`, `objective_name`, `objective_model`, `objective`                              |
| `objective.immersion`             | `immersion`, `Immersion`, `objective_immersion`, `immersion_media`, `immersion_medium`   |

`microscope` always resolves to at least `"LifeCanvas SmartSPIM"` — a
SmartSPIM export was made on a LifeCanvas microscope by construction.
Refractive-index shorthand (`"1.52"` / `"1.52+"`) in the immersion field
degrades to the OME `"Other"` enum value since OME has no cleared-tissue
enum. Vendor-shape acquisition-date formats (`YYYY_MM_DD_HHMMSS`,
`YYYYMMDD_HHMMSS`) are normalised to ISO 8601.

## Metadata sidecar

The SmartSPIM acquisition software writes a JSON metadata file. It is
**Latin-1 encoded** (Windows `cp1252`) because the SmartSPIM PC writes a raw
`µ` byte (`0xB5`) into the `µm/pix` key. The parser reads with `latin-1` for
that reason; do not re-encode the file to UTF-8 before feeding it to the
plugin — the key would be lost.

Place the metadata JSON at the top of the export directory. Any file matching
`metadata*.json` is accepted (the vendor typically names it
`metadata_<sample-id>.json`).

### Sidecar stored outside a read-only export

If the export directory is read-only (a common LifeCanvas deployment shape —
the acquisition PC's share is exposed as read-only, and the sidecar lives on
a project drive), pass the sidecar path directly. The reader skips its usual
top-of-directory lookup and reads from wherever you point it:

```python
from zarrmony_smartspim import SmartSpimReader

reader = SmartSpimReader(
    "/read-only/mount/<sample-id>",
    metadata_path="/writable/project/metadata_<sample-id>.json",
)
```

This kwarg is only exposed on the direct `SmartSpimReader` constructor —
zarrmony's plugin entry point still looks for the sidecar at the top of the
export directory.

## Why a separate package?

SmartSPIM ships its own on-disk shape (directory-of-channel-dirs, no
bundled OME-XML, Latin-1 sidecar) that would not fit cleanly into
zarrmony's built-in reader graph. See zarrmony
[ADR-0003](https://github.com/ferrinm/zarrmony/blob/main/docs/adr/0003-external-adapter-package-for-non-bioio-readers.md)
for the full rationale, and the
[reader-plugin authoring guide](https://github.com/ferrinm/zarrmony/blob/main/docs/writing-a-reader-plugin.md)
for how to build your own.

## License

Apache-2.0. See [LICENSE](LICENSE).
