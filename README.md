# zarrmony-smartspim

LifeCanvas [SmartSPIM](https://lifecanvastech.com/products/smartspim/) reader
plugin for [zarrmony](https://github.com/ferrinm/zarrmony). Detects a
SmartSPIM stitched-export directory (one that contains one or more
`Ex_<λ>_Ch<N>_stitched/` channel subdirs) and converts it to OME-NGFF 0.5:

```bash
zarrmony convert /path/to/<sample-id> ./out
```

## Install

_Not yet on PyPI._ Install from source:

```bash
pip install git+https://github.com/ferrinm/zarrmony-smartspim
```

This pulls `zarrmony` from PyPI as a transitive dependency.

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
- The ADR-0008 instrument audit block (microscope, serial, acquisition
  date) lands in a follow-up slice.

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

## Metadata sidecar

The SmartSPIM acquisition software writes a JSON metadata file. It is
**Latin-1 encoded** (Windows `cp1252`) because the SmartSPIM PC writes a raw
`µ` byte (`0xB5`) into the `µm/pix` key. The parser reads with `latin-1` for
that reason; do not re-encode the file to UTF-8 before feeding it to the
plugin — the key would be lost.

Place the metadata JSON at the top of the export directory. Any file matching
`metadata*.json` is accepted (the vendor typically names it
`metadata_<sample-id>.json`).

## License

Apache-2.0. See [LICENSE](LICENSE).
