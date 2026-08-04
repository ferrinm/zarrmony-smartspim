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

## Scope of v0.1 (tracer bullet)

The v0.1 plugin ships the smallest end-to-end slice:

- Single-scene per SmartSPIM export (one Z-stack, from the first
  `Ex_<λ>_Ch<N>_stitched/` channel dir it finds).
- Physical pixel sizes populated from the sidecar metadata JSON
  (`session_config.µm/pix` for X/Y, `session_config.z_step_um` for Z).
- Channel identity, OME-XML synthesis, and the ADR-0008 instrument audit
  block land in follow-up slices.

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
