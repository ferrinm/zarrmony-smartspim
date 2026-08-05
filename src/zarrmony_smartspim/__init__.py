"""zarrmony-smartspim — LifeCanvas SmartSPIM reader plugin for zarrmony.

The package ships one ``ReaderPlugin`` value, registered under the
``zarrmony.readers`` entry point declared in ``pyproject.toml``:

- ``plugin`` — matches a SmartSPIM stitched-export directory (one containing
  at least one ``Ex_<λ>_Ch<N>_stitched/`` child) and reads the first channel
  as a single-scene volume.

End users do not import from this package directly; they
``pip install zarrmony-smartspim`` and zarrmony picks the plugin up
automatically.
"""

from pathlib import Path

from zarrmony.readers.plugin import ReaderPlugin

from ._metadata import SmartSpimMetadata, SmartSpimMetadataError
from .adapter import SmartSpimDataError, SmartSpimError, SmartSpimReader
from .match import match

__all__ = [
    "SmartSpimDataError",
    "SmartSpimError",
    "SmartSpimMetadata",
    "SmartSpimMetadataError",
    "SmartSpimReader",
    "match",
    "plugin",
]


def _open(
    path: Path, *, metadata_path: str | Path | None = None
) -> SmartSpimReader:
    """Plugin entry point — forwards ``metadata_path`` to the reader.

    Zarrmony >= 0.13 calls ``plugin.open(p, **reader_kwargs)`` so that
    callers can reach reader-specific options through the public API
    (``convert(..., reader_kwargs={...})``, ``inspect(..., reader_kwargs={...})``)
    and the CLI (``--reader-kwarg metadata_path=...``). The motivating
    LifeCanvas deployment shape is a read-only export mount paired with a
    sidecar JSON on a separate writable drive.

    Callers on zarrmony < 0.13 continue to work with the default sidecar
    lookup (zarrmony just calls ``_open(path)`` with no kwargs); reaching
    the override requires the passthrough shipped in zarrmony 0.13.
    """
    return SmartSpimReader(path, metadata_path=metadata_path)


plugin = ReaderPlugin(
    name="zarrmony-smartspim",
    match=match,
    open=_open,
    distribution="zarrmony-smartspim",
    source="entry_point",
)
