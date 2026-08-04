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


def _open(path: Path) -> SmartSpimReader:
    return SmartSpimReader(path)


plugin = ReaderPlugin(
    name="zarrmony-smartspim",
    match=match,
    open=_open,
    distribution="zarrmony-smartspim",
    source="entry_point",
)
