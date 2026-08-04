"""Parser for the SmartSPIM ``metadata_<sample>.json`` sidecar.

The vendor's acquisition software (LifeCanvas SmartSPIM, running on Windows)
writes a JSON file that is **Latin-1 encoded**: the ``µm/pix`` key contains a
raw ``0xB5`` byte for the micro sign rather than UTF-8's two-byte ``0xC2 0xB5``
sequence. ``json.loads(raw.decode("utf-8"))`` therefore raises
``UnicodeDecodeError`` on real files; we open with ``latin-1`` to round-trip
the bytes verbatim.

Only the two fields v0.1 needs (X/Y pixel size and Z step, both in microns)
are surfaced on the dataclass. The verbatim raw dict is preserved on
``raw`` for the ``metadata`` attribute and future audit-block use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SmartSpimMetadataError(ValueError):
    """The metadata sidecar is missing, unreadable, or missing required fields."""


_METADATA_GLOB = "metadata*.json"
_XY_PIXEL_KEY = "µm/pix"  # literal 'µm/pix' — matches the vendor's latin-1 byte
_Z_STEP_KEY = "z_step_um"


@dataclass(frozen=True)
class SmartSpimMetadata:
    """Structured view of a SmartSPIM ``metadata_<sample>.json`` sidecar."""

    raw: dict[str, Any]
    raw_text: str
    xy_pixel_size_um: float
    z_step_um: float


def find_metadata_file(export_dir: Path) -> Path:
    """Locate the metadata JSON at the top of ``export_dir``.

    The LifeCanvas naming convention is ``metadata_<sample-id>.json``, but we
    accept any ``metadata*.json`` at the top level. Missing or ambiguous
    files raise ``SmartSpimMetadataError`` with an actionable message — the
    JSON is often stored outside the export dir, and the user needs to place
    (or symlink) it in.
    """
    candidates = sorted(p for p in export_dir.glob(_METADATA_GLOB) if p.is_file())
    if not candidates:
        raise SmartSpimMetadataError(
            f"no {_METADATA_GLOB} sidecar in {export_dir}; "
            "copy the SmartSPIM metadata JSON into the export directory "
            "(the vendor names it metadata_<sample-id>.json)"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise SmartSpimMetadataError(
            f"multiple {_METADATA_GLOB} sidecars in {export_dir}: {names}; "
            "keep exactly one so the reader has an unambiguous source of truth"
        )
    return candidates[0]


def parse_metadata_file(path: Path) -> SmartSpimMetadata:
    # SmartSPIM writes latin-1 bytes for the µ sign — see module docstring.
    raw_text = path.read_text(encoding="latin-1")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SmartSpimMetadataError(f"{path} is not valid JSON: {exc}") from exc

    try:
        session_config = raw["session_config"]
    except (KeyError, TypeError) as exc:
        raise SmartSpimMetadataError(f"{path} is missing the 'session_config' block") from exc

    try:
        xy_pixel_size_um = float(session_config[_XY_PIXEL_KEY])
    except KeyError as exc:
        raise SmartSpimMetadataError(
            f"{path} session_config is missing the {_XY_PIXEL_KEY!r} key"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise SmartSpimMetadataError(
            f"{path} session_config[{_XY_PIXEL_KEY!r}] is not a number: "
            f"{session_config[_XY_PIXEL_KEY]!r}"
        ) from exc

    try:
        z_step_um = float(session_config[_Z_STEP_KEY])
    except KeyError as exc:
        raise SmartSpimMetadataError(
            f"{path} session_config is missing the {_Z_STEP_KEY!r} key"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise SmartSpimMetadataError(
            f"{path} session_config[{_Z_STEP_KEY!r}] is not a number: "
            f"{session_config[_Z_STEP_KEY]!r}"
        ) from exc

    return SmartSpimMetadata(
        raw=raw,
        raw_text=raw_text,
        xy_pixel_size_um=xy_pixel_size_um,
        z_step_um=z_step_um,
    )
