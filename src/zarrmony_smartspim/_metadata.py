"""Parser for the SmartSPIM ``metadata_<sample>.json`` sidecar.

The vendor's acquisition software (LifeCanvas SmartSPIM, running on Windows)
writes a JSON file that is **Latin-1 encoded**: the ``µm/pix`` key contains a
raw ``0xB5`` byte for the micro sign rather than UTF-8's two-byte ``0xC2 0xB5``
sequence. ``json.loads(raw.decode("utf-8"))`` therefore raises
``UnicodeDecodeError`` on real files; we open with ``latin-1`` to round-trip
the bytes verbatim.

Three blocks are surfaced today:

* ``session_config.µm/pix`` and ``session_config.z_step_um`` — physical
  pixel sizes, required (slice #1).
* ``wavelength_config`` — an optional dict keyed by excitation-wavelength
  string (e.g. ``"488"``) that carries per-channel identity fields
  (``name`` / ``dye`` / ``fluor`` / ``emission_low_nm`` / ``emission_high_nm``
  / ``emission_nm``). Any subset of those keys is accepted; the whole block
  is optional (readers fall back to directory-name-derived names).
* Instrument audit fields for ADR-0008 (slice #3): microscope model +
  serial number, objective (magnification / NA / immersion / model),
  acquisition date, imaging modality. Every field is optional — a
  stripped sidecar degrades to :class:`InstrumentIdentity` with all
  fields ``None`` and no exception raised.

The verbatim raw dict is preserved on ``raw`` for the ``metadata``
attribute and audit-block use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


class SmartSpimMetadataError(ValueError):
    """The metadata sidecar is missing, unreadable, or missing required fields."""


_METADATA_GLOB = "metadata*.json"
_XY_PIXEL_KEY = "µm/pix"  # literal 'µm/pix' — matches the vendor's latin-1 byte
_Z_STEP_KEY = "z_step_um"
_WAVELENGTH_CONFIG_KEY = "wavelength_config"

# Keys we accept for the microscope model. LifeCanvas software has shipped
# several spellings over the years; first-writer-wins across this list.
_MICROSCOPE_MODEL_KEYS: tuple[str, ...] = (
    "microscope_model",
    "microscope",
    "system_model",
    "system",
    "model",
    "instrument",
)

_MICROSCOPE_SERIAL_KEYS: tuple[str, ...] = (
    "microscope_serial",
    "serial_number",
    "serial",
    "machine_id",
    "machine",
    "system_serial",
    "instrument_serial",
)

_ACQUISITION_DATE_KEYS: tuple[str, ...] = (
    "acquisition_date",
    "date",
    "start_time",
    "acquisition_start",
    "session_start",
    "timestamp",
)

_OBJECTIVE_MAGNIFICATION_KEYS: tuple[str, ...] = (
    "obj_magnification",
    "objective_magnification",
    "magnification",
    "nominal_magnification",
)

_OBJECTIVE_NA_KEYS: tuple[str, ...] = (
    "NA",
    "na",
    "numerical_aperture",
    "obj_NA",
    "objective_NA",
    "objective_na",
)

_OBJECTIVE_MODEL_KEYS: tuple[str, ...] = (
    "obj_name",
    "objective_name",
    "objective_model",
    "objective",
)

_OBJECTIVE_IMMERSION_KEYS: tuple[str, ...] = (
    "immersion",
    "Immersion",
    "objective_immersion",
    "immersion_media",
    "immersion_medium",
)

# Map free-form immersion strings from the sidecar to the OME
# ``Objective_Immersion`` enum values. Unknown-but-present strings degrade to
# ``"Other"`` per the enum's own catch-all (matches the LIF extractor's
# stance — "reader saw something we don't recognise" is distinct from "no
# immersion metadata"). The refractive-index shorthand the vendor uses
# (e.g. ``"1.52"`` or ``"1.52+"``) is what most real-world sidecars carry
# for cleared-tissue SmartSPIM runs; those all map to ``"Other"`` since OME
# has no cleared-tissue enum value.
_IMMERSION_TO_OME: dict[str, str] = {
    "OIL": "Oil",
    "WATER": "Water",
    "AIR": "Air",
    "DRY": "Air",
    "GLYC": "Glycerol",
    "GLYCEROL": "Glycerol",
    "MULTI": "Multi",
    "OTHER": "Other",
    "WATERDIPPING": "WaterDipping",
    "DIPPING": "WaterDipping",
}

# SmartSPIM is a light-sheet fluorescence microscope by construction — every
# stitched export it produces was acquired in that modality. Surfacing this
# as an audit constant matches the ADR-0008 ``imaging_method`` shape (a
# ``list[str]`` of OME-conventional tokens) so downstream ingest can consume
# it byte-for-byte alongside the LIF extractor's output.
IMAGING_METHOD_TOKENS: tuple[str, ...] = ("light_sheet",)


@dataclass(frozen=True)
class ChannelIdentity:
    """Per-channel identity fields for one SmartSPIM channel.

    Excitation is always known — it comes from the ``Ex_<λ>_Ch<N>_stitched``
    directory name and the adapter passes it in verbatim. Every other field
    is optional and only present when the sidecar provided it. This mirrors
    the ADR-0008 audit shape: "reader didn't extract this field" (attribute
    ``None``) must be distinguishable from "reader tried and got nothing"
    so downstream ingest can tell the two apart. ``name`` is filled in by
    the adapter (from the sidecar, or derived from the excitation) so it is
    always present in practice.
    """

    excitation_nm: int
    name: str | None = None
    dye: str | None = None
    fluor: str | None = None
    emission_low_nm: float | None = None
    emission_high_nm: float | None = None


@dataclass(frozen=True)
class InstrumentIdentity:
    """Instrument / objective / acquisition fields for the ADR-0008 audit block.

    Every field is optional: the vendor sidecar shape drifts across LifeCanvas
    software versions and some deployments (Aperture ingest of legacy exports,
    Windows PCs missing the plumbing to log a serial) strip fields wholesale.
    Missing fields degrade to ``None`` so the adapter can omit them from the
    audit rather than record placeholder junk (per ADR-0008's omit-not-null
    rule).

    ``imaging_method`` is a ``list[str]`` when populated (matches the BQ
    ``REPEATED STRING`` shape); SmartSPIM populates it unconditionally with
    ``["light_sheet"]`` because a stitched SmartSPIM export was acquired in
    that modality by construction.
    """

    microscope_model: str | None = None
    microscope_serial: str | None = None
    acquisition_date: str | None = None
    objective_magnification: float | None = None
    objective_numerical_aperture: float | None = None
    objective_model: str | None = None
    objective_immersion: str | None = None
    imaging_method: tuple[str, ...] = ()


@dataclass(frozen=True)
class SmartSpimMetadata:
    """Structured view of a SmartSPIM ``metadata_<sample>.json`` sidecar."""

    raw: dict[str, Any]
    raw_text: str
    xy_pixel_size_um: float
    z_step_um: float
    wavelength_config: dict[int, dict[str, Any]] = field(default_factory=dict)
    instrument: InstrumentIdentity = field(default_factory=InstrumentIdentity)

    def channel_identity_for(self, excitation_nm: int) -> ChannelIdentity:
        """Merge sidecar identity fields (if any) with the required excitation.

        Missing sidecar → an identity carrying just the excitation. Missing
        emission band with a single ``emission_nm`` scalar → ``low == high``
        matching #61's uniform-band convention.
        """
        entry = self.wavelength_config.get(excitation_nm, {})
        emission_low = entry.get("emission_low_nm")
        emission_high = entry.get("emission_high_nm")
        emission_scalar = entry.get("emission_nm")
        if emission_scalar is not None and emission_low is None and emission_high is None:
            emission_low = emission_high = float(emission_scalar)
        return ChannelIdentity(
            excitation_nm=excitation_nm,
            name=entry.get("name"),
            dye=entry.get("dye"),
            fluor=entry.get("fluor"),
            emission_low_nm=float(emission_low) if emission_low is not None else None,
            emission_high_nm=float(emission_high) if emission_high is not None else None,
        )


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

    wavelength_config = _parse_wavelength_config(raw.get(_WAVELENGTH_CONFIG_KEY))
    instrument = _parse_instrument(raw, session_config)

    return SmartSpimMetadata(
        raw=raw,
        raw_text=raw_text,
        xy_pixel_size_um=xy_pixel_size_um,
        z_step_um=z_step_um,
        wavelength_config=wavelength_config,
        instrument=instrument,
    )


def _parse_instrument(raw: dict[str, Any], session_config: dict[str, Any]) -> InstrumentIdentity:
    """Extract the ADR-0008 instrument audit fields from the sidecar.

    Every field is optional and fail-closed: a missing/malformed value never
    raises, it just falls through to ``None`` so the audit block omits it.
    LifeCanvas ships several key spellings across its software versions; we
    accept a small alias list per field and take the first non-empty hit.
    Both the ``session_config`` sub-dict and the top-level ``raw`` dict are
    searched (in that order) because different fields land in different spots
    depending on export vintage.

    ``imaging_method`` is set unconditionally to ``("light_sheet",)`` —
    a stitched SmartSPIM export is a light-sheet acquisition by construction,
    and the audit consumer (ADR-0008 / zarrmony#62) expects a
    ``list[str]`` of OME-conventional modality tokens.
    """
    return InstrumentIdentity(
        microscope_model=_pick_str(session_config, raw, keys=_MICROSCOPE_MODEL_KEYS),
        microscope_serial=_pick_str(session_config, raw, keys=_MICROSCOPE_SERIAL_KEYS),
        acquisition_date=_pick_date(session_config, raw, keys=_ACQUISITION_DATE_KEYS),
        objective_magnification=_pick_number(
            session_config, raw, keys=_OBJECTIVE_MAGNIFICATION_KEYS
        ),
        objective_numerical_aperture=_pick_number(session_config, raw, keys=_OBJECTIVE_NA_KEYS),
        objective_model=_pick_str(session_config, raw, keys=_OBJECTIVE_MODEL_KEYS),
        objective_immersion=_pick_immersion(session_config, raw),
        imaging_method=IMAGING_METHOD_TOKENS,
    )


def _pick_str(*sources: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """First non-empty stringified value across ``sources`` under any of ``keys``."""
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return None


def _pick_number(*sources: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """First numeric value across ``sources`` under any of ``keys``, else ``None``."""
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number:  # NaN guard — never surface NaN to the audit
                continue
            return number
    return None


def _pick_date(*sources: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """First parseable date string across ``sources``, normalised to ISO 8601.

    Accepts anything ``datetime.fromisoformat`` can parse plus a handful of
    common vendor formats (LifeCanvas has shipped ``YYYY_MM_DD_HHMMSS`` and
    ``YYYYMMDD_HHMMSS``). A raw string we can't parse still comes through
    verbatim — a non-standard timestamp is more useful in the audit than a
    dropped one, and consumers can still parse it themselves.
    """
    raw = _pick_str(*sources, keys=keys)
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(raw).isoformat()
    except ValueError:
        pass
    for fmt in ("%Y_%m_%d_%H%M%S", "%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return raw


def _pick_immersion(*sources: dict[str, Any]) -> str | None:
    """Map the vendor's immersion string to an OME ``Objective_Immersion`` value.

    Numeric-looking values (``"1.52"`` / ``"1.52+"``) are what most real-world
    cleared-tissue sidecars carry — a refractive index rather than a medium
    name. OME has no cleared-tissue enum value, so those degrade to ``"Other"``
    (the enum's own catch-all) while preserving that we saw *something*.
    """
    raw = _pick_str(*sources, keys=_OBJECTIVE_IMMERSION_KEYS)
    if raw is None:
        return None
    key = raw.strip().upper().replace(" ", "").replace("-", "").rstrip("+")
    if not key:
        return None
    return _IMMERSION_TO_OME.get(key, "Other")


def _parse_wavelength_config(block: Any) -> dict[int, dict[str, Any]]:
    """Normalize the optional ``wavelength_config`` block to int-keyed dicts.

    Real SmartSPIM sidecars key this block by excitation-wavelength STRING;
    we cast to ``int`` so callers can look up by the same integer they parsed
    out of the directory name. Non-dict block, non-integer keys, or non-dict
    entries are silently dropped — the whole block is optional and
    fail-closed is the right stance for identity metadata.
    """
    if not isinstance(block, dict):
        return {}
    normalized: dict[int, dict[str, Any]] = {}
    for key, entry in block.items():
        if not isinstance(entry, dict):
            continue
        try:
            wavelength = int(key)
        except (TypeError, ValueError):
            continue
        normalized[wavelength] = entry
    return normalized
