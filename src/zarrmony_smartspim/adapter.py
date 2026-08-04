"""Reader Protocol adapter for a SmartSPIM stitched-export directory.

Multi-channel support (slice #2): every ``Ex_<λ>_Ch<N>_stitched/`` subdir
under the export root is treated as one channel and they are stacked along
``C`` in alphabetical order of the source directory names, matching the
tracer-bullet's deterministic-selection convention. Physical pixel sizes
come from the sidecar; per-channel identity (dye / fluor / emission band)
is optional and driven by an optional ``wavelength_config`` block.

The reader also synthesises an ``ome_types.OME`` with one ``Image`` whose
``Pixels/Channel`` elements carry ``Name`` (from sidecar or derived from the
excitation wavelength), ``Fluor`` and ``ExcitationWavelength`` / ``EmissionWavelength``
when the sidecar provides them. zarrmony consumes this directly, bypassing
the XML round-trip.

Instrument audit metadata (slice #3, ADR-0008 / zarrmony#63–#65) lands on
the same ``ome_types.OME`` object: one ``<Instrument>`` at the OME root
carrying ``<Microscope Manufacturer="LifeCanvas" Model=... SerialNumber=.../>``
+ ``<Objective NominalMagnification=... LensNA=... Immersion=... Model=.../>``,
and an ``AcquisitionDate`` on the ``<Image>``. zarrmony's OME extractor
(``zarrmony.metadata.ome_extractors``) projects those into
``attrs.zarrmony.audit.per_scene[0].{objective,acquisition}`` without any
SmartSPIM-specific handling — the audit block matches zarrmony#63–#65
byte-for-byte for the keys OME can carry. ``InstrumentRef`` /
``ObjectiveSettings`` are deliberately NOT stamped on the Image because
zarrmony's per-scene writer (LIF-only ``_instrument_for_objective``) drops
non-LIF instruments when serialising ``OME/METADATA.ome.xml``, so those
refs would round-trip as orphaned ID references on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from ome_types.model import (
    OME,
    Channel,
    Image,
    Instrument,
    Microscope,
    Objective,
    Pixels,
)

from ._metadata import (
    ChannelIdentity,
    InstrumentIdentity,
    SmartSpimMetadata,
    SmartSpimMetadataError,
    find_metadata_file,
    parse_metadata_file,
)


class SmartSpimError(Exception):
    """Base class for zarrmony-smartspim errors."""


class SmartSpimDataError(SmartSpimError):
    """The export directory has no readable channel data."""


__all__ = [
    "SmartSpimDataError",
    "SmartSpimError",
    "SmartSpimMetadataError",
    "SmartSpimReader",
]


_STITCHED_DIR_RE = re.compile(r"^Ex_(?P<excitation>\d+)_Ch(?P<channel>\d+)_stitched$")

# Every SmartSPIM export is by definition made by LifeCanvas — surfacing that
# in the OME Microscope element lets zarrmony's OME extractor combine it with
# the sidecar-derived model into a ``microscope`` audit value of
# ``"LifeCanvas <model>"``. Matches the shape zarrmony#63–#65 produce for
# their vendors (``"Zeiss AxioObserver"``, ``"Nikon Ti2"`` etc.).
_MICROSCOPE_MANUFACTURER = "LifeCanvas"
_DEFAULT_MICROSCOPE_MODEL = "SmartSPIM"

_INSTRUMENT_ID = "Instrument:0"
_OBJECTIVE_ID = "Objective:0:0"


@dataclass(frozen=True)
class _PixelSizes:
    X: float | None
    Y: float | None
    Z: float | None


@dataclass(frozen=True)
class _ChannelSource:
    """One channel's on-disk footprint plus its derived identity."""

    directory: Path
    slice_files: list[Path]
    excitation_nm: int
    channel_dir_index: int  # the ``N`` in ``Ex_<λ>_Ch<N>_stitched``
    identity: ChannelIdentity


def _read_tif(path: str) -> np.ndarray:
    """Read a single Z-slice ``.tif`` and return a 2-D ``(Y, X)`` array.

    SmartSPIM stitched output writes one 2-D image per file; tifffile's
    ``imread`` returns it directly. Kept as a top-level function so
    ``dask.delayed`` can pickle it for distributed schedulers.
    """
    return tifffile.imread(path)


class SmartSpimReader:
    layout_hint = "flat"
    plate_layout = None
    scenes = ["volume"]

    def __init__(self, path: Path) -> None:
        self._dir = Path(path)

        metadata_path = find_metadata_file(self._dir)
        self._metadata_path = metadata_path
        self._meta: SmartSpimMetadata = parse_metadata_file(metadata_path)

        self._channels: list[_ChannelSource] = _discover_channels(self._dir, self._meta)

        # Every channel in a stitched export shares the same (Z, Y, X) shape
        # and dtype (the vendor writes matched stacks). We peek only the first
        # channel's first slice to lock those in without materializing the
        # whole graph; a per-channel mismatch would surface at compute time.
        first_slice = self._channels[0].slice_files[0]
        first = tifffile.imread(str(first_slice))
        if first.ndim != 2:
            raise SmartSpimDataError(
                f"expected 2-D (Y, X) slice in {first_slice}, got shape {first.shape}"
            )
        self._plane_shape: tuple[int, int] = first.shape
        self._plane_dtype: np.dtype = first.dtype

        expected_z = len(self._channels[0].slice_files)
        for channel in self._channels[1:]:
            if len(channel.slice_files) != expected_z:
                raise SmartSpimDataError(
                    f"channel {channel.directory.name} has {len(channel.slice_files)} "
                    f"Z-slices; first channel {self._channels[0].directory.name} has "
                    f"{expected_z}. All channels in an export must share a Z depth."
                )
        self._size_z = expected_z

        self._active = 0

    def set_scene(self, index: int) -> None:
        if not 0 <= index < len(self.scenes):
            raise IndexError(
                f"scene index {index} out of range; valid indices are 0..{len(self.scenes) - 1}"
            )
        self._active = index

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        channel_stacks = []
        for channel in self._channels:
            planes = [
                da.from_delayed(
                    dask.delayed(_read_tif)(str(path)),
                    shape=self._plane_shape,
                    dtype=self._plane_dtype,
                )
                for path in channel.slice_files
            ]
            channel_stacks.append(da.stack(planes, axis=0))  # (Z, Y, X) per channel
        stacked = da.stack(channel_stacks, axis=0)  # (C, Z, Y, X)
        # Add singleton T to reach OME-Zarr 0.5's (T, C, Z, Y, X).
        volume = stacked[np.newaxis, :, :, :, :]
        return xr.DataArray(volume, dims=("T", "C", "Z", "Y", "X"))

    @property
    def physical_pixel_sizes(self) -> _PixelSizes:
        return _PixelSizes(
            X=self._meta.xy_pixel_size_um,
            Y=self._meta.xy_pixel_size_um,
            Z=self._meta.z_step_um,
        )

    @property
    def dtype(self) -> np.dtype:
        # zarrmony >=0.9 reads reader.dtype to compute the OME-NGFF display
        # window; mirror the on-disk slice dtype so we don't materialize the
        # dask graph just to answer this.
        return self._plane_dtype

    @property
    def channel_names(self) -> list[str]:
        return [_channel_name(channel) for channel in self._channels]

    @property
    def ome_metadata(self) -> OME:
        """Synthesized OME with one Image carrying per-channel identity.

        Returned as a native ``ome_types.OME`` object so zarrmony's
        ``_try_get_ome_image`` path picks up the Image (and its channels)
        directly, without an XML round-trip. Missing identity fields are
        omitted rather than nulled — matches the ADR-0008 stance that
        "reader didn't extract this" is distinct from "reader tried and got
        nothing".

        The instrument audit block (ADR-0008 / zarrmony#63–#65) is folded in
        as ``<Instrument>`` (``<Microscope>`` + ``<Objective>``), an
        ``AcquisitionDate`` on the ``<Image>``, and matching
        ``InstrumentRef`` + ``ObjectiveSettings`` linkages so zarrmony's
        ``extract_{objective,acquisition}_from_ome`` project the fields into
        ``attrs.zarrmony.audit.per_scene[0].{objective,acquisition}`` with no
        SmartSPIM-specific handling.
        """
        size_y, size_x = self._plane_shape
        channels: list[Channel] = []
        for i, source in enumerate(self._channels):
            channels.append(_build_ome_channel(i, source))
        pixels = Pixels(
            id="Pixels:0",
            dimension_order="XYZCT",
            type=str(self._plane_dtype),
            size_x=int(size_x),
            size_y=int(size_y),
            size_z=int(self._size_z),
            size_c=len(self._channels),
            size_t=1,
            physical_size_x=self._meta.xy_pixel_size_um,
            physical_size_y=self._meta.xy_pixel_size_um,
            physical_size_z=self._meta.z_step_um,
            channels=channels,
        )
        image_kwargs: dict[str, Any] = {
            "id": "Image:0",
            "name": self.scenes[self._active],
            "pixels": pixels,
        }
        acquisition_dt = _parse_iso_datetime(self._meta.instrument.acquisition_date)
        if acquisition_dt is not None:
            image_kwargs["acquisition_date"] = acquisition_dt
        image = Image(**image_kwargs)
        # The Instrument is emitted at the OME root (where zarrmony's
        # ``extract_{objective,acquisition}_from_ome`` reads it), but we
        # deliberately do NOT stamp ``InstrumentRef`` / ``ObjectiveSettings``
        # on the Image: zarrmony's per-scene writer (LIF-only ``_instrument_for_objective``)
        # ignores non-LIF ``reader.ome_metadata.instruments`` when serialising
        # the METADATA.ome.xml, so those refs would round-trip as orphaned
        # ID references on disk. Leaving them off keeps the OME-XML clean
        # while still populating the audit block via the extractor path.
        instrument = _build_ome_instrument(self._meta.instrument)
        return OME(
            images=[image],
            instruments=[instrument] if instrument is not None else [],
        )

    @property
    def instrument_audit(self) -> dict[str, Any]:
        """The ADR-0008 instrument block projected into the audit shape.

        Convenience surface for tests and for callers that want the block
        without round-tripping through zarrmony's OME extractors. Shape
        matches what ``zarrmony.metadata.ome_extractors`` produces for a
        non-LIF reader, plus an ``imaging_method`` list (``["light_sheet"]``)
        that OME can't carry natively. Every top-level key is optional and
        absent when the sidecar had no extractable value; ``imaging_method``
        is always present for a SmartSPIM export.
        """
        return _instrument_audit_dict(self._meta.instrument)

    @property
    def channel_audit(self) -> list[dict[str, Any]]:
        """Per-channel identity dicts in the ADR-0008 / zarrmony#61 shape.

        Consumers (and, once zarrmony ships #61, the audit projector in
        ``_convert_per_scene``) read this to populate ``per_scene[i].channels``.
        Missing sidecar fields are OMITTED rather than nulled — key absent
        means "reader didn't extract this field", per ADR-0008.
        """
        return [_channel_audit_entry(i, source) for i, source in enumerate(self._channels)]

    @property
    def metadata(self) -> str:
        return self._meta.raw_text

    def close(self) -> None:
        pass


def _discover_channels(
    export_dir: Path,
    meta: SmartSpimMetadata,
) -> list[_ChannelSource]:
    """Enumerate every ``Ex_<λ>_Ch<N>_stitched/`` subdir under ``export_dir``.

    Sorted alphabetically for determinism (matches slice #1's single-channel
    behavior). ``_MIP_stitched`` dirs are excluded by the anchored regex. An
    empty channel dir — or an export with no matching subdirs at all —
    raises ``SmartSpimDataError`` so callers hear about it before we start
    building a phantom empty pyramid.
    """
    matched: list[tuple[Path, int, int]] = []
    for entry in sorted(export_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = _STITCHED_DIR_RE.match(entry.name)
        if m is None:
            continue
        matched.append((entry, int(m.group("excitation")), int(m.group("channel"))))

    if not matched:
        raise SmartSpimDataError(f"no Ex_<λ>_Ch<N>_stitched channel dir in {export_dir}")

    channels: list[_ChannelSource] = []
    for directory, excitation, channel_index in matched:
        slice_files = sorted(p for p in directory.glob("*.tif") if p.is_file())
        if not slice_files:
            raise SmartSpimDataError(f"no .tif files in {directory}")
        identity = meta.channel_identity_for(excitation)
        channels.append(
            _ChannelSource(
                directory=directory,
                slice_files=slice_files,
                excitation_nm=excitation,
                channel_dir_index=channel_index,
                identity=identity,
            )
        )
    return channels


def _channel_name(source: _ChannelSource) -> str:
    """Best available human-readable label for a channel.

    Priority: explicit ``name`` from the sidecar → ``dye`` from the sidecar →
    derived from the excitation wavelength (``Ex488``). Never falls back to
    ``C:0``/``C:1`` for a SmartSPIM export because the excitation is
    unambiguously encoded in the directory name — a raw ``C:i`` label would
    hide information zarrmony can carry into the omero block for free. The
    ``C:0``/``C:1`` fallback exists in the plugin doc §2 soft-optional table
    only for the (currently hypothetical) case of a channel dir whose name
    doesn't match ``Ex_<λ>_Ch<N>_stitched`` — the matcher ensures we never
    reach that.
    """
    if source.identity.name:
        return source.identity.name
    if source.identity.dye:
        return source.identity.dye
    return f"Ex{source.excitation_nm}"


def _build_ome_channel(index: int, source: _ChannelSource) -> Channel:
    """Build one ``ome_types.model.Channel`` from a channel source.

    Excitation is always present (from the directory name); everything else
    is omitted when the sidecar doesn't provide it. ``emission_wavelength``
    is the low end of the band when a band is provided (a single scalar in
    OME's model) — the full band is preserved in :meth:`channel_audit` for
    downstream ingest.
    """
    identity = source.identity
    kwargs: dict[str, Any] = {
        "id": f"Channel:0:{index}",
        "name": _channel_name(source),
        "excitation_wavelength": float(source.excitation_nm),
    }
    if identity.fluor:
        kwargs["fluor"] = identity.fluor
    if identity.emission_low_nm is not None:
        kwargs["emission_wavelength"] = identity.emission_low_nm
    return Channel(**kwargs)


def _build_ome_instrument(identity: InstrumentIdentity) -> Instrument | None:
    """Build the OME ``<Instrument>`` from the parsed sidecar fields.

    ``manufacturer`` is always ``LifeCanvas`` — a stitched SmartSPIM export
    was made on a LifeCanvas microscope by construction. ``model`` falls back
    to the family name ``"SmartSPIM"`` when the sidecar doesn't specify a
    model, so ``extract_acquisition_from_ome`` still surfaces a non-empty
    ``microscope`` audit value. Serial and every objective field are
    strictly optional: missing → attribute omitted, never nulled.

    Returns ``None`` only if we somehow ended up with nothing worth putting
    in the instrument at all — currently unreachable because we always know
    the manufacturer + model, but the ``None`` path keeps the caller's
    fold-into-Image logic uniform.
    """
    microscope = Microscope(
        manufacturer=_MICROSCOPE_MANUFACTURER,
        model=identity.microscope_model or _DEFAULT_MICROSCOPE_MODEL,
        serial_number=identity.microscope_serial,
    )
    objective = _build_ome_objective(identity)
    return Instrument(
        id=_INSTRUMENT_ID,
        microscope=microscope,
        objectives=[objective] if objective is not None else [],
    )


def _build_ome_objective(identity: InstrumentIdentity) -> Objective | None:
    """Build the OME ``<Objective>`` from the parsed sidecar fields.

    Returns ``None`` when the sidecar exposed no objective data at all — in
    that case the Instrument still ships (for the microscope block) but the
    ``ObjectiveSettings`` linkage is omitted from the Image, and
    ``extract_objective_from_ome`` returns ``None`` so the audit's
    ``objective`` key is absent per the ADR-0008 omit-not-null rule.
    """
    kwargs: dict[str, Any] = {"id": _OBJECTIVE_ID}
    if identity.objective_magnification is not None:
        kwargs["nominal_magnification"] = identity.objective_magnification
    if identity.objective_numerical_aperture is not None:
        kwargs["lens_na"] = identity.objective_numerical_aperture
    if identity.objective_immersion is not None:
        kwargs["immersion"] = identity.objective_immersion
    if identity.objective_model is not None:
        kwargs["model"] = identity.objective_model
    if len(kwargs) == 1:  # only the id — no real objective data
        return None
    return Objective(**kwargs)


def _parse_iso_datetime(raw: str | None):
    """Parse an ISO 8601 string into a ``datetime``, or ``None`` on failure.

    ``_metadata._pick_date`` normalises the sidecar's timestamp to ISO 8601
    when it can; ``ome_types.model.Image.acquisition_date`` requires a
    ``datetime`` object, so we round-trip through ``fromisoformat``. A
    non-parseable string (rare fallback path from ``_pick_date``) yields
    ``None`` and the ``AcquisitionDate`` attribute is omitted from the OME.
    """
    if not raw:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _instrument_audit_dict(identity: InstrumentIdentity) -> dict[str, Any]:
    """Project the parsed sidecar fields into the ADR-0008 audit shape.

    Matches the union of what ``extract_acquisition_from_ome`` +
    ``extract_objective_from_ome`` produce for a non-LIF reader, keyed by
    the same names zarrmony#63–#65 landed. Missing fields are OMITTED
    rather than nulled per the ADR-0008 omit-not-null rule.
    """
    acquisition: dict[str, Any] = {}
    if identity.acquisition_date is not None:
        acquisition["date"] = identity.acquisition_date
    microscope_parts = [_MICROSCOPE_MANUFACTURER]
    microscope_parts.append(identity.microscope_model or _DEFAULT_MICROSCOPE_MODEL)
    acquisition["microscope"] = " ".join(microscope_parts)
    if identity.microscope_serial is not None:
        acquisition["microscope_serial"] = identity.microscope_serial
    if identity.imaging_method:
        acquisition["imaging_method"] = list(identity.imaging_method)

    objective: dict[str, Any] = {}
    if identity.objective_magnification is not None:
        objective["nominal_magnification"] = _as_int_when_integral(identity.objective_magnification)
    if identity.objective_numerical_aperture is not None:
        objective["numerical_aperture"] = _as_int_when_integral(
            identity.objective_numerical_aperture
        )
    if identity.objective_immersion is not None:
        objective["immersion"] = identity.objective_immersion
    if identity.objective_model is not None:
        objective["model"] = identity.objective_model

    result: dict[str, Any] = {"acquisition": acquisition}
    if objective:
        result["objective"] = objective
    return result


def _as_int_when_integral(value: float) -> int | float:
    """Match zarrmony's ``_to_number`` shape: int if integral, else float."""
    return int(value) if float(value).is_integer() else value


def _channel_audit_entry(index: int, source: _ChannelSource) -> dict[str, Any]:
    identity = source.identity
    entry: dict[str, Any] = {
        "index": index,
        "name": _channel_name(source),
        "excitation_nm": int(source.excitation_nm),
    }
    if identity.dye:
        entry["dye"] = identity.dye
    if identity.fluor:
        entry["fluor"] = identity.fluor
    if identity.emission_low_nm is not None:
        entry["emission_low_nm"] = identity.emission_low_nm
    if identity.emission_high_nm is not None:
        entry["emission_high_nm"] = identity.emission_high_nm
    return entry
