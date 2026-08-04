"""Reader Protocol adapter for a SmartSPIM stitched-export directory.

v0.1 scope (tracer bullet): a single scene named ``volume`` built from the
first ``Ex_<λ>_Ch<N>_stitched/`` channel directory found under the export
root (sorted alphabetically for determinism). The scene's Z-stack is
assembled by lazily reading each ``.tif`` slice with ``tifffile`` and
stacking along Z; T and C are singleton dimensions.

Multi-channel support (all ``Ex_*_Ch*_stitched`` dirs concatenated along C
with real channel names) and OME-XML synthesis land in slice #2. Instrument
audit metadata lands in slice #3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import tifffile
import xarray as xr

from ._metadata import (
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


_STITCHED_DIR_RE = re.compile(r"^Ex_\d+_Ch\d+_stitched$")


@dataclass(frozen=True)
class _PixelSizes:
    X: float | None
    Y: float | None
    Z: float | None


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

        channel_dir = _first_stitched_channel_dir(self._dir)
        self._channel_dir = channel_dir

        slices = sorted(p for p in channel_dir.glob("*.tif") if p.is_file())
        if not slices:
            raise SmartSpimDataError(f"no .tif files in {channel_dir}")
        self._slice_files: list[Path] = slices

        # Peek the first slice to lock in shape and dtype without materializing
        # the whole stack. Every slice in a stitched export shares the same
        # (Y, X) shape and dtype; a mismatch would surface at compute time.
        first = tifffile.imread(str(slices[0]))
        if first.ndim != 2:
            raise SmartSpimDataError(
                f"expected 2-D (Y, X) slice in {slices[0]}, got shape {first.shape}"
            )
        self._plane_shape: tuple[int, int] = first.shape
        self._plane_dtype: np.dtype = first.dtype

        metadata_path = find_metadata_file(self._dir)
        self._metadata_path = metadata_path
        self._meta: SmartSpimMetadata = parse_metadata_file(metadata_path)

        self._active = 0

    def set_scene(self, index: int) -> None:
        if not 0 <= index < len(self.scenes):
            raise IndexError(
                f"scene index {index} out of range; valid indices are 0..{len(self.scenes) - 1}"
            )
        self._active = index

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        planes = [
            da.from_delayed(
                dask.delayed(_read_tif)(str(path)),
                shape=self._plane_shape,
                dtype=self._plane_dtype,
            )
            for path in self._slice_files
        ]
        stacked = da.stack(planes, axis=0)  # (Z, Y, X)
        # Add singleton T and C axes to reach OME-Zarr 0.5's (T, C, Z, Y, X).
        volume = stacked[np.newaxis, np.newaxis, :, :, :]
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
    def metadata(self) -> str:
        return self._meta.raw_text

    def close(self) -> None:
        pass


def _first_stitched_channel_dir(export_dir: Path) -> Path:
    """Return the first ``Ex_<λ>_Ch<N>_stitched/`` subdir under ``export_dir``.

    Sorted alphabetically so scene selection is deterministic across runs.
    ``_MIP_stitched`` dirs are excluded by the anchored regex. Raises
    ``SmartSpimDataError`` if no matching subdir exists — the matcher fires
    on the same regex, so reaching this branch means the export was mutated
    between match and open.
    """
    stitched = sorted(
        entry
        for entry in export_dir.iterdir()
        if entry.is_dir() and _STITCHED_DIR_RE.match(entry.name)
    )
    if not stitched:
        raise SmartSpimDataError(f"no Ex_<λ>_Ch<N>_stitched channel dir in {export_dir}")
    return stitched[0]
