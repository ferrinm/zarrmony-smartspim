"""Adapter unit tests: Reader Protocol shape, pixel content, sidecar wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarrmony.readers.plugin import ReaderProtocol

from tests.conftest import write_synthetic_smartspim
from zarrmony_smartspim.adapter import (
    SmartSpimDataError,
    SmartSpimMetadataError,
    SmartSpimReader,
)


def test_adapter_satisfies_reader_protocol(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    assert isinstance(reader, ReaderProtocol)


def test_layout_hint_is_flat_and_plate_layout_none(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    assert reader.layout_hint == "flat"
    assert reader.plate_layout is None


def test_scenes_reports_single_volume(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    assert reader.scenes == ["volume"]


def test_set_scene_zero_is_noop(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    reader.set_scene(0)  # should not raise


def test_set_scene_out_of_range_raises(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    with pytest.raises(IndexError):
        reader.set_scene(1)


def test_xarray_dims_and_shape(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    xr_da = reader.xarray_dask_data
    assert xr_da.dims == ("T", "C", "Z", "Y", "X")
    assert xr_da.shape == (
        1,
        1,
        synthetic_smartspim.size_z,
        synthetic_smartspim.size_y,
        synthetic_smartspim.size_x,
    )
    assert xr_da.dtype == np.uint16


def test_xarray_pixel_content_matches_source_slices(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    computed = reader.xarray_dask_data.data.compute()
    for z in range(synthetic_smartspim.size_z):
        plane = computed[0, 0, z, :, :]
        assert plane.shape == (synthetic_smartspim.size_y, synthetic_smartspim.size_x)
        assert (plane == synthetic_smartspim.value_for(z)).all()


def test_physical_pixel_sizes_match_sidecar(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    pps = reader.physical_pixel_sizes
    assert pps.X == pytest.approx(synthetic_smartspim.xy_pixel_size_um)
    assert pps.Y == pytest.approx(synthetic_smartspim.xy_pixel_size_um)
    assert pps.Z == pytest.approx(synthetic_smartspim.z_step_um)


def test_metadata_returns_raw_sidecar_text(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    on_disk = synthetic_smartspim.metadata_path.read_bytes().decode("latin-1")
    assert reader.metadata == on_disk


def test_dtype_reflects_slice_dtype(synthetic_smartspim) -> None:
    # zarrmony >=0.9 reads reader.dtype to compute the OME-NGFF display
    # window; must match the on-disk .tif dtype without materializing the
    # dask graph.
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    assert reader.dtype == np.uint16


def test_selects_first_channel_dir_alphabetically(tmp_path: Path) -> None:
    # A three-channel fixture exposes exactly the tracer-bullet scope:
    # one scene per export, sourced from the alphabetically-first stitched
    # channel dir. Multi-channel semantics arrive in slice #2.
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3), (639, 4)),
    )
    reader = SmartSpimReader(fixture.export_dir)
    # Content of Ch1 (first alphabetically) — value_for(z, channel_index=0).
    computed = reader.xarray_dask_data.data.compute()
    for z in range(fixture.size_z):
        assert (computed[0, 0, z] == fixture.value_for(z, channel_index=0)).all()


def test_missing_metadata_sidecar_raises_actionable_error(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(tmp_path)
    fixture.metadata_path.unlink()
    with pytest.raises(SmartSpimMetadataError, match="copy the SmartSPIM metadata JSON"):
        SmartSpimReader(fixture.export_dir)


def test_empty_channel_dir_raises(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(tmp_path)
    for tif in fixture.channel_dirs[0].glob("*.tif"):
        tif.unlink()
    with pytest.raises(SmartSpimDataError, match="no .tif files"):
        SmartSpimReader(fixture.export_dir)


def test_no_stitched_channel_dir_raises(tmp_path: Path) -> None:
    # If someone calls SmartSpimReader directly on a dir the matcher wouldn't
    # have accepted, we should still fail loudly rather than proceed with a
    # phantom empty stack.
    d = tmp_path / "not_smartspim"
    d.mkdir()
    (d / "some_other_dir").mkdir()
    with pytest.raises(SmartSpimDataError, match="no Ex_"):
        SmartSpimReader(d)
