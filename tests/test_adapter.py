"""Adapter unit tests: Reader Protocol shape, pixel content, sidecar wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
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


def test_xarray_dims_and_shape_single_channel(synthetic_smartspim) -> None:
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


def test_stacks_all_channels_along_c_in_directory_order(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3), (639, 4)),
    )
    reader = SmartSpimReader(fixture.export_dir)
    xr_da = reader.xarray_dask_data
    assert xr_da.shape == (
        1,
        3,
        fixture.size_z,
        fixture.size_y,
        fixture.size_x,
    )
    computed = xr_da.data.compute()
    for channel_index in range(3):
        for z in range(fixture.size_z):
            assert (
                computed[0, channel_index, z] == fixture.value_for(z, channel_index=channel_index)
            ).all(), f"C={channel_index}, Z={z}"


def test_channel_names_derived_from_excitation_when_sidecar_silent(
    tmp_path: Path,
) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3)),
    )
    reader = SmartSpimReader(fixture.export_dir)
    # No wavelength_config → fall back to Ex<λ>.
    assert reader.channel_names == ["Ex488", "Ex561"]


def test_channel_names_pull_from_wavelength_config(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {"name": "GFP", "dye": "GFP", "fluor": "GFP"},
            "561": {"dye": "mCherry", "fluor": "mCherry"},  # falls back to dye
        },
    )
    reader = SmartSpimReader(fixture.export_dir)
    assert reader.channel_names == ["GFP", "mCherry"]


def test_channel_audit_shape_matches_zarrmony_61(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {
                "name": "GFP",
                "dye": "GFP",
                "fluor": "GFP",
                "emission_low_nm": 500,
                "emission_high_nm": 550,
            },
            "561": {"dye": "mCherry", "fluor": "mCherry", "emission_nm": 610},
        },
    )
    reader = SmartSpimReader(fixture.export_dir)
    audit = reader.channel_audit
    assert audit[0] == {
        "index": 0,
        "name": "GFP",
        "excitation_nm": 488,
        "dye": "GFP",
        "fluor": "GFP",
        "emission_low_nm": 500.0,
        "emission_high_nm": 550.0,
    }
    # Single ``emission_nm`` → low == high (uniform-band convention).
    assert audit[1] == {
        "index": 1,
        "name": "mCherry",
        "excitation_nm": 561,
        "dye": "mCherry",
        "fluor": "mCherry",
        "emission_low_nm": 610.0,
        "emission_high_nm": 610.0,
    }


def test_channel_audit_omits_missing_fields(tmp_path: Path) -> None:
    # Silent sidecar → only the fields the reader could actually determine
    # (index, name, excitation_nm) appear. ADR-0008 forbids null.
    fixture = write_synthetic_smartspim(tmp_path, channel_specs=((488, 1),))
    reader = SmartSpimReader(fixture.export_dir)
    audit = reader.channel_audit
    assert audit == [{"index": 0, "name": "Ex488", "excitation_nm": 488}]


def test_ome_metadata_returns_ome_object_with_channels(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {"name": "GFP", "fluor": "GFP", "emission_low_nm": 500, "emission_high_nm": 550},
            "561": {"name": "mCherry", "fluor": "mCherry", "emission_nm": 610},
        },
    )
    reader = SmartSpimReader(fixture.export_dir)
    ome = reader.ome_metadata
    assert len(ome.images) == 1
    pixels = ome.images[0].pixels
    assert pixels.size_c == 2
    assert pixels.size_z == fixture.size_z
    assert pixels.size_y == fixture.size_y
    assert pixels.size_x == fixture.size_x
    assert pixels.physical_size_x == pytest.approx(fixture.xy_pixel_size_um)
    assert pixels.physical_size_y == pytest.approx(fixture.xy_pixel_size_um)
    assert pixels.physical_size_z == pytest.approx(fixture.z_step_um)

    ch0, ch1 = pixels.channels
    assert ch0.name == "GFP"
    assert ch0.fluor == "GFP"
    assert ch0.excitation_wavelength == pytest.approx(488)
    # A band collapses to its low edge in OME's single-scalar model.
    assert ch0.emission_wavelength == pytest.approx(500)
    assert ch1.name == "mCherry"
    assert ch1.fluor == "mCherry"
    assert ch1.excitation_wavelength == pytest.approx(561)
    assert ch1.emission_wavelength == pytest.approx(610)


def test_ome_metadata_round_trips_through_xml(tmp_path: Path) -> None:
    from ome_types import from_xml

    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {"name": "GFP", "fluor": "GFP", "emission_nm": 525},
            "561": {"name": "mCherry", "fluor": "mCherry", "emission_nm": 610},
        },
    )
    reader = SmartSpimReader(fixture.export_dir)
    xml = reader.ome_metadata.to_xml()
    parsed = from_xml(xml)
    assert len(parsed.images) == 1
    assert [c.name for c in parsed.images[0].pixels.channels] == ["GFP", "mCherry"]
    assert [c.fluor for c in parsed.images[0].pixels.channels] == ["GFP", "mCherry"]


def test_ome_metadata_omits_fluor_when_sidecar_silent(synthetic_smartspim) -> None:
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    ch = reader.ome_metadata.images[0].pixels.channels[0]
    assert ch.fluor is None
    assert ch.emission_wavelength is None
    # Excitation always survives — it comes from the directory name.
    assert ch.excitation_wavelength == pytest.approx(488)


def test_missing_metadata_sidecar_raises_actionable_error(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(tmp_path)
    fixture.metadata_path.unlink()
    with pytest.raises(SmartSpimMetadataError, match="copy the SmartSPIM metadata JSON"):
        SmartSpimReader(fixture.export_dir)


def test_metadata_path_override_reads_sidecar_from_elsewhere(tmp_path: Path) -> None:
    # Read-only export dirs are the motivating case: the sidecar can live
    # anywhere the process can read, and physical_pixel_sizes / channel names
    # still resolve from it.
    fixture = write_synthetic_smartspim(tmp_path)
    external = tmp_path / "elsewhere" / "sidecar.json"
    external.parent.mkdir()
    external.write_bytes(fixture.metadata_path.read_bytes())
    fixture.metadata_path.unlink()

    reader = SmartSpimReader(fixture.export_dir, metadata_path=external)

    assert reader._metadata_path == external
    assert reader.physical_pixel_sizes.X == pytest.approx(fixture.xy_pixel_size_um)
    assert reader.physical_pixel_sizes.Z == pytest.approx(fixture.z_step_um)


def test_metadata_path_override_missing_file_raises(synthetic_smartspim, tmp_path: Path) -> None:
    with pytest.raises(SmartSpimMetadataError, match="does not exist or is not a file"):
        SmartSpimReader(
            synthetic_smartspim.export_dir,
            metadata_path=tmp_path / "nope.json",
        )


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
    (d / "metadata_x.json").write_bytes(
        b'{"session_config": {"\xb5m/pix": "1.0", "z_step_um": "1.0"}}'
    )
    with pytest.raises(SmartSpimDataError, match="no Ex_"):
        SmartSpimReader(d)


def test_channels_with_mismatched_z_depth_raises(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3)),
    )
    # Delete one slice from the second channel to induce a Z mismatch.
    doomed = sorted(fixture.channel_dirs[1].glob("*.tif"))[-1]
    doomed.unlink()
    with pytest.raises(SmartSpimDataError, match="Z-slices"):
        SmartSpimReader(fixture.export_dir)


def test_ome_metadata_carries_instrument_and_acquisition_date(tmp_path: Path) -> None:
    # ``reader.ome_metadata`` folds the parsed sidecar into an ``<Instrument>``
    # (Microscope + Objective) at the OME root plus an ``AcquisitionDate`` on
    # the Image, so zarrmony's ``extract_{objective,acquisition}_from_ome``
    # projects them straight into the audit block.
    fixture = write_synthetic_smartspim(
        tmp_path,
        extra_session_config={"NA": "0.55", "model": "SmartSPIM XL", "Immersion": "Oil"},
        extra_top_level={"machine_id": "SN-4242", "date": "2025-06-15T09:30:00"},
    )
    reader = SmartSpimReader(fixture.export_dir)
    ome = reader.ome_metadata
    assert len(ome.instruments) == 1
    inst = ome.instruments[0]
    assert inst.microscope.manufacturer == "LifeCanvas"
    assert inst.microscope.model == "SmartSPIM XL"
    assert inst.microscope.serial_number == "SN-4242"
    obj = inst.objectives[0]
    assert obj.nominal_magnification == pytest.approx(3.6)
    assert obj.lens_na == pytest.approx(0.55)
    assert obj.immersion.value == "Oil"
    assert obj.model == "LCT 3.6x"
    image = ome.images[0]
    assert image.acquisition_date is not None
    assert image.acquisition_date.isoformat() == "2025-06-15T09:30:00"


def test_ome_metadata_defaults_microscope_model_when_sidecar_silent(
    synthetic_smartspim,
) -> None:
    # A sidecar without a model still surfaces the family name ``SmartSPIM``
    # in the audit — we know the export was made on a LifeCanvas SmartSPIM
    # by construction (the plugin matcher only accepts SmartSPIM exports).
    reader = SmartSpimReader(synthetic_smartspim.export_dir)
    inst = reader.ome_metadata.instruments[0]
    assert inst.microscope.manufacturer == "LifeCanvas"
    assert inst.microscope.model == "SmartSPIM"
    assert inst.microscope.serial_number is None
    # Default fixture has obj_magnification + obj_name + immersion; those
    # still land on the Objective even when the sidecar is otherwise minimal.
    obj = inst.objectives[0]
    assert obj.nominal_magnification == pytest.approx(3.6)
    assert obj.model == "LCT 3.6x"
    # AcquisitionDate is absent when the sidecar didn't carry one — no
    # placeholder date is invented.
    assert reader.ome_metadata.images[0].acquisition_date is None


def test_ome_metadata_omits_objective_when_sidecar_carries_none(tmp_path: Path) -> None:
    # A sidecar stripped of every objective field: no magnification, no
    # model, no NA, no immersion. The Instrument still ships (for the
    # microscope) but the Objective is omitted — zarrmony's
    # ``extract_objective_from_ome`` returns ``None`` for such an OME, so the
    # audit's ``objective`` key is absent per the ADR-0008 omit-not-null rule.
    from pathlib import Path as _P

    path = _P(tmp_path) / "export" / "metadata_x.json"
    path.parent.mkdir()
    payload = '{"session_config": {"\xb5m/pix": "1.0", "z_step_um": "1.0"}}'
    path.write_bytes(payload.encode("latin-1"))
    (path.parent / "Ex_488_Ch1_stitched").mkdir()
    import numpy as _np
    import tifffile as _tf

    for z in range(2):
        _tf.imwrite(
            path.parent / "Ex_488_Ch1_stitched" / f"000000_000000_{z * 10:06d}_Ch1.tif",
            _np.zeros((4, 4), dtype=_np.uint16),
        )
    reader = SmartSpimReader(path.parent)
    inst = reader.ome_metadata.instruments[0]
    assert inst.objectives == []


def test_instrument_audit_shape_matches_adr_0008(tmp_path: Path) -> None:
    # ``reader.instrument_audit`` is the block projected in the ADR-0008 /
    # zarrmony#63–#65 shape — same key names + value types as the CZI / ND2 /
    # OME-TIFF readers surface via ``reader.ome_metadata`` extraction, plus
    # an ``imaging_method`` list which OME can't carry natively.
    fixture = write_synthetic_smartspim(
        tmp_path,
        extra_session_config={"NA": "0.55", "model": "SmartSPIM"},
        extra_top_level={"machine_id": "SN-4242", "date": "2025-06-15T09:30:00"},
    )
    reader = SmartSpimReader(fixture.export_dir)
    audit = reader.instrument_audit
    assert audit["acquisition"] == {
        "date": "2025-06-15T09:30:00",
        "microscope": "LifeCanvas SmartSPIM",
        "microscope_serial": "SN-4242",
        "imaging_method": ["light_sheet"],
    }
    assert audit["objective"] == {
        "nominal_magnification": 3.6,
        "numerical_aperture": 0.55,
        "immersion": "Other",
        "model": "LCT 3.6x",
    }


def test_instrument_audit_omits_missing_optional_fields(tmp_path: Path) -> None:
    # A stripped sidecar carries only what the reader could actually determine.
    # Only ``microscope`` (falls back to the family name) and ``imaging_method``
    # (a construction constant) are non-optional.
    path = tmp_path / "export" / "metadata_x.json"
    path.parent.mkdir()
    payload = '{"session_config": {"\xb5m/pix": "1.0", "z_step_um": "1.0"}}'
    path.write_bytes(payload.encode("latin-1"))
    (path.parent / "Ex_488_Ch1_stitched").mkdir()
    for z in range(2):
        tifffile.imwrite(
            path.parent / "Ex_488_Ch1_stitched" / f"000000_000000_{z * 10:06d}_Ch1.tif",
            np.zeros((4, 4), dtype=np.uint16),
        )
    reader = SmartSpimReader(path.parent)
    audit = reader.instrument_audit
    assert audit == {
        "acquisition": {
            "microscope": "LifeCanvas SmartSPIM",
            "imaging_method": ["light_sheet"],
        },
    }
