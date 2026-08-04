"""End-to-end conversion: synthetic SmartSPIM export → OME-Zarr 0.5 store.

Uses the same registry-isolation shape as zarrmony-blaze/-phenix/-snouty:
snapshot ``_PLUGINS``, register just our plugin, run ``zarrmony.convert``,
then restore. Keeps the smoke independent of whatever other plugins may be
installed in the test environment.
"""

from __future__ import annotations

from pathlib import Path

import tifffile
import zarr
from ome_types import from_xml

from tests.conftest import write_synthetic_smartspim
from zarrmony_smartspim import plugin


def _convert_with_plugin(src: Path, out: Path) -> None:
    from zarrmony.api import convert
    from zarrmony.readers import plugin as plugin_mod

    snap_plugins = dict(plugin_mod._PLUGINS)
    snap_loaded = plugin_mod._ENTRY_POINTS_LOADED
    plugin_mod._PLUGINS.clear()
    plugin_mod._ENTRY_POINTS_LOADED = True
    try:
        plugin_mod.register_plugin(plugin)
        convert(str(src), str(out))
    finally:
        plugin_mod._PLUGINS.clear()
        plugin_mod._PLUGINS.update(snap_plugins)
        plugin_mod._ENTRY_POINTS_LOADED = snap_loaded


def test_convert_produces_valid_ome_zarr_store(synthetic_smartspim, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    _convert_with_plugin(synthetic_smartspim.export_dir, out_dir)

    stores = sorted(out_dir.glob("*.ome.zarr"))
    assert len(stores) == 1, f"expected one .ome.zarr, got {[s.name for s in stores]}"

    grp = zarr.open_group(str(stores[0]), mode="r")
    # OME-Zarr v0.5 multiscales puts the full-resolution level at key "0".
    arr = grp["0"]
    assert arr.shape == (
        1,
        1,
        synthetic_smartspim.size_z,
        synthetic_smartspim.size_y,
        synthetic_smartspim.size_x,
    )
    assert str(arr.dtype) == "uint16"


def test_convert_round_trips_source_pixel_values(synthetic_smartspim, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    _convert_with_plugin(synthetic_smartspim.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    arr = zarr.open_group(str(store), mode="r")["0"]

    for z in range(synthetic_smartspim.size_z):
        plane = arr[0, 0, z, :, :]
        expected = synthetic_smartspim.value_for(z)
        assert plane.shape == (synthetic_smartspim.size_y, synthetic_smartspim.size_x)
        assert (plane == expected).all(), (
            f"z={z} sampled voxel does not match source tif slice value {expected}"
        )


def test_convert_preserves_source_tif_pixels_verbatim(synthetic_smartspim, tmp_path: Path) -> None:
    # Direct source-vs-output equality — pin the pixel round-trip on the
    # actual on-disk .tif contents, not just on the fixture's generator.
    out_dir = tmp_path / "out"
    _convert_with_plugin(synthetic_smartspim.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    arr = zarr.open_group(str(store), mode="r")["0"]

    channel_dir = synthetic_smartspim.channel_dirs[0]
    source_slices = sorted(channel_dir.glob("*.tif"))
    for z, tif_path in enumerate(source_slices):
        on_disk = tifffile.imread(str(tif_path))
        assert (arr[0, 0, z, :, :] == on_disk).all(), (
            f"z={z}: OME-Zarr voxel plane differs from source {tif_path.name}"
        )


def test_convert_multichannel_writes_all_channels(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path / "in",
        channel_specs=((488, 1), (561, 3), (639, 4)),
        wavelength_config={
            "488": {"name": "GFP", "fluor": "GFP", "emission_nm": 525},
            "561": {"name": "mCherry", "fluor": "mCherry", "emission_nm": 610},
            "639": {"name": "Cy5", "fluor": "Cy5", "emission_nm": 670},
        },
    )
    out_dir = tmp_path / "out"
    _convert_with_plugin(fixture.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    arr = zarr.open_group(str(store), mode="r")["0"]
    assert arr.shape == (
        1,
        3,
        fixture.size_z,
        fixture.size_y,
        fixture.size_x,
    )
    # Verify each channel's pixel content survived stacking into C.
    for channel_index in range(3):
        for z in range(fixture.size_z):
            expected = fixture.value_for(z, channel_index=channel_index)
            assert (arr[0, channel_index, z, :, :] == expected).all(), f"C={channel_index} Z={z}"


def test_omero_channel_labels_match_reader_channel_names(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path / "in",
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {"name": "GFP", "fluor": "GFP", "emission_nm": 525},
            "561": {"name": "mCherry", "fluor": "mCherry", "emission_nm": 610},
        },
    )
    out_dir = tmp_path / "out"
    _convert_with_plugin(fixture.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    grp = zarr.open_group(str(store), mode="r")
    ome_attrs = grp.attrs["ome"]
    omero_labels = [c["label"] for c in ome_attrs["omero"]["channels"]]
    assert omero_labels == ["GFP", "mCherry"]


def test_per_scene_channels_projected_into_audit(tmp_path: Path) -> None:
    # zarrmony >=0.10 projects reader.ome_metadata.images[0].pixels.channels
    # into per_scene[i].channels using the ADR-0008 / #61 shape. The OME
    # <Channel> model has no `dye`, so the projection carries name/fluor/
    # excitation_nm/emission_low_nm/emission_high_nm; single OME
    # emission_wavelength collapses to low == high per #61's uniform-band
    # convention.
    fixture = write_synthetic_smartspim(
        tmp_path / "in",
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {"name": "GFP", "fluor": "GFP", "emission_nm": 525},
            "561": {"name": "mCherry", "fluor": "mCherry", "emission_nm": 610},
        },
    )
    out_dir = tmp_path / "out"
    _convert_with_plugin(fixture.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    audit = zarr.open_group(str(store), mode="r").attrs["zarrmony"]
    assert audit["audit_schema_version"] >= 8
    channels_audit = audit["per_scene"][0]["channels"]
    assert [c["index"] for c in channels_audit] == [0, 1]
    assert [c["name"] for c in channels_audit] == ["GFP", "mCherry"]
    assert [c["fluor"] for c in channels_audit] == ["GFP", "mCherry"]
    assert [c["excitation_nm"] for c in channels_audit] == [488, 561]
    assert [c["emission_low_nm"] for c in channels_audit] == [525, 610]
    assert [c["emission_high_nm"] for c in channels_audit] == [525, 610]


def test_instrument_audit_block_in_zarr_attrs(tmp_path: Path) -> None:
    # Populated sidecar → the ADR-0008 acquisition + objective blocks land
    # under ``attrs.zarrmony.per_scene[0]`` through zarrmony's OME extractor,
    # matching zarrmony#63–#65's shape byte-for-byte.
    fixture = write_synthetic_smartspim(
        tmp_path / "in",
        extra_session_config={"NA": "0.55", "model": "SmartSPIM XL", "Immersion": "Oil"},
        extra_top_level={"machine_id": "SN-4242", "date": "2025-06-15T09:30:00"},
    )
    out_dir = tmp_path / "out"
    _convert_with_plugin(fixture.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    audit = zarr.open_group(str(store), mode="r").attrs["zarrmony"]
    assert audit["audit_schema_version"] >= 8
    scene = audit["per_scene"][0]
    assert scene["acquisition"] == {
        "date": "2025-06-15T09:30:00",
        "microscope": "LifeCanvas SmartSPIM XL",
        "microscope_serial": "SN-4242",
    }
    assert scene["objective"] == {
        "nominal_magnification": 3.6,
        "numerical_aperture": 0.55,
        "immersion": "Oil",
        "model": "LCT 3.6x",
    }


def test_instrument_audit_degrades_gracefully_on_stripped_sidecar(tmp_path: Path) -> None:
    # A sidecar with no instrument fields yields the family-name fallback
    # (``microscope = "LifeCanvas SmartSPIM"``) and omits every other key
    # per the ADR-0008 omit-not-null rule — no exception, no placeholder junk.
    fixture = write_synthetic_smartspim(tmp_path / "in")
    out_dir = tmp_path / "out"
    _convert_with_plugin(fixture.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    scene = zarr.open_group(str(store), mode="r").attrs["zarrmony"]["per_scene"][0]
    assert scene["acquisition"] == {"microscope": "LifeCanvas SmartSPIM"}
    # Default fixture has obj_magnification + obj_name + refractive-index
    # immersion; those still surface. NA is absent (no key in fixture).
    assert scene["objective"] == {
        "nominal_magnification": 3.6,
        "immersion": "Other",
        "model": "LCT 3.6x",
    }


def test_inspect_surfaces_acquisition_block(tmp_path: Path) -> None:
    # ``zarrmony inspect`` should read the acquisition block off
    # ``reader.ome_metadata`` before conversion runs — same fields the audit
    # block will carry, minus the objective (inspect only surfaces acquisition).
    from zarrmony.api import inspect
    from zarrmony.readers import plugin as plugin_mod

    fixture = write_synthetic_smartspim(
        tmp_path,
        extra_session_config={"model": "SmartSPIM XL"},
        extra_top_level={"machine_id": "SN-1234", "date": "2025-06-15T09:30:00"},
    )

    snap_plugins = dict(plugin_mod._PLUGINS)
    snap_loaded = plugin_mod._ENTRY_POINTS_LOADED
    plugin_mod._PLUGINS.clear()
    plugin_mod._ENTRY_POINTS_LOADED = True
    try:
        plugin_mod.register_plugin(plugin)
        info = inspect(str(fixture.export_dir))
    finally:
        plugin_mod._PLUGINS.clear()
        plugin_mod._PLUGINS.update(snap_plugins)
        plugin_mod._ENTRY_POINTS_LOADED = snap_loaded

    scene = info["scenes"][0]
    assert scene["acquisition"] == {
        "date": "2025-06-15T09:30:00",
        "microscope": "LifeCanvas SmartSPIM XL",
        "microscope_serial": "SN-1234",
    }


def test_ome_xml_written_and_round_trips_to_ome(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path / "in",
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {"name": "GFP", "fluor": "GFP", "emission_nm": 525},
            "561": {"name": "mCherry", "fluor": "mCherry", "emission_nm": 610},
        },
    )
    out_dir = tmp_path / "out"
    _convert_with_plugin(fixture.export_dir, out_dir)

    store = next(out_dir.glob("*.ome.zarr"))
    xml_path = store / "OME" / "METADATA.ome.xml"
    assert xml_path.is_file(), (
        f"expected OME/METADATA.ome.xml under {store}, got contents: "
        f"{sorted(p.name for p in store.iterdir())}"
    )
    ome = from_xml(xml_path.read_text())
    assert len(ome.images) == 1
    channels = ome.images[0].pixels.channels
    assert [c.name for c in channels] == ["GFP", "mCherry"]
    assert [c.fluor for c in channels] == ["GFP", "mCherry"]
