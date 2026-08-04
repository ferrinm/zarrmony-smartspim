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
