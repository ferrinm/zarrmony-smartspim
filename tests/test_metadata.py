"""Metadata parser tests.

The SmartSPIM sidecar is Latin-1 encoded because the vendor's Windows PC
writes a raw ``0xB5`` byte for the µ sign in ``µm/pix``. These tests keep
that behavior pinned — if someone accidentally switches to ``utf-8``,
``UnicodeDecodeError`` on real files would only surface at customer sites.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import write_synthetic_smartspim
from zarrmony_smartspim._metadata import (
    SmartSpimMetadataError,
    find_metadata_file,
    parse_metadata_file,
)


def test_parse_synthetic_metadata_extracts_pixel_sizes(synthetic_smartspim) -> None:
    meta = parse_metadata_file(synthetic_smartspim.metadata_path)
    assert meta.xy_pixel_size_um == pytest.approx(synthetic_smartspim.xy_pixel_size_um)
    assert meta.z_step_um == pytest.approx(synthetic_smartspim.z_step_um)


def test_parse_preserves_raw_dict_and_text(synthetic_smartspim) -> None:
    meta = parse_metadata_file(synthetic_smartspim.metadata_path)
    assert "session_config" in meta.raw
    # raw_text is what will be written into OME/source/raw.smartspim.json —
    # it must match the file byte-for-byte after latin-1 decode.
    on_disk = synthetic_smartspim.metadata_path.read_bytes().decode("latin-1")
    assert meta.raw_text == on_disk


def test_latin_1_metadata_survives_micro_sign(tmp_path: Path) -> None:
    # Explicit reproduction of the real-world encoding: a lone 0xB5 byte for µ.
    path = tmp_path / "metadata_latin1.json"
    payload = '{"session_config": {"\xb5m/pix": "1.800", "z_step_um": "2.00"}}'
    path.write_bytes(payload.encode("latin-1"))
    meta = parse_metadata_file(path)
    assert meta.xy_pixel_size_um == pytest.approx(1.800)
    assert meta.z_step_um == pytest.approx(2.00)


def test_missing_session_config_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata_bad.json"
    path.write_text(json.dumps({"tile_config": {}}))
    with pytest.raises(SmartSpimMetadataError, match="session_config"):
        parse_metadata_file(path)


def test_missing_xy_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata_bad.json"
    path.write_text(json.dumps({"session_config": {"z_step_um": "2.0"}}))
    with pytest.raises(SmartSpimMetadataError, match="µm/pix"):
        parse_metadata_file(path)


def test_missing_z_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata_bad.json"
    # Write with latin-1 so the µm/pix key round-trips as a lone 0xB5.
    payload = '{"session_config": {"\xb5m/pix": "1.8"}}'
    path.write_bytes(payload.encode("latin-1"))
    with pytest.raises(SmartSpimMetadataError, match="z_step_um"):
        parse_metadata_file(path)


def test_non_numeric_pixel_size_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata_bad.json"
    payload = '{"session_config": {"\xb5m/pix": "wat", "z_step_um": "2.0"}}'
    path.write_bytes(payload.encode("latin-1"))
    with pytest.raises(SmartSpimMetadataError, match="is not a number"):
        parse_metadata_file(path)


def test_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "metadata_bad.json"
    path.write_text("{not: valid json}")
    with pytest.raises(SmartSpimMetadataError, match="not valid JSON"):
        parse_metadata_file(path)


def test_find_metadata_returns_matching_file(synthetic_smartspim) -> None:
    found = find_metadata_file(synthetic_smartspim.export_dir)
    assert found == synthetic_smartspim.metadata_path


def test_find_metadata_missing_raises(tmp_path: Path) -> None:
    empty = tmp_path / "export"
    empty.mkdir()
    with pytest.raises(SmartSpimMetadataError, match="no metadata"):
        find_metadata_file(empty)


def test_find_metadata_multiple_raises(tmp_path: Path) -> None:
    d = tmp_path / "export"
    d.mkdir()
    (d / "metadata_a.json").write_text("{}")
    (d / "metadata_b.json").write_text("{}")
    with pytest.raises(SmartSpimMetadataError, match="multiple"):
        find_metadata_file(d)


def test_wavelength_config_parsed_when_present(tmp_path: Path) -> None:
    fixture = write_synthetic_smartspim(
        tmp_path,
        channel_specs=((488, 1), (561, 3)),
        wavelength_config={
            "488": {"dye": "GFP", "emission_low_nm": 500, "emission_high_nm": 550},
            "561": {"dye": "mCherry", "emission_nm": 610},
        },
    )
    meta = parse_metadata_file(fixture.metadata_path)
    # Keys are cast to int so callers can look up by the excitation they
    # parsed out of the directory name.
    assert set(meta.wavelength_config) == {488, 561}
    assert meta.wavelength_config[488]["dye"] == "GFP"

    # A single ``emission_nm`` scalar becomes low==high per zarrmony#61's
    # uniform-band convention.
    identity = meta.channel_identity_for(561)
    assert identity.dye == "mCherry"
    assert identity.emission_low_nm == 610.0
    assert identity.emission_high_nm == 610.0


def test_wavelength_config_absent_yields_empty_map(synthetic_smartspim) -> None:
    meta = parse_metadata_file(synthetic_smartspim.metadata_path)
    assert meta.wavelength_config == {}
    identity = meta.channel_identity_for(488)
    # Only excitation is populated; other fields fall to None.
    assert identity.excitation_nm == 488
    assert identity.dye is None
    assert identity.fluor is None
    assert identity.emission_low_nm is None
