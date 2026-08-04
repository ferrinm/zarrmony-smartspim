"""Synthetic SmartSPIM stitched-export fixture.

Writes the minimal on-disk shape the adapter reads: an export directory
containing (a) a Latin-1-encoded ``metadata_<sample>.json`` sidecar with
the ``session_config`` fields v0.1 consumes, and (b) one or more
``Ex_<λ>_Ch<N>_stitched/`` channel subdirs each holding N 2-D ``.tif``
Z-slices filled with distinct per-plane values so pixel round-trip asserts
can pin down Z-order.

The sidecar is written with ``latin-1`` on purpose: a real SmartSPIM PC
writes a raw ``0xB5`` byte for the ``µ`` in ``µm/pix``, and the adapter
reads with ``latin-1`` for that reason. Writing the fixture in UTF-8
would make the tests pass for the wrong reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile


@dataclass(frozen=True)
class SmartSpimFixture:
    export_dir: Path
    metadata_path: Path
    channel_dirs: list[Path]
    size_z: int
    size_y: int
    size_x: int
    xy_pixel_size_um: float
    z_step_um: float
    z_step_tenths_um: int  # e.g. 20 for z_step_um=2.0 (files stepped by 20)

    def value_for(self, z: int, channel_index: int = 0) -> int:
        # Distinct per-(channel, z) fill so tests can pin Z-order without
        # relying on all-zeros arrays. Values stay comfortably in uint16.
        return 1000 * (channel_index + 1) + z + 1


def write_synthetic_smartspim(
    root: Path,
    *,
    sample_id: str = "sample-a",
    size_z: int = 4,
    size_y: int = 6,
    size_x: int = 8,
    xy_pixel_size_um: float = 1.800,
    z_step_um: float = 2.00,
    channel_specs: tuple[tuple[int, int], ...] = ((488, 1),),
) -> SmartSpimFixture:
    """Write a synthetic SmartSPIM export under ``root``.

    ``channel_specs`` is a tuple of ``(excitation_wavelength, channel_index)``
    pairs — e.g. ``((488, 1), (561, 3), (639, 4))`` for a typical three-channel
    SmartSPIM export. v0.1's adapter only reads the first channel;
    multi-channel semantics arrive in slice #2.
    """
    export_dir = root / sample_id
    export_dir.mkdir(parents=True)

    channel_dirs: list[Path] = []
    z_step_tenths_um = int(round(z_step_um * 10))
    for ex_wavelength, ch_index in channel_specs:
        ch_dir = export_dir / f"Ex_{ex_wavelength}_Ch{ch_index}_stitched"
        ch_dir.mkdir()
        channel_dirs.append(ch_dir)
        # Also drop a sibling MIP dir so tests can prove the adapter's regex
        # skips it — real SmartSPIM exports ship one MIP per channel.
        (export_dir / f"Ex_{ex_wavelength}_Ch{ch_index}_MIP_stitched").mkdir()
        for z in range(size_z):
            z_stamp = z_step_tenths_um * z
            filename = f"000000_000000_{z_stamp:06d}_Ch{ch_index}.tif"
            plane = np.full(
                (size_y, size_x),
                _fixture_value(z, list(channel_specs).index((ex_wavelength, ch_index))),
                dtype=np.uint16,
            )
            tifffile.imwrite(ch_dir / filename, plane, photometric="minisblack")

    metadata = {
        "session_config": {
            "Blank": "Blank",
            "Immersion": "1.52+",
            "Version": "5.0.0.67",
            "obj_magnification": "3.600000",
            "obj_name": "LCT 3.6x",
            "sampling": "1/1",
            "scanning": "Fast",
            "v_res": "1600",
            "z_step_um": f"{z_step_um:.2f}",
            "µm/pix": f"{xy_pixel_size_um:.3f}",
        }
    }
    metadata_path = export_dir / f"metadata_{sample_id}.json"
    # latin-1 on purpose — see module docstring. json.dumps with
    # ensure_ascii=False lets the µ character survive; encoding to latin-1
    # then writes it as a single 0xB5 byte, matching real SmartSPIM output.
    metadata_path.write_bytes(json.dumps(metadata, ensure_ascii=False).encode("latin-1"))

    return SmartSpimFixture(
        export_dir=export_dir,
        metadata_path=metadata_path,
        channel_dirs=channel_dirs,
        size_z=size_z,
        size_y=size_y,
        size_x=size_x,
        xy_pixel_size_um=xy_pixel_size_um,
        z_step_um=z_step_um,
        z_step_tenths_um=z_step_tenths_um,
    )


def _fixture_value(z: int, channel_index: int) -> int:
    return 1000 * (channel_index + 1) + z + 1


@pytest.fixture
def synthetic_smartspim(tmp_path: Path) -> SmartSpimFixture:
    return write_synthetic_smartspim(tmp_path)
