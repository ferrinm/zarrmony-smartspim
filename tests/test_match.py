"""Matcher unit tests. The matcher must be cheap and side-effect-free."""

from __future__ import annotations

from pathlib import Path

from zarrmony_smartspim.match import match


def test_matches_export_with_stitched_channel_dir(synthetic_smartspim) -> None:
    assert match(synthetic_smartspim.export_dir) == 100


def test_rejects_non_existent_path(tmp_path: Path) -> None:
    assert match(tmp_path / "nope") is None


def test_rejects_file_at_path(tmp_path: Path) -> None:
    p = tmp_path / "not_a_dir.txt"
    p.write_text("hello")
    assert match(p) is None


def test_rejects_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert match(empty) is None


def test_rejects_directory_with_only_mip_stitched_dirs(tmp_path: Path) -> None:
    # A MIP-only export (no full-res stitched channel dir) is not the target
    # of this plugin — the anchored regex ends in ``_stitched$``, so
    # ``Ex_488_Ch1_MIP_stitched`` must NOT match.
    d = tmp_path / "mip_only"
    (d / "Ex_488_Ch1_MIP_stitched").mkdir(parents=True)
    (d / "Ex_561_Ch3_MIP_stitched").mkdir()
    assert match(d) is None


def test_rejects_directory_with_only_unrelated_subdirs(tmp_path: Path) -> None:
    d = tmp_path / "unrelated"
    (d / "raw_data").mkdir(parents=True)
    (d / "notes").mkdir()
    assert match(d) is None


def test_case_sensitive_prefix(tmp_path: Path) -> None:
    # The vendor writes ``Ex_`` with a capital E — lowercase ``ex_`` should
    # not match, so we don't accidentally hijack unrelated directories.
    d = tmp_path / "wrong_case"
    (d / "ex_488_ch1_stitched").mkdir(parents=True)
    assert match(d) is None


def test_matches_when_mip_and_full_res_dirs_coexist(tmp_path: Path) -> None:
    # Real SmartSPIM exports have both — matcher fires because the full-res
    # dir is present.
    d = tmp_path / "mixed"
    (d / "Ex_488_Ch1_stitched").mkdir(parents=True)
    (d / "Ex_488_Ch1_MIP_stitched").mkdir()
    assert match(d) == 100
