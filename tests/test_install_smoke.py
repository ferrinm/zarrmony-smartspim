"""Smoke test: confirm the plugin is discoverable through zarrmony's entry-point lookup.

Mirrors what an end user does after ``pip install zarrmony-smartspim``:

    from zarrmony.readers.plugin import list_plugins
    [p.name for p in list_plugins()]  # -> [..., 'zarrmony-smartspim']

If this test passes in CI (which runs ``uv pip install -e ".[dev]"`` from a
fresh venv), the entry-point declaration in ``pyproject.toml`` is wired up.
"""

from __future__ import annotations

from pathlib import Path

from zarrmony.readers.plugin import get_reader, list_plugins

from tests.conftest import write_synthetic_smartspim


def test_plugin_registered_via_entry_point() -> None:
    plugins = {p.name: p for p in list_plugins()}
    assert "zarrmony-smartspim" in plugins, (
        "zarrmony-smartspim did not appear in list_plugins(); check that "
        '[project.entry-points."zarrmony.readers"] in pyproject.toml is intact '
        "and that the package was installed (pip install -e .)."
    )


def test_registered_plugin_carries_expected_provenance() -> None:
    plugins = {p.name: p for p in list_plugins()}
    p = plugins["zarrmony-smartspim"]
    assert p.distribution == "zarrmony-smartspim"
    assert p.source == "entry_point"


def test_plugin_open_forwards_metadata_path_kwarg(tmp_path: Path) -> None:
    """End-to-end: reader_kwargs={"metadata_path": ...} on zarrmony's
    ``get_reader()`` reaches ``SmartSpimReader.__init__`` via the plugin's
    ``_open()`` — the gap that made zarrmony 0.13's passthrough surface a
    TypeError against the v0.2.0 plugin.
    """
    fixture = write_synthetic_smartspim(tmp_path)
    external = tmp_path / "elsewhere" / "sidecar.json"
    external.parent.mkdir()
    external.write_bytes(fixture.metadata_path.read_bytes())
    fixture.metadata_path.unlink()  # export dir now has no sidecar of its own

    reader, plugin, _score = get_reader(
        fixture.export_dir,
        reader_kwargs={"metadata_path": external},
    )
    assert plugin.name == "zarrmony-smartspim"
    assert reader._metadata_path == external
