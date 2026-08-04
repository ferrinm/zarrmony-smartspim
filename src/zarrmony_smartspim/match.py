"""Cheap predicate that identifies a SmartSPIM stitched-export directory.

The marker is the presence of at least one immediate child directory whose
name matches ``Ex_<digits>_Ch<digits>_stitched`` (LifeCanvas's per-channel
stitched-output convention, e.g. ``Ex_488_Ch1_stitched``). MIP dirs like
``Ex_488_Ch1_MIP_stitched`` do NOT match — the anchored ``$`` in the regex
excludes them.

The matcher does no I/O beyond ``iterdir`` and early-returns on the first
match. It deliberately does NOT require the sidecar metadata JSON to be
colocated: real exports frequently have the JSON stored elsewhere (Google
Drive, project metadata folder), and the adapter surfaces a clear error if
the file cannot be found at ``open()`` time.
"""

from __future__ import annotations

import re
from pathlib import Path

_STITCHED_DIR_RE = re.compile(r"^Ex_\d+_Ch\d+_stitched$")


def match(path: Path) -> int | None:
    if not path.is_dir():
        return None
    for entry in path.iterdir():
        if entry.is_dir() and _STITCHED_DIR_RE.match(entry.name):
            return 100
    return None
