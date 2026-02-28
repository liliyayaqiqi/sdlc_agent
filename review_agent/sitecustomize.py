"""Test environment defaults for this repository."""

from __future__ import annotations

import os


# Keep local and CI test runs hermetic unless callers explicitly opt in.
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
