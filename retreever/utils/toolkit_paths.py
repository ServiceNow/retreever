"""Resolve all on-disk paths for ReTreever from a single user-local file.

Paths come from ``local_paths.py`` at the repo root, which is gitignored so
machine-specific values never get committed. Copy ``local_paths.py.example``
to ``local_paths.py`` and edit it for your setup.
"""

try:
    from local_paths import DATA_PATHS, HF_CACHE_DIR
except ImportError as exc:  # pragma: no cover - hit on first install
    raise RuntimeError(
        "Could not import `local_paths`. Copy `local_paths.py.example` to "
        "`local_paths.py` at the repo root and fill in your machine-specific "
        "paths."
    ) from exc

# Re-export under the name the rest of the package historically used.
PATH_HF_CACHE_RW = HF_CACHE_DIR
